"""
elbow_50k_standalone.py
────────────────────────
DEPRECATED — superseded by ``run_elbow.py``.

This wrapper is preserved for backward compatibility with the original 50k
notebook. ``run_elbow.py`` is the canonical elbow analysis script for the
thesis: it accepts either ``--dataset`` (resolved to the default outputs path)
or an explicit ``--person_parquet``, applies the documented filter
(``trajectory_cluster >= 0``), and uses the kneedle algorithm to locate the
elbow.

For the 50k cohort, the recommended invocation is now:

    python run_elbow.py --dataset synthetic_50k --max_k 10 \
                        --spark_driver_memory 12g

This standalone script delegates to the same code path and is mathematically
identical to ``run_elbow.py --dataset synthetic_50k``.
"""

import argparse
import os
import sys
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── PySpark imports ──────────────────────────────────────────────────────────
try:
    from pyspark.sql import SparkSession
    from pyspark.ml.clustering import BisectingKMeans
    from pyspark.ml.feature import VectorAssembler, StandardScaler, Normalizer
    from pyspark.sql import functions as F
    HAS_PYSPARK = True
except ImportError:
    HAS_PYSPARK = False


# ── feature columns (CORRECTED to match parquet) ───────────
FEATURE_COLS = [
    "mean_active_n", "poly_month_prop", "mean_turnover", "burden_slope",
    "n_ingredient_eras", "n_distinct_ingredients", "early_disc_90_rate", 
    "restart_180_rate", "switch_60_rate", "median_era_days",
    "mean_burden_w1", "mean_burden_w2", "mean_burden_w3", "mean_burden_w4",
    "prop_NoRx", "prop_Initiation", "prop_StableMono", "prop_StableLowPoly",
    "prop_StablePolypharmacy", "prop_Intensifying", "prop_Deintensifying",
    "prop_HighTurnover", "prop_ModerateFlux"
]


def get_spark(driver_memory: str) -> "SparkSession":
    return (
        SparkSession.builder
        .appName("elbow_50k_standalone")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def load_features_spark(spark, parquet_path: str, feature_cols: list[str]):
    df = spark.read.parquet(parquet_path)
    # drop unassigned rows (cluster = -1 from safe-failure mode)
    if "trajectory_cluster" in df.columns:
        df = df.filter(F.col("trajectory_cluster") >= 0)

    # keep only columns that exist
    present = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} feature columns missing from parquet: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    print(f"  Using {len(present)} features · {df.count():,} patients")

    # assemble → standardise → L2-normalise 
    assembler = VectorAssembler(inputCols=present, outputCol="_raw_vec", handleInvalid="keep")
    scaler = StandardScaler(inputCol="_raw_vec", outputCol="_scaled_vec", withMean=True, withStd=True)
    normalizer = Normalizer(inputCol="_scaled_vec", outputCol="features", p=2.0)

    df_assembled = assembler.transform(df.na.fill(0))
    df_scaled = scaler.fit(df_assembled).transform(df_assembled)
    df_norm = normalizer.transform(df_scaled).select("person_id", "features")
    
    return df_norm, present


def run_elbow(df_norm, max_k: int, seed: int = 42) -> dict[int, float]:
    """Run BisectingKMeans for k=2..max_k, return {k: wssse}."""
    wssse = {}
    for k in range(2, max_k + 1):
        t0 = time.time()
        bkm = BisectingKMeans(
            featuresCol="features",
            predictionCol="_cluster",
            k=k,
            seed=seed,
            maxIter=20,
        )
        model = bkm.fit(df_norm)
        wssse[k] = model.summary.trainingCost
        elapsed = time.time() - t0
        print(f"  k={k:2d}  WSSSE={wssse[k]:,.1f}  ({elapsed:.0f}s)")
    return wssse


def kneedle_elbow(ks: list[int], vals: list[float]) -> int:
    """
    Kneedle elbow detector for a monotone-decreasing convex WSSSE curve.

    Both axes are min-max scaled to [0, 1]. The elbow is the k whose
    distance below the chord (0, 1)->(1, 0) is greatest, i.e.
    argmax_k {1 - x_norm_k - y_norm_k}.

    NOTE: This replaces the previous "largest WSSSE drop" heuristic, which
    is mathematically invalid for a monotone-convex curve (it always selects
    the smallest k in the grid, because the first difference is itself
    monotone-decreasing).
    """
    arr_k = np.asarray(ks, dtype=float)
    arr_v = np.asarray(vals, dtype=float)
    x_norm = (arr_k - arr_k.min()) / (arr_k.max() - arr_k.min())
    y_norm = (arr_v - arr_v.min()) / (arr_v.max() - arr_v.min())
    distance_below_chord = 1.0 - x_norm - y_norm
    return int(ks[int(np.argmax(distance_below_chord))])


def plot_elbow(wssse: dict[int, float], out_path: str, label: str = "cohort") -> None:
    ks = sorted(wssse)
    vals = [wssse[k] for k in ks]
    deltas = [vals[i - 1] - vals[i] for i in range(1, len(vals))]
    elbow_k = kneedle_elbow(ks, vals)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # left: WSSSE curve
    axes[0].plot(ks, vals, "o-", color="#1f77b4", lw=2, ms=7)
    for k, v in zip(ks, vals):
        axes[0].annotate(f"{v:.0f}", (k, v),
                         textcoords="offset points", xytext=(4, 4), fontsize=7)
    axes[0].axvline(elbow_k, color="#c53030", lw=1.4, ls="--",
                    label=f"Kneedle elbow at k={elbow_k}")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("WSSSE")
    axes[0].set_title(f"Bisecting K-means elbow ({label}, L2-normalised, "
                      "evaluable only)")
    axes[0].set_xticks(ks)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9, loc="upper right")

    # right: ΔWSSSE
    axes[1].bar(ks[1:], deltas, color="#ff7f0e", alpha=0.75)
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("ΔWSSSE  (WSSSE_{k-1} − WSSSE_k)")
    axes[1].set_title("Rate of WSSSE decrease (descriptive only)")
    axes[1].set_xticks(ks[1:])
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].axvline(elbow_k, color="#c53030", lw=1.4, ls="--",
                    label=f"Kneedle elbow at k={elbow_k}")
    axes[1].legend(fontsize=9, loc="upper right")

    plt.suptitle(f"K selection via bisecting K-means elbow ({label})",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Elbow plot saved: {out_path}")


def main() -> None:
    if not HAS_PYSPARK:
        print("ERROR: PySpark not available. "
              "Run this script in your PySpark environment.")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--person_parquet",    required=True,
                        help="Path to baseline_cohort_person_level_phenotypes.parquet")
    parser.add_argument("--out_dir",           default="outputs/synthetic_50k/elbow")
    parser.add_argument("--label",             default="cohort",
                        help="Cohort label for plot title (e.g. 'synthetic_50k').")
    parser.add_argument("--max_k",             type=int, default=10)
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--spark_driver_memory", default="8g")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Starting Spark (driver memory: {args.spark_driver_memory}) ...")
    spark = get_spark(args.spark_driver_memory)
    spark.sparkContext.setLogLevel("WARN")

    print(f"\nLoading features from:\n  {args.person_parquet}")
    df_norm, used_features = load_features_spark(
        spark, args.person_parquet, FEATURE_COLS
    )
    df_norm.cache()
    df_norm.count()   # materialise cache before timing loop

    print(f"\nRunning bisecting K-means for k=2..{args.max_k} ...")
    t_total = time.time()
    wssse = run_elbow(df_norm, args.max_k, args.seed)
    print(f"Total elbow runtime: {(time.time() - t_total) / 60:.1f} min")

    # save WSSSE table
    csv_path = os.path.join(args.out_dir, "elbow_wssse_50k.csv")
    pd.DataFrame(
        [{"k": k, "wssse": v} for k, v in wssse.items()]
    ).to_csv(csv_path, index=False)
    print(f"  WSSSE table saved: {csv_path}")

    plot_elbow(wssse, os.path.join(args.out_dir, f"elbow_plot_{args.label}.png"),
               label=args.label)

    # Identify the elbow via the kneedle algorithm (mathematically valid for
    # monotone-decreasing convex curves; replaces the previous "largest drop"
    # heuristic, which is degenerate at k_min).
    ks = sorted(wssse)
    vals = [wssse[k] for k in ks]
    elbow_k = kneedle_elbow(ks, vals)
    print(f"\nKneedle elbow:  K = {elbow_k}")
    print(f"  Visual inspection of the curve is still recommended; the kneedle\n"
          f"  estimator is a tool, not a definitive K-selection.")

    spark.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
