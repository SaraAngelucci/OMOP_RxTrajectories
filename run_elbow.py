"""
run_elbow.py
============

Bisecting K-means elbow analysis on the pipeline's person-level features.

This is the canonical elbow-analysis script for the thesis. It operates on
the parquet output of ``main.py`` (``<label>_person_level_phenotypes.parquet``)
and reproduces, end to end, the L2-normalised cosine-equivalent preprocessing
used inside ``run_trajectory_pipeline``.

Mathematical contract
---------------------
* Only participants with ``trajectory_cluster >= 0`` are clustered. The
  pipeline assigns ``trajectory_cluster = -1`` to non-evaluable participants
  via the safe-failure path (``disc_evaluable = False``); including them in
  the elbow analysis collapses an artificial cluster of near-zero feature
  vectors at low *k*, which produces a spurious large drop at *k = 2..3* and
  biases the elbow estimator. Filtering on ``trajectory_cluster >= 0`` is
  therefore the only mathematically consistent input for the elbow.
* Preprocessing reproduces ``pipeline.py``:
  ``VectorAssembler → StandardScaler(withMean, withStd) → Normalizer(p=2)``.
  After L2 normalisation, squared Euclidean distance equals
  ``2 * (1 - cos(theta))``, so WSSSE is bounded above by ``4 * N`` and is
  directly comparable across cohorts after dividing by ``N``.
* The "elbow" is identified by the **kneedle algorithm**: both axes are
  min-max scaled to [0, 1], a chord is drawn from the first to the last
  point, and the elbow is the *k* whose distance below the chord is
  greatest. Equivalently, ``argmax_k {1 - x_norm_k - y_norm_k}``. This
  estimator is robust to the monotone-decreasing convex shape of the WSSSE
  curve, where naive ``argmax(first-difference)`` always degenerates to the
  smallest *k* in the grid.
* The second-difference (``WSSSE_{k-1} - 2 WSSSE_k + WSSSE_{k+1}``) is also
  reported as a secondary diagnostic.

Usage
-----
    python run_elbow.py --dataset synthetic
    python run_elbow.py --dataset synthetic_50k --max_k 10
    python run_elbow.py --person_parquet outputs/synthetic/baseline_cohort_person_level_phenotypes.parquet \
                        --out_dir outputs/synthetic/elbow --label synthetic_1k

The default ``--dataset`` resolves to
``outputs/<dataset>/baseline_cohort_person_level_phenotypes.parquet``.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, Normalizer
from pyspark.ml.clustering import BisectingKMeans


# 23 feature columns produced by run_trajectory_pipeline()
FEATURE_COLS = [
    "mean_active_n", "poly_month_prop", "mean_turnover", "burden_slope",
    "n_ingredient_eras", "n_distinct_ingredients",
    "early_disc_90_rate", "restart_180_rate", "switch_60_rate", "median_era_days",
    "mean_burden_w1", "mean_burden_w2", "mean_burden_w3", "mean_burden_w4",
    "prop_NoRx", "prop_Initiation", "prop_StableMono", "prop_StableLowPoly",
    "prop_StablePolypharmacy", "prop_Intensifying", "prop_Deintensifying",
    "prop_HighTurnover", "prop_ModerateFlux",
]


def get_spark(driver_memory: str) -> SparkSession:
    return (
        SparkSession.builder
        .appName("run_elbow")
        .master("local[*]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def load_and_preprocess(spark: SparkSession, parquet_path: str):
    """
    Read the person-level parquet, filter to evaluable participants, then apply
    the exact preprocessing used by run_trajectory_pipeline:
    StandardScaler(withMean, withStd) followed by L2 Normalizer(p=2).
    """
    df = spark.read.parquet(parquet_path)

    n_total = df.count()
    if "trajectory_cluster" in df.columns:
        df = df.filter(F.col("trajectory_cluster") >= 0)
    elif "disc_evaluable" in df.columns:
        df = df.filter(F.col("disc_evaluable") == True)  # noqa: E712
    n_eval = df.count()
    print(f"  Total persons in parquet: {n_total:,}")
    print(f"  Evaluable persons used:   {n_eval:,}")

    present = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} feature columns absent from parquet: {missing}")

    # No additional fillna(0) here. The pipeline's writer already fills the
    # global summaries; the discontinuation-rate columns are populated for
    # every evaluable row by construction (disc_summary aggregates only
    # maintenance_eligible == 1 eras, then is left-joined to the eligible set
    # which is filtered to evaluable above).
    df_present = df.select("person_id", *present)

    # ----- Sanity check: assert no NaNs/Nulls remain ------------------------
    null_counts = df_present.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c) for c in present
    ]).collect()[0].asDict()
    bad_cols = {c: n for c, n in null_counts.items() if n and n > 0}
    if bad_cols:
        # Defensive imputation, but warn loudly: this should not happen
        # for properly-generated pipeline outputs.
        print(f"  WARNING: NaN/Null values found in evaluable rows: {bad_cols}. "
              "Imputing 0 (zero-after-standardisation = -mean/sigma, "
              "which is non-zero and biases the standardised feature; "
              "this should not happen for a correctly-generated parquet).")
        df_present = df_present.na.fill(0)

    assembler = VectorAssembler(
        inputCols=present, outputCol="features_raw", handleInvalid="keep")
    scaler = StandardScaler(
        inputCol="features_raw", outputCol="features_scaled",
        withMean=True, withStd=True)
    normalizer = Normalizer(
        inputCol="features_scaled", outputCol="features_cosine", p=2.0)

    assembled = assembler.transform(df_present)
    scaled = scaler.fit(assembled).transform(assembled)
    normalised = normalizer.transform(scaled).select("person_id", "features_cosine")

    return normalised, n_eval, present


def fit_bisecting_kmeans_grid(df_norm, max_k: int, seed: int = 42) -> dict:
    """Fit BisectingKMeans for k = 2..max_k and return {k: WSSSE}."""
    wssse = {}
    for k in range(2, max_k + 1):
        t0 = time.time()
        bkm = BisectingKMeans(
            featuresCol="features_cosine",
            predictionCol="_cluster",
            k=k, seed=seed, maxIter=20,
        )
        model = bkm.fit(df_norm)
        wssse[k] = float(model.summary.trainingCost)
        print(f"  k={k:2d}  WSSSE={wssse[k]:,.4f}  ({time.time() - t0:.0f}s)")
    return wssse


def kneedle_elbow(ks: list[int], vals: list[float]) -> int:
    """
    Kneedle elbow detector for a decreasing convex curve.

    Both axes are min-max-scaled to [0, 1]. The elbow is the k whose
    distance below the chord (0, 1) -> (1, 0) is greatest, i.e.
    argmax_k {1 - x_norm_k - y_norm_k}.

    Returns the k in ``ks`` with the largest such distance.
    """
    arr_k = np.asarray(ks, dtype=float)
    arr_v = np.asarray(vals, dtype=float)
    x_norm = (arr_k - arr_k.min()) / (arr_k.max() - arr_k.min())
    y_norm = (arr_v - arr_v.min()) / (arr_v.max() - arr_v.min())
    distance_below_chord = 1.0 - x_norm - y_norm
    elbow_idx = int(np.argmax(distance_below_chord))
    return int(ks[elbow_idx])


def second_differences(ks: list[int], vals: list[float]) -> dict[int, float]:
    """Return {k: WSSSE_{k-1} - 2*WSSSE_k + WSSSE_{k+1}} for interior k."""
    out = {}
    for i in range(1, len(ks) - 1):
        out[ks[i]] = vals[i - 1] - 2 * vals[i] + vals[i + 1]
    return out


def plot_elbow(wssse: dict, n_eval: int, label: str, out_path: str) -> None:
    ks = sorted(wssse.keys())
    vals = [wssse[k] for k in ks]
    first_diff = [vals[i - 1] - vals[i] for i in range(1, len(vals))]
    elbow_k = kneedle_elbow(ks, vals)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: WSSSE curve
    axes[0].plot(ks, vals, "o-", color="#1f77b4", lw=2, ms=7)
    for k, v in zip(ks, vals):
        axes[0].annotate(f"{v:.0f}", (k, v),
                         textcoords="offset points", xytext=(5, 5), fontsize=8)
    axes[0].axvline(elbow_k, color="#c53030", lw=1.4, ls="--",
                    label=f"Kneedle elbow at k={elbow_k}")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("WSSSE")
    axes[0].set_title(f"Bisecting K-means elbow — {label}\n"
                      f"L2-normalised features, n={n_eval:,} evaluable")
    axes[0].set_xticks(ks)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9, loc="upper right")

    # Panel B: First-difference (rate of WSSSE decrease)
    axes[1].bar(ks[1:], first_diff, color="#ff7f0e", alpha=0.8)
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("ΔWSSSE  (WSSSE_{k-1} − WSSSE_k)")
    axes[1].set_title("Rate of WSSSE decrease")
    axes[1].set_xticks(ks[1:])
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].axvline(elbow_k, color="#c53030", lw=1.4, ls="--",
                    label=f"Kneedle elbow at k={elbow_k}")
    axes[1].legend(fontsize=9, loc="upper right")

    plt.suptitle(f"K-selection via bisecting K-means elbow ({label})",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Elbow plot saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default=None,
        help="Convenience flag: resolves to "
             "outputs/<dataset>/baseline_cohort_person_level_phenotypes.parquet "
             "and outputs/<dataset>/elbow/ as out_dir. "
             "Examples: 'synthetic', 'synthetic_50k', 'external_eunomia'.")
    parser.add_argument(
        "--person_parquet", default=None,
        help="Explicit parquet path. Overrides --dataset.")
    parser.add_argument(
        "--out_dir", default=None,
        help="Output directory. Overrides --dataset.")
    parser.add_argument(
        "--label", default=None,
        help="Cohort label used in the plot title (e.g. 'synthetic_1k'). "
             "Defaults to --dataset.")
    parser.add_argument("--max_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spark_driver_memory", default="8g")
    args = parser.parse_args()

    if args.person_parquet is None and args.dataset is None:
        parser.error("Provide either --dataset or --person_parquet.")

    if args.person_parquet is None:
        args.person_parquet = (
            f"outputs/{args.dataset}/baseline_cohort_person_level_phenotypes.parquet"
        )
    if args.out_dir is None:
        args.out_dir = (
            f"outputs/{args.dataset}/elbow" if args.dataset
            else "outputs/elbow"
        )
    if args.label is None:
        args.label = args.dataset if args.dataset else "cohort"

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Starting Spark (driver memory: {args.spark_driver_memory}) ...")
    spark = get_spark(args.spark_driver_memory)
    spark.sparkContext.setLogLevel("WARN")

    print(f"\nLoading features from: {args.person_parquet}")
    df_norm, n_eval, used_features = load_and_preprocess(spark, args.person_parquet)
    df_norm.cache()
    df_norm.count()

    print(f"\nRunning bisecting K-means for k = 2 .. {args.max_k} ...")
    t_total = time.time()
    wssse = fit_bisecting_kmeans_grid(df_norm, args.max_k, args.seed)
    print(f"Total elbow runtime: {(time.time() - t_total) / 60:.1f} min")

    # Save WSSSE table
    ks_sorted = sorted(wssse.keys())
    vals_sorted = [wssse[k] for k in ks_sorted]
    csv_path = os.path.join(args.out_dir, f"elbow_wssse_{args.label}.csv")
    pd.DataFrame({"k": ks_sorted, "wssse": vals_sorted}).to_csv(csv_path, index=False)
    print(f"  WSSSE table saved: {csv_path}")

    # Report kneedle elbow and second differences
    elbow = kneedle_elbow(ks_sorted, vals_sorted)
    second_diff = second_differences(ks_sorted, vals_sorted)
    second_diff_csv = os.path.join(args.out_dir, f"elbow_second_diff_{args.label}.csv")
    pd.DataFrame(
        [{"k": k, "second_difference": v} for k, v in second_diff.items()]
    ).to_csv(second_diff_csv, index=False)
    print(f"  Second-difference table saved: {second_diff_csv}")
    print(f"\nKneedle elbow: k = {elbow}")
    if second_diff:
        sd_top = max(second_diff.items(), key=lambda kv: kv[1])
        print(f"Largest second difference: k = {sd_top[0]}  (Δ² = {sd_top[1]:.4f})")

    # Plot
    out_plot = os.path.join(args.out_dir, f"elbow_plot_{args.label}.png")
    plot_elbow(wssse, n_eval, args.label, out_plot)

    spark.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
