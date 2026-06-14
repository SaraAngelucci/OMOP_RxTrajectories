"""
silhouette_grid_standalone.py
=============================

Stratified silhouette grid analysis on the pipeline's person-level features
parquet. Designed to be runnable after the production pipeline has fixed
``clustering.k_grid`` to a single value (e.g. ``[4]``) and the parquet
contains a single-K ``trajectory_cluster``; this script regenerates the
silhouette curve over ``k = k_min .. k_max`` without re-running the heavy
feature-engineering stages.

Mathematical contract
---------------------
* The clustering algorithm is the same Spark ML Lloyd K-means used by the
  pipeline; the feature representation is the same L2-normalised, cosine-
  equivalent representation produced by
  ``VectorAssembler → StandardScaler(withMean, withStd) → Normalizer(p=2)``.
* Only participants with ``trajectory_cluster >= 0`` are clustered. The
  pipeline assigns ``trajectory_cluster = -1`` to non-evaluable participants
  via the safe-failure path; their feature vectors are dominated by NoRx
  proportions and collapse onto a single point on the unit hypersphere,
  which both inflates the silhouette at ``k = 2`` (because the "evaluable
  vs non-evaluable" boolean is the largest axis of variance) and biases
  every other K-selection downstream. Filtering on ``trajectory_cluster >= 0``
  is therefore the mathematically appropriate input.
* Stratified silhouette sampling reproduces the pipeline:
  ``f_c = max(f_min, min(f_target * N / n_c, 1))`` with
  ``f_min = 0.05`` and ``f_target = 0.15`` for cohorts ≥ 20 000, full
  evaluation otherwise.

Usage
─────
    python silhouette_grid_standalone.py --dataset synthetic
    python silhouette_grid_standalone.py --dataset synthetic_50k
    python silhouette_grid_standalone.py --person_parquet outputs/synthetic/baseline_cohort_person_level_phenotypes.parquet \\
                                         --out_dir outputs/synthetic --label synthetic_1k
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, Normalizer
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


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
        .appName("silhouette_grid_standalone")
        .master("local[*]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def preprocess(spark: SparkSession, parquet_path: str):
    df = spark.read.parquet(parquet_path)
    n_total = df.count()
    if "trajectory_cluster" in df.columns:
        df = df.filter(F.col("trajectory_cluster") >= 0)
    elif "disc_evaluable" in df.columns:
        df = df.filter(F.col("disc_evaluable") == True)  # noqa: E712
    n_eval = df.count()
    print(f"  Total persons: {n_total:,};  evaluable: {n_eval:,}")

    present = [c for c in FEATURE_COLS if c in df.columns]
    df_present = df.select("person_id", *present)

    # Sanity check on nulls (should be zero for evaluable rows by construction)
    null_counts = df_present.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c) for c in present
    ]).collect()[0].asDict()
    if any(v and v > 0 for v in null_counts.values()):
        bad = {k: v for k, v in null_counts.items() if v and v > 0}
        print(f"  WARNING: nulls in evaluable rows: {bad}; imputing 0.")
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
    normalised = normalizer.transform(scaled)
    return normalised, n_eval


def stratified_silhouette(model_pred, evaluator, seed: int) -> float:
    """Stratified silhouette evaluation reproducing pipeline.py."""
    cluster_counts = model_pred.groupBy("trajectory_cluster").count().collect()
    total = sum(row["count"] for row in cluster_counts)
    target_sample_fraction = 1.0 if total < 20_000 else 0.15
    if target_sample_fraction == 1.0:
        sil_input = model_pred
    else:
        min_fraction = 0.05
        fractions = {
            row["trajectory_cluster"]: max(
                min_fraction,
                min(target_sample_fraction * total / row["count"], 1.0),
            )
            for row in cluster_counts
        }
        sil_input = model_pred.sampleBy(
            "trajectory_cluster", fractions=fractions, seed=seed,
        )
    return float(evaluator.evaluate(sil_input))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--person_parquet", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--k_min", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=6)
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
        args.out_dir = f"outputs/{args.dataset}" if args.dataset else "outputs"
    if args.label is None:
        args.label = args.dataset if args.dataset else "cohort"

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Starting Spark (driver memory: {args.spark_driver_memory}) ...")
    spark = get_spark(args.spark_driver_memory)
    spark.sparkContext.setLogLevel("WARN")

    print(f"\nLoading features from: {args.person_parquet}")
    df_norm, n_eval = preprocess(spark, args.person_parquet)
    df_norm.cache()
    df_norm.count()

    evaluator = ClusteringEvaluator(
        featuresCol="features_cosine",
        predictionCol="trajectory_cluster",
        metricName="silhouette",
        distanceMeasure="cosine",
    )

    records = []
    print(f"\nRunning Lloyd K-means silhouette grid for k = "
          f"{args.k_min} .. {args.k_max} ...")
    for k in range(args.k_min, args.k_max + 1):
        t0 = time.time()
        km = KMeans(
            k=k, seed=args.seed,
            featuresCol="features_cosine",
            predictionCol="trajectory_cluster",
        )
        model = km.fit(df_norm)
        pred = model.transform(df_norm)
        score = stratified_silhouette(pred, evaluator, args.seed)
        elapsed = time.time() - t0
        records.append({"label": args.label, "k": k, "silhouette": score})
        print(f"  k={k:2d}  silhouette={score:.4f}  ({elapsed:.0f}s)")

    out_csv = os.path.join(
        args.out_dir, f"silhouette_grid_{args.label}.csv"
    )
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"\nSilhouette grid CSV saved: {out_csv}")

    best = max(records, key=lambda r: r["silhouette"])
    print(f"\nSilhouette-best K: k = {best['k']}  (silhouette = {best['silhouette']:.4f})")
    print("Use this CSV with make_plots.py --silhouette-csv to regenerate "
          "the silhouette-grid figure.")

    spark.stop()


if __name__ == "__main__":
    main()
