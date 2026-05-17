import sys
from pathlib import Path
import argparse
from pyspark.sql import SparkSession, functions as F

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thesis_rx.config   import load_config, ensure_output_dir, validate_config
from thesis_rx.io       import load_tables, standardize_tables
from thesis_rx.pipeline import (
    build_primary_eras,
    build_exposure_derived_eras,
    run_trajectory_pipeline
)


def main():
    parser = argparse.ArgumentParser(
        description="Run UKB OMOP prescription trajectory pipeline"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_config(cfg)          # fail fast on missing keys
    ensure_output_dir(cfg)

    spark = (
        SparkSession.builder
        .appName(cfg["project"]["name"])
        .getOrCreate()
    )

    raw_tables = load_tables(spark, cfg)
    tables     = standardize_tables(raw_tables)

    primary_eras    = build_primary_eras(tables["drug_era"])
    primary_results = run_trajectory_pipeline(
        era_input_df       = primary_eras,
        observation_period = tables["observation_period"],
        death              = tables["death"],
        have_death         = tables["have_death"],
        ingredient_concepts= tables["ingredient_concepts"],
        cfg                = cfg,
        label              = "drug_era_primary"
    )

    if cfg["run"]["run_exposure_sensitivity"]:
        exposure_eras = build_exposure_derived_eras(
            drug_exposure    = tables["drug_exposure"],
            concept          = tables["concept"],
            concept_ancestor = tables["concept_ancestor"],
            cfg              = cfg
        )
        exposure_results = run_trajectory_pipeline(
            era_input_df       = exposure_eras,
            observation_period = tables["observation_period"],
            death              = tables["death"],
            have_death         = tables["have_death"],
            ingredient_concepts= tables["ingredient_concepts"],
            cfg                = cfg,
            label              = "drug_exposure_sensitivity"
        )

        primary_small = primary_results["final_person"].selectExpr(
            "person_id as pid",
            "trajectory_cluster   as cluster_primary",
            "discontinuation_phenotype as phenotype_primary",
            "mean_active_n        as mean_active_n_primary",
            "mean_turnover        as mean_turnover_primary",
            "early_disc_90_rate   as early_disc_90_rate_primary"
        )
        exposure_small = exposure_results["final_person"].selectExpr(
            "person_id as pid",
            "trajectory_cluster   as cluster_exposure",
            "discontinuation_phenotype as phenotype_exposure",
            "mean_active_n        as mean_active_n_exposure",
            "mean_turnover        as mean_turnover_exposure",
            "early_disc_90_rate   as early_disc_90_rate_exposure"
        )

        comparison = (
            primary_small
            .join(exposure_small, on="pid", how="inner")
            # FIX: use F.col() not DataFrame["col"] to avoid post-join ambiguity
            .withColumn(
                "same_phenotype",
                (F.col("phenotype_primary") == F.col("phenotype_exposure")).cast("int")
            )
        )
        comparison.write.mode("overwrite").parquet(
            f"{cfg['project']['output_dir']}/primary_vs_exposure_sensitivity.parquet"
        )

    spark.stop()


if __name__ == "__main__":
    main()
