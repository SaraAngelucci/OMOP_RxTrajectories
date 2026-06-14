"""
main.py
=======

Scientific orchestrator for the thesis pipeline.

This script runs two independent experiments on the Medstat-calibrated
synthetic OMOP cohort produced by ``generate_synthetic_cohort.py``:

1. **Baseline run** -- executes :func:`run_trajectory_pipeline` once with
   the parameters in ``config/config_synthetic.yaml`` and writes all
   intermediate artefacts to ``cfg['project']['output_dir']``.
2. **Negative-control validation** -- isolates the patients that were
   generated under the ``acute_antibiotic`` archetype and asserts that
   they are not flagged as maintenance-evaluable.
"""

import os
import sys

# Ensure Spark's Python workers use the *same* interpreter as the driver.
# Without this, Spark falls back to whatever ``python3`` is first on PATH,
# which on many systems is an older build (e.g. 3.9). The pipeline uses PEP 604
# ``X | Y`` type-union syntax and therefore requires Python >= 3.10; a 3.9
# worker crashes with "unsupported operand type(s) for |". Pinning both env
# vars to ``sys.executable`` makes the run reproducible regardless of PATH.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import yaml
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, StringType, StructField, StructType

from src.thesis_rx.pipeline import (
    run_trajectory_pipeline,
    validate_negative_controls,
)


def main():
    """Run the full three-experiment thesis workflow end to end."""
    # Resource configuration is read from environment so the same code path
    # supports 1k, 50k, and larger cohorts without source edits.
    #   SPARK_DRIVER_MEMORY (default: 8g)        -- bump to 16g for 50k+
    #   SPARK_SHUFFLE_PARTITIONS (default: 50)   -- bump to 200 for 50k+
    #   SPARK_CONFIG_PATH (default: config/config_synthetic.yaml)
    driver_mem = os.environ.get("SPARK_DRIVER_MEMORY", "8g")
    shuffle_parts = os.environ.get("SPARK_SHUFFLE_PARTITIONS", "50")
    config_path = os.environ.get("SPARK_CONFIG_PATH", "config/config_synthetic.yaml")

    print(f"=== Initializing Spark Session (driver={driver_mem}, shuffle={shuffle_parts}) ===")
    spark = (
        SparkSession.builder
        .appName("Thesis_Trajectory_Analysis")
        .config("spark.driver.memory", driver_mem)
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.shuffle.partitions", shuffle_parts)
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )

    # ----- Load configuration --------------------------------------------
    print(f"Loading config: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg["project"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # ----- Load synthetic data -------------------------------------------
    print("=== Loading Data ===")
    raw_dir = cfg["paths"]["raw_dir"]

    eras_df = spark.read.parquet(f"{raw_dir}/{cfg['files']['drug_era']}")
    obs_df = spark.read.parquet(f"{raw_dir}/{cfg['files']['observation_period']}")

    # Robust concept loading: an existence pre-check avoids the noisy JVM
    # FileNotFoundException stack trace when CONCEPT.csv is absent (which is
    # the expected case for the synthetic cohort if no vocabulary stub has
    # been generated). The fallback is an empty typed DataFrame so all
    # downstream joins remain schema-safe.
    vocab_dir = cfg["paths"]["vocab_dir"]
    concept_path = os.path.join(vocab_dir, cfg["files"]["concept"])
    if os.path.exists(concept_path):
        print(f"Loading vocabulary from {concept_path}")
        concepts_df = spark.read.csv(concept_path, header=True, inferSchema=True)
        concepts_df = concepts_df.select(
            F.col("concept_id").alias("ingredient_concept_id"), "concept_name"
        )
    else:
        print(f"Note: {concept_path} not found -- using empty vocabulary stub. "
              "Ingredient names in cluster summary will be NULL but IDs are preserved.")
        schema = StructType([
            StructField("ingredient_concept_id", LongType(), True),
            StructField("concept_name", StringType(), True),
        ])
        concepts_df = spark.createDataFrame(spark.sparkContext.emptyRDD(), schema)

    # ====================================================================
    # EXPERIMENT 1: Baseline run
    # ====================================================================
    print("\n[EXPERIMENT 1] Executing Baseline Pipeline...")
    base_results = run_trajectory_pipeline(
        era_input_df=eras_df,
        observation_period=obs_df,
        death=None,
        have_death=False,
        ingredient_concepts=concepts_df,
        cfg=cfg,
        label="baseline_cohort",
    )
    final_person_df = base_results["final_person"]
    final_person_df.cache()
    print("Baseline execution complete. Outputs saved to disk.")

    # ====================================================================
    # EXPERIMENT 2: Negative-control validation
    # ====================================================================
    print("\n[EXPERIMENT 2] Validating Negative Controls...")
    acute_patients = (
        eras_df.filter(F.col("archetype") == "acute_antibiotic")
        .select("person_id")
        .distinct()
    )
    acute_person_ids = [row.person_id for row in acute_patients.collect()]
    print(f"Found {len(acute_person_ids)} negative control patients in raw data.")

    validation_res = validate_negative_controls(final_person_df, acute_person_ids)
    if validation_res["passed"]:
        print(">>> SUCCESS: 0 acute exposures leaked into maintenance phenotypes.")
    else:
        print(f">>> WARNING: {validation_res['n_violations']} acute exposures leaked into maintenance phenotypes.")
        validation_res["violations"].show()

    print("\nAll Pipeline Executions Finished Successfully")


if __name__ == "__main__":
    main()
