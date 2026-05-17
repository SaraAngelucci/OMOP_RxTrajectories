"""
main.py
=======

This script runs three independent experiments on the Medstat-calibrated
synthetic OMOP cohort produced by ``generate_synthetic_cohort.py``:

1. Baseline run: executes :func:`run_trajectory_pipeline` once with
   the parameters in ``config/config_synthetic.yaml`` and writes all
   intermediate artefacts to ``cfg['project']['output_dir']``.
2. Negative-control validation: isolates the patients that were
   generated under the ``acute_antibiotic`` archetype and asserts that
   they are not flagged as maintenance-evaluable.
3. Structural sensitivity grid: runs a 3x3 grid over
   ``polypharmacy_threshold`` and ``maintenance_min_total_days`` and
   computes the Adjusted Rand Index (ARI) between every pair of
   configurations. By default it fixes ``GRID_FIXED_K=2`` (fast); set
   ``GRID_SILHOUETTE_IN_GRID=1`` to re-select ``k`` with silhouette inside
   every cell.

All long-running Spark caches are released via
``spark.catalog.clearCache()`` between grid iterations to avoid JVM
``SparkOutOfMemoryError`` on machines with limited driver memory.
"""

import os
import uuid

import pandas as pd
import yaml
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, StringType, StructField, StructType

from src.thesis_rx.pipeline import (
    run_sensitivity_grid,
    run_trajectory_pipeline,
    validate_negative_controls,
)


def main():
    """Run the full three-experiment thesis workflow end to end."""
    # Resource configuration is read from environment so the same code path
    # supports 1k, 50k, and larger cohorts without source edits.
    #   SPARK_DRIVER_MEMORY (default: 8g)         or to 16g for 50k+
    #   SPARK_SHUFFLE_PARTITIONS (default: 50)    or to 200 for 50k+
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

    # Load configuration 
    print(f"Loading config: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg["project"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Load synthetic data 
    print("=== Loading Data ===")
    raw_dir = cfg["paths"]["raw_dir"]

    eras_df = spark.read.parquet(f"{raw_dir}/{cfg['files']['drug_era']}")
    obs_df = spark.read.parquet(f"{raw_dir}/{cfg['files']['observation_period']}")

    # pre-check avoids the FileNotFoundException stack trace when CONCEPT.csv is absent. The fallback is an empty typed DataFrame so all
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

   
    #EXPERIMENT 1: Baseline run
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

    
    #EXPERIMENT 2: Negative-control validation
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

   
    #EXPERIMENT 3: Structural sensitivity grid (ARI matrix)
    print("\n[EXPERIMENT 3] Running Structural Sensitivity Analysis (ARI Grid)...")

    param_grid = {
        "polypharmacy_threshold": [4, 5, 6],
        "maintenance_min_total_days": [14, 28, 56],
    }

    # Experiment 3: default is fixed K inside the grid (fast). Baseline always
    # runs the full silhouette grid (Experiment 1 calls run_trajectory_pipeline
    # with fixed_k omitted). Override with GRID_SILHOUETTE_IN_GRID=1 to re-evaluate
    # silhouette per grid cell (~9× the clustering cost).

    silhouette_in_grid = os.environ.get("GRID_SILHOUETTE_IN_GRID", "").strip().lower() in (
        "1", "true", "yes",
    )

    grid_fixed_k = (
        None
        if silhouette_in_grid else int(os.environ.get("GRID_FIXED_K", "2")))

    def pipeline_wrapper(current_cfg):
        """
        Wrap ``run_trajectory_pipeline`` for the grid orchestrator.

        Assigns a unique ``label``, suppresses Spark INFO logs, clears
        the catalog cache between cells, and threads ``fixed_k``: by default it
        is set from ``GRID_FIXED_K`` so silhouette evaluation is skipped inside
        the grid; set ``GRID_SILHOUETTE_IN_GRID`` to preserve full silhouette-based
        $K$ selection in every cell (~9-fold clustering cost increase).
        """
        run_id = f"grid_run_{uuid.uuid4().hex[:6]}"
        spark.sparkContext.setLogLevel("ERROR")

        res = run_trajectory_pipeline(
            era_input_df=eras_df,
            observation_period=obs_df,
            death=None,
            have_death=False,
            ingredient_concepts=concepts_df,
            cfg=current_cfg,
            label=run_id,
            fixed_k=None if silhouette_in_grid else grid_fixed_k,
        )
        spark.catalog.clearCache()
        return res

    configs_used, ari_matrix = run_sensitivity_grid(
        spark=spark,
        run_pipeline_fn=pipeline_wrapper,
        base_config=cfg,
        param_grid=param_grid,
    )

    config_labels = [
        f"Poly:{c['analysis']['polypharmacy_threshold']}_Maint:{c['analysis']['maintenance_min_total_days']}d"
        for c in configs_used
    ]
    ari_df = pd.DataFrame(ari_matrix, index=config_labels, columns=config_labels)
    ari_path = f"{out_dir}/ari_sensitivity_matrix.csv"
    ari_df.to_csv(ari_path)

    print("\n>>> SENSITIVITY GRID COMPLETE")
    print(f"ARI Matrix saved to: {ari_path}")
    print("\nARI Matrix Preview:")
    print(ari_df.round(3))

    print("\nAll Pipeline Executions Finished Successfully")


if __name__ == "__main__":
    main()
