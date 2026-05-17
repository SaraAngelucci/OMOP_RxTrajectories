from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F


def _csv_read_options(cfg: dict):
    return {
        "header": cfg["csv"]["header"],
        "inferSchema": cfg["csv"]["inferSchema"]
    }


def load_tables(spark: SparkSession, cfg: dict) -> dict:
    raw_dir = cfg["paths"]["raw_dir"]
    vocab_dir = cfg["paths"]["vocab_dir"]
    files = cfg["files"]
    opts = _csv_read_options(cfg)

    drug_era = spark.read.csv(f"{raw_dir}/{files['drug_era']}", **opts)
    drug_exposure = spark.read.csv(f"{raw_dir}/{files['drug_exposure']}", **opts)
    observation_period = spark.read.csv(f"{raw_dir}/{files['observation_period']}", **opts)

    concept = spark.read.csv(
        f"{vocab_dir}/{files['concept']}",
        sep=cfg["csv"]["concept_sep"],
        **opts
    )

    concept_ancestor = spark.read.csv(f"{vocab_dir}/{files['concept_ancestor']}", **opts)

    try:
        death = spark.read.csv(f"{raw_dir}/{files['death']}", **opts)
        have_death = True
    except Exception:
        death = None
        have_death = False

    return {
        "drug_era": drug_era,
        "drug_exposure": drug_exposure,
        "observation_period": observation_period,
        "concept": concept,
        "concept_ancestor": concept_ancestor,
        "death": death,
        "have_death": have_death
    }


def standardize_tables(tbl: dict) -> dict:
    for k in ["drug_era", "drug_exposure", "observation_period", "death"]:
        if tbl.get(k) is not None and "eid" in tbl[k].columns:
            tbl[k] = tbl[k].withColumnRenamed("eid", "person_id")

    drug_era = (
        tbl["drug_era"]
        .withColumn("drug_era_start_date", F.to_date("drug_era_start_date", "yyyy-MM-dd"))
        .withColumn("drug_era_end_date", F.to_date("drug_era_end_date", "yyyy-MM-dd"))
        .select(
            "person_id",
            F.col("ingredient_concept_id").alias("ingredient_concept_id"),
            "drug_era_start_date",
            "drug_era_end_date",
            "drug_exposure_count",
            "gap_days"
        )
        .filter(F.col("drug_era_start_date").isNotNull())
        .filter(F.col("drug_era_end_date").isNotNull())
        .filter(F.col("drug_era_end_date") >= F.col("drug_era_start_date"))
    )

    drug_exposure = (
        tbl["drug_exposure"]
        .withColumn("drug_exposure_start_date", F.to_date("drug_exposure_start_date", "yyyy-MM-dd"))
        .withColumn("drug_exposure_end_date", F.to_date("drug_exposure_end_date", "yyyy-MM-dd"))
        .select(
            "person_id",
            "drug_concept_id",
            "drug_exposure_start_date",
            "drug_exposure_end_date",
            "days_supply",
            "quantity"
        )
        .filter(F.col("drug_exposure_start_date").isNotNull())
    )

    observation_period = (
        tbl["observation_period"]
        .withColumn("observation_period_start_date", F.to_date("observation_period_start_date"))
        .withColumn("observation_period_end_date", F.to_date("observation_period_end_date"))
        .select("person_id", "observation_period_start_date", "observation_period_end_date")
        .filter(F.col("observation_period_start_date").isNotNull())
        .filter(F.col("observation_period_end_date").isNotNull())
    )

    concept = tbl["concept"].select(
        "concept_id", "concept_name", "domain_id", "vocabulary_id",
        "concept_class_id", "standard_concept", "concept_code"
    )

    concept_ancestor = tbl["concept_ancestor"]

    if tbl["have_death"]:
        death = (
            tbl["death"]
            .withColumn("death_date", F.to_date("death_date", "yyyy-MM-dd"))
            .groupBy("person_id")
            .agg(F.min("death_date").alias("death_date"))
        )
    else:
        death = None

    ingredient_concepts = (
        concept
        .filter((F.col("domain_id") == "Drug") & (F.col("concept_class_id") == "Ingredient"))
        .select(
            F.col("concept_id").alias("ingredient_concept_id"),
            F.col("concept_name").alias("ingredient_name")
        )
    )

    return {
        "drug_era": drug_era,
        "drug_exposure": drug_exposure,
        "observation_period": observation_period,
        "concept": concept,
        "concept_ancestor": concept_ancestor,
        "death": death,
        "have_death": tbl["have_death"],
        "ingredient_concepts": ingredient_concepts
    }
