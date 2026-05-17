"""
danish_register_adapter.py
==========================
Is suposed to map Danish National Prescription Registry (LMDB / Receptregisteret) data
to the intermediate era format expected by run_trajectory_pipeline(). (not tested yet)

Input columns expected (Statistics Denmark LMDB extract):
  - pnr          : pseudonymized person identifier (str)
  - eksd         : dispensing date (str YYYY-MM-DD or datetime)
  - atc          : ATC code (str, e.g. 'N06AB06')
  - packsize     : number of units in pack (float, optional)
  - apk          : number of packs dispensed (float, optional)
  - indo         : indication text (optional, not used)
  - volume       : DDD-based volume (float, optional)

Output should match the schema of build_primary_eras() / build_exposure_derived_eras():
  - person_id              : long (hash of pnr)
  - ingredient_concept_id  : long (numeric ATC group ID, level 5)
  - era_start_date         : date
  - era_end_date           : date
  - drug_exposure_count    : int
  - gap_days               : int

Usage:
    from danish_register_adapter import build_danish_eras, build_danish_observation_period
    eras = build_danish_eras(spark, lmdb_path, atc_filter=["N05A", "N06A"], gap_days=30)
    obs  = build_danish_observation_period(spark, cpr_path, death_path)
    # then pass directly to run_trajectory_pipeline()

fx Psychiatric ATC filters:
  N05A  antipsychotics
  N06A  antidepressants
  N05B  anxiolytics
  N05C  hypnotics & sedatives
  N06D  anti-dementia (if relevant)
"""

from pyspark.sql import functions as F, Window
from pyspark.sql.types import LongType
import hashlib


#  ATC → ingredient_concept_id mapping 
#  use the 7-character ATC code (level 5 = substance) as the "ingredient".
# Since there are no OMOP concept IDs, derive a stable integer ID from
# the ATC string using a deterministic hash.  

def _atc_to_concept_id_udf():
    """Returns a UDF that hashes an ATC-5 code to a stable long integer."""
    def _hash(atc5: str) -> int:
        if atc5 is None:
            return -1
        return int(hashlib.md5(atc5.encode()).hexdigest()[:15], 16)
    return F.udf(_hash, LongType())


def _extract_atc5(atc_col):
    """Truncate any ATC code to the first 7 characters (level 5 / substance)."""
    return F.when(
        F.length(atc_col) >= 7,
        F.substring(atc_col, 1, 7)
    ).otherwise(atc_col)


#  Main era builder 

def build_danish_eras(
    spark,
    lmdb_path: str,
    atc_filter: list[str] | None = None,
    gap_days: int = 30,
    ddd_per_day: float = 1.0,
    default_supply_days: int = 30
) -> "DataFrame":
    """
    Build ingredient-level prescription eras from Danish LMDB data.

    Parameters
    ----------
    lmdb_path        : path to LMDB CSV (or parquet) on the analysis server
    atc_filter       : list of ATC prefixes to include, e.g. ["N05A","N06A"].
                       None = all drugs.
    gap_days         : maximum gap (days) between dispensings to be merged
                       into one continuous era.  Default 30.
    ddd_per_day      : assumed DDDs consumed per day (default 1.0).
                       Used to estimate days_supply from volume.
    default_supply_days : fallback days_supply when volume is also missing.
    """
    atc_hash = _atc_to_concept_id_udf()

    raw = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(lmdb_path)          # swap for .parquet() if that's the format
    )

    # Normalise column names to lowercase
    raw = raw.toDF(*[c.lower() for c in raw.columns])

    # Parse dispensing date
    lmdb = raw.withColumn(
        "dispensing_date",
        F.to_date(F.col("eksd"))
    ).filter(F.col("dispensing_date").isNotNull())

    # Optional: filter to psychiatric ATC prefixes
    if atc_filter:
        conditions = [F.col("atc").startswith(prefix) for prefix in atc_filter]
        combined   = conditions[0]
        for c in conditions[1:]:
            combined = combined | c
        lmdb = lmdb.filter(combined)

    # Extract ATC level 5 (substance) and hash to integer ID
    lmdb = (
        lmdb
        .withColumn("atc5", _extract_atc5(F.col("atc")))
        .filter(F.col("atc5").isNotNull())
        .withColumn("ingredient_concept_id", atc_hash(F.col("atc5")))
    )

    # Stable integer person_id from pseudonymized PNR
    lmdb = lmdb.withColumn(
        "person_id",
        F.conv(F.substring(F.md5(F.col("pnr").cast("string")), 1, 15), 16, 10).cast(LongType())
    )

    # Estimate days_supply from DDD volume when available, else fallback
    lmdb = lmdb.withColumn(
        "days_supply",
        F.when(
            F.col("volume").isNotNull() & (F.col("volume") > 0),
            (F.col("volume") / F.lit(ddd_per_day)).cast("int")
        ).otherwise(F.lit(default_supply_days))
    )

    # Exposure end date = dispensing date + days_supply - 1
    exposures = (
        lmdb
        .withColumn("start_date", F.col("dispensing_date"))
        .withColumn("end_date",   F.date_add(F.col("start_date"), F.col("days_supply") - 1))
        .filter(F.col("end_date") >= F.col("start_date"))
        .select("person_id", "ingredient_concept_id", "start_date", "end_date", "atc5")
    )

    # Deduplicate same-person, same-ingredient, same-start
    exposures = (
        exposures
        .groupBy("person_id", "ingredient_concept_id", "start_date")
        .agg(F.max("end_date").alias("end_date"))
    )

    # Rolling-max gap merge (same algorithm as build_exposure_derived_eras)
    w_ord = Window.partitionBy("person_id", "ingredient_concept_id").orderBy("start_date", "end_date")
    w_run = w_ord.rowsBetween(Window.unboundedPreceding, 0)

    merged = (
        exposures
        .withColumn("running_end", F.max("end_date").over(w_run))
        .withColumn("prev_running_end", F.lag("running_end").over(w_ord))
        .withColumn(
            "new_group",
            F.when(
                F.col("prev_running_end").isNull() |
                (F.col("start_date") > F.date_add(F.col("prev_running_end"), gap_days)),
                1
            ).otherwise(0)
        )
        .withColumn("era_group", F.sum("new_group").over(w_ord))
    )

    eras = (
        merged
        .groupBy("person_id", "ingredient_concept_id", "era_group")
        .agg(
            F.min("start_date").alias("era_start_date"),
            F.max("end_date").alias("era_end_date"),
            F.count("*").alias("drug_exposure_count")
        )
        .withColumn("gap_days", F.lit(gap_days))
        .select("person_id", "ingredient_concept_id",
                "era_start_date", "era_end_date",
                "drug_exposure_count", "gap_days")
    )

    return eras


def build_danish_observation_period(
    spark,
    cpr_path: str,
    death_path: str | None = None,
    registry_start: str = "1995-01-01",
    registry_end: str   = "2023-12-31"
) -> tuple:
    """
    Build observation_period and death tables from Danish civil registration.

    Parameters
    ----------
    cpr_path       : path to CPR/civil registration extract with columns:
                     pnr, foed_dato (birth date), haend_dato (event date),
                     haend_kode (event code: emigration=U, death=D),
                     or separate death file.
    death_path     : optional separate death register path with pnr, dod_dato.
    registry_start : first date with reliable prescription data.
    registry_end   : analysis end date.

    Returns
    -------
    (observation_period_df, death_df, have_death: bool)
    """
    reg_start = F.to_date(F.lit(registry_start))
    reg_end   = F.to_date(F.lit(registry_end))

    cpr = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(cpr_path)
        .toDF(*[c.lower() for c in
                spark.read.option("header","true").csv(cpr_path).columns])
    )

    cpr = cpr.withColumn(
        "person_id",
        F.conv(F.substring(F.md5(F.col("pnr").cast("string")), 1, 15), 16, 10)
         .cast("long")
    )

    # Birth date → 18th birthday as earliest possible start
    cpr = cpr.withColumn("birth_date", F.to_date(F.col("foed_dato")))
    cpr = cpr.withColumn("adult_date", F.add_months(F.col("birth_date"), 18 * 12))

    # Emigration: take earliest emigration date per person
    emigration = (
        cpr
        .filter(F.upper(F.col("haend_kode")).isin(["U", "EM", "EMIGRATION"]))
        .groupBy("person_id")
        .agg(F.min(F.to_date(F.col("haend_dato"))).alias("emigration_date"))
    )

    persons = (
        cpr
        .select("person_id", "birth_date", "adult_date")
        .distinct()
        .join(emigration, on="person_id", how="left")
    )

    # Observation period start = max(registry_start, adult_date)
    # Observation period end   = min(registry_end, emigration_date)
    obs = (
        persons
        .withColumn("observation_period_start_date",
                    F.greatest(reg_start, F.col("adult_date")))
        .withColumn("observation_period_end_date",
                    F.when(
                        F.col("emigration_date").isNotNull(),
                        F.least(reg_end, F.col("emigration_date"))
                    ).otherwise(reg_end))
        .filter(F.col("observation_period_end_date") > F.col("observation_period_start_date"))
        .select("person_id",
                "observation_period_start_date",
                "observation_period_end_date")
    )

    have_death = death_path is not None
    if have_death:
        death_raw = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(death_path)
            .toDF(*[c.lower() for c in
                    spark.read.option("header","true").csv(death_path).columns])
        )
        death = (
            death_raw
            .withColumn("person_id",
                        F.conv(F.substring(F.md5(F.col("pnr").cast("string")), 1, 15), 16, 10)
                         .cast("long"))
            .withColumn("death_date", F.to_date(F.col("dod_dato")))
            .select("person_id", "death_date")
            .distinct()
        )
        # Also clip observation period at death
        obs = (
            obs
            .join(death.select("person_id", F.col("death_date").alias("_death")),
                  on="person_id", how="left")
            .withColumn("observation_period_end_date",
                        F.when(
                            F.col("_death").isNotNull(),
                            F.least(F.col("observation_period_end_date"), F.col("_death"))
                        ).otherwise(F.col("observation_period_end_date")))
            .drop("_death")
        )
    else:
        death = spark.createDataFrame([], "person_id long, death_date date")

    return obs, death, have_death


def build_danish_ingredient_concepts(spark, eras_df) -> "DataFrame":
    """
    Build a minimal ingredient_concepts lookup table from the ATC codes
    present in the era DataFrame. No OMOP vocabulary,
    so use the ATC5 code as the concept name.
    """
    # Re-derive ATC5 → concept_id mapping from the eras
    atc_hash = _atc_to_concept_id_udf()


    concepts = (
        eras_df
        .select("ingredient_concept_id")
        .distinct()
        .withColumn("concept_name", F.col("ingredient_concept_id").cast("string"))
        # Caller should join in a real ATC name lookup table here.
    )
    return concepts
