"""
pipeline.py
===========

Core PySpark pipeline for the Master's thesis

    *Development and validation of a scalable OMOP-compatible framework for
    longitudinal prescription trajectory phenotyping.*

The module implements four conceptual stages:

1.  **Cohort eligibility** -- selecting persons with sufficient pre-index
    observation time and a valid follow-up window
    (:func:`build_eligible_from_eras`).
2.  **Era construction** -- using the OMOP ``drug_era`` table directly
    (:func:`build_primary_eras`) or reconstructing ingredient eras from
    ``drug_exposure`` and ``concept_ancestor``
    (:func:`build_exposure_derived_eras`).
3.  **Trajectory phenotyping** -- monthly active-ingredient burden, Jaccard
    turnover, rule-based prescribing states, sub-window features,
    maintenance-aware discontinuation events, and rule-based discontinuation
    phenotypes (:func:`run_trajectory_pipeline`).
4.  **Unsupervised clustering** -- K-means on the standardised feature vector
    with a stratified silhouette evaluator and cluster-level aggregation of
    the most prescribed ingredients.

Engineering notes
-----------------
* All wide joins (era-to-month overlap, switch detection, restart detection)
  are written as **range-filter joins on UNIX timestamps** rather than as
  ``crossJoin``-style Cartesian products.  This is what makes the pipeline
  scale to biobank-sized inputs.
* The silhouette is evaluated on a **stratified sample** of cluster
  assignments to bound memory pressure while keeping cluster proportions
  intact.
* The cluster-level ingredient summary **pre-aggregates** the per-month
  ingredient arrays before ``explode``, avoiding the well-known
  ``SparkOutOfMemoryError`` blow-up that occurs when long array columns are
  exploded at the row level.
* The :func:`validate_negative_controls` utility provides the
  negative-control check used in the thesis Results chapter.
"""

from pyspark.sql import functions as F, Window
from pyspark.ml.feature import VectorAssembler, StandardScaler, Normalizer
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from .utils import jaccard_distance
import os
import pandas as pd


# ---------------------------------------------------------------------------
# Cohort eligibility
# ---------------------------------------------------------------------------

def build_eligible_from_eras(era_df, observation_period, death, have_death, cfg):
    """
    Build the eligible analytic cohort with index and censor dates.

    A person is eligible if their earliest drug-era start date falls inside
    an observation period that is preceded by at least ``washout_days``.
    The censor date is the earliest of (a) the observation-period end,
    (b) ``index_date + followup_months - 1 day``, and (c) the death date
    when ``have_death`` is true.

    Parameters
    ----------
    era_df : pyspark.sql.DataFrame
        Era table containing ``person_id`` and ``drug_era_start_date``.
    observation_period : pyspark.sql.DataFrame
        OMOP ``observation_period`` table.
    death : pyspark.sql.DataFrame or None
        Deduplicated OMOP ``death`` table (one row per person).
    have_death : bool
        If True, mortality is used to clip the censor date.
    cfg : dict
        Pipeline configuration; reads ``analysis.washout_days`` and
        ``analysis.followup_months``.

    Returns
    -------
    pyspark.sql.DataFrame
        One row per eligible person with columns
        ``person_id``, ``index_date``, ``observation_period_start_date``,
        ``observation_period_end_date``, and ``censor_date``.
    """
    washout_days   = cfg["analysis"]["washout_days"]
    followup_months = cfg["analysis"]["followup_months"]

    first_era = (
        era_df
        .groupBy("person_id")
        .agg(F.min("drug_era_start_date").alias("index_date"))
    )

    obs_candidates = (
        first_era.alias("f")
        .join(
            observation_period.alias("o"),
            (F.col("f.person_id") == F.col("o.person_id")) &
            (F.col("f.index_date") >= F.col("o.observation_period_start_date")) &
            (F.col("f.index_date") <= F.col("o.observation_period_end_date")) &
            (F.datediff(F.col("f.index_date"), F.col("o.observation_period_start_date")) >= washout_days),
            "inner"
        )
        .select(
            F.col("f.person_id"),
            "index_date",
            "observation_period_start_date",
            "observation_period_end_date"
        )
    )

    # Pick observation period with the latest end date (most follow-up)
    w_obs = Window.partitionBy("person_id").orderBy(F.col("observation_period_end_date").desc())
    eligible = (
        obs_candidates
        .withColumn("rn", F.row_number().over(w_obs))
        .filter(F.col("rn") == 1)
        .drop("rn")
    )

    if have_death:
        eligible = (
            eligible
            .join(death, on="person_id", how="left")
            .withColumn("max_followup_date",
                        F.date_sub(F.add_months(F.col("index_date"), followup_months), 1))
            .withColumn("death_or_far_future",
                        F.coalesce(F.col("death_date"), F.to_date(F.lit("2100-01-01"))))
            .withColumn("censor_date",
                        F.least("observation_period_end_date",
                                "max_followup_date",
                                "death_or_far_future"))
            .drop("max_followup_date", "death_or_far_future")
        )
    else:
        eligible = (
            eligible
            .withColumn(
                "censor_date",
                F.least(
                    F.col("observation_period_end_date"),
                    F.date_sub(F.add_months(F.col("index_date"), followup_months), 1)
                )
            )
        )

    return eligible.filter(F.col("censor_date") >= F.col("index_date"))


# ---------------------------------------------------------------------------
# Era builders
# ---------------------------------------------------------------------------

def build_primary_eras(drug_era):
    """
    Return the primary-analysis era table directly from OMOP ``drug_era``.

    This is the canonical exposure representation used by the trajectory
    pipeline: each row encodes a continuous span of inferred exposure to a
    single active ingredient.  No additional merging is performed because the
    OMOP ``drug_era`` construct already embodies an exposure-continuity
    assumption.
    """
    return drug_era.select(
        "person_id",
        "ingredient_concept_id",
        F.col("drug_era_start_date").alias("era_start_date"),
        F.col("drug_era_end_date").alias("era_end_date"),
        "drug_exposure_count",
        "gap_days"
    )


def build_exposure_derived_eras(drug_exposure, concept, concept_ancestor, cfg):
    """
    Reconstruct ingredient-level eras directly from OMOP ``drug_exposure``.

    Used as a sensitivity analysis against :func:`build_primary_eras`.
    Each exposure is mapped to its ingredient ancestor through
    ``concept_ancestor`` and intervals separated by no more than
    ``analysis.exposure_gap_days`` are merged into a single era using a
    rolling-maximum gap algorithm that correctly handles transitive overlaps.
    """
    gap_days = cfg["analysis"]["exposure_gap_days"]

    de = (
        drug_exposure
        .withColumn(
            "exposure_end_date",
            F.coalesce(
                F.col("drug_exposure_end_date"),
                F.when(
                    F.col("days_supply").isNotNull() & (F.col("days_supply") > 0),
                    F.date_add(F.col("drug_exposure_start_date"), F.col("days_supply") - 1)
                ),
                F.col("drug_exposure_start_date")
            )
        )
        .filter(F.col("exposure_end_date") >= F.col("drug_exposure_start_date"))
    )

    ingredient_map = (
        concept_ancestor.alias("ca")
        .join(
            concept.alias("c"),
            F.col("ca.ancestor_concept_id") == F.col("c.concept_id"),
            "inner"
        )
        .filter(
            (F.col("c.domain_id") == "Drug") &
            (F.col("c.concept_class_id") == "Ingredient")
        )
        .select(
            F.col("ca.descendant_concept_id").alias("drug_concept_id"),
            F.col("ca.ancestor_concept_id").alias("ingredient_concept_id")
        )
        .distinct()
    )

    de_ing = (
        de.alias("d")
        .join(ingredient_map.alias("m"), on="drug_concept_id", how="inner")
        .select(
            F.col("d.person_id"),
            F.col("m.ingredient_concept_id"),
            F.col("d.drug_exposure_start_date").alias("start_date"),
            F.col("d.exposure_end_date").alias("end_date")
        )
        .filter(F.col("start_date").isNotNull() & F.col("end_date").isNotNull())
        .filter(F.col("end_date") >= F.col("start_date"))
    )

    # Deduplicate same start-day records, keep longest end
    de_ing = (
        de_ing
        .groupBy("person_id", "ingredient_concept_id", "start_date")
        .agg(F.max("end_date").alias("end_date"))
    )

    # Rolling-max gap merge (handles transitive overlaps correctly)
    w_ord = Window.partitionBy("person_id", "ingredient_concept_id").orderBy("start_date", "end_date")
    w_run = w_ord.rowsBetween(Window.unboundedPreceding, 0)

    merged = (
        de_ing
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

    return (
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


# ---------------------------------------------------------------------------
# Main trajectory pipeline
# ---------------------------------------------------------------------------

def run_trajectory_pipeline(
        era_input_df, observation_period, death, have_death,
        ingredient_concepts, cfg, label,
        fixed_k=None):
    """
    Execute the full trajectory-phenotyping pipeline end to end.

    The function is intentionally monolithic so that all transformations
    share a single Spark execution plan and the optimiser can pipeline
    shuffles where possible.  It is also re-entrant: it can be called
    repeatedly with different ``cfg`` payloads, and each call writes its
    outputs under a unique ``label`` prefix.

    The ``fixed_k`` argument controls the clustering step:

    * ``fixed_k=None`` (default): the full silhouette grid over
      ``cfg['clustering']['k_grid']`` is evaluated and the K that
      maximises the stratified silhouette is selected.  This is the
      behaviour required for the baseline run, which publishes the
      silhouette curve.
    * ``fixed_k=<int>``: the silhouette grid is skipped entirely and
      K-means is fitted once at K=``fixed_k``.  This fast path skips the
      per-cell silhouette (only the cluster labels are produced), giving an
      order-of-magnitude speed-up on large cohorts because the
      :math:`O(N^{2})`-bounded silhouette is the dominant cost.

    Parameters
    ----------
    era_input_df : pyspark.sql.DataFrame
        Either the primary OMOP ``drug_era`` table or the exposure-derived
        era table produced by :func:`build_exposure_derived_eras`.
    observation_period : pyspark.sql.DataFrame
        OMOP ``observation_period`` table.
    death : pyspark.sql.DataFrame or None
        Deduplicated OMOP ``death`` table.  Ignored when ``have_death`` is
        ``False``.
    have_death : bool
        Whether to include mortality censoring.
    ingredient_concepts : pyspark.sql.DataFrame
        Lookup with columns ``ingredient_concept_id`` and ``concept_name``.
    cfg : dict
        Configuration with ``analysis``, ``clustering``, ``project``,
        and ``run`` sections (see ``config/config_synthetic.yaml``).
    label : str
        Output filename prefix for this run.

    Returns
    -------
    dict
        Handles to all intermediate and final Spark DataFrames; the same
        artefacts are also written to ``cfg['project']['output_dir']``.

    Performance contract
    --------------------
    * **Switch detection** is implemented as a range-filter inequality join
      on UNIX timestamps (``unix_timestamp``) bounded by
      ``switch_window_days``.  This avoids the :math:`O(N^{2})` Cartesian
      blow-up that occurs with a naive self-join over date columns.
    * **Silhouette evaluation** subsamples within each cluster
      proportionally to its size so that all clusters are represented even
      when one dominates.  Cohorts smaller than 20,000 persons are evaluated
      in full.
    * **Cluster-level ingredient ranking** pre-aggregates the per-month
      ``active_set`` arrays at the cluster grain before exploding, keeping
      the row count of the exploded table linear in the number of distinct
      ingredients per cluster rather than in the total person-months.
    """

    followup_months            = cfg["analysis"]["followup_months"]
    maintenance_min_total_days = cfg["analysis"]["maintenance_min_total_days"]
    maintenance_min_eras       = int(cfg["analysis"].get("maintenance_min_eras", 2))
    maintenance_single_era_min_days = int(cfg["analysis"].get("maintenance_single_era_min_days", 180))
    early_discontinuation_days = cfg["analysis"]["early_discontinuation_days"]
    restart_window_days        = cfg["analysis"]["restart_window_days"]
    switch_window_days         = cfg["analysis"]["switch_window_days"]
    polypharmacy_threshold     = cfg["analysis"]["polypharmacy_threshold"]
    turnover_low               = cfg["analysis"]["turnover_low"]
    turnover_high              = cfg["analysis"]["turnover_high"]
    k_grid                     = cfg["clustering"]["k_grid"]
    final_k                    = cfg["clustering"].get("final_k")
    seed                       = cfg["clustering"]["seed"]

    # Rule-based discontinuation-phenotype thresholds (Methods §3.10),
    # externalised to config with the historical defaults as fallbacks.
    pheno_cfg                  = cfg.get("phenotype", {}) or {}
    pheno_stable_turnover_max  = float(pheno_cfg.get("stable_turnover_max", 0.20))
    pheno_early_disc_low       = float(pheno_cfg.get("early_disc_low", 0.25))
    pheno_early_disc_high      = float(pheno_cfg.get("early_disc_high", 0.50))
    pheno_restart_high         = float(pheno_cfg.get("restart_high", 0.50))
    pheno_switch_high          = float(pheno_cfg.get("switch_high", 0.50))
    pheno_burden_slope_neg     = float(pheno_cfg.get("burden_slope_neg", -0.10))

    outdir                     = cfg["project"]["output_dir"]
    os.makedirs(outdir, exist_ok=True)
    top_n                      = cfg["run"]["save_top_ingredients_per_cluster"]

    # Detect ingredient_concept_id type at runtime so that PySpark 3.x strict
    # type casting does not break on empty arrays of unknown element type.
    _id_type    = dict(era_input_df.dtypes).get("ingredient_concept_id", "long")
    _empty_array = F.array().cast(f"array<{_id_type}>")

    # ----- Optional: restrict to focal ingredient concept_ids ------------
    # Used for interoperability demonstrations where the full prescribing
    # profile is heterogeneous (everything in OMOP Drug) but the thesis
    # validation question is narrower (e.g. three chronic cardio-metabolic
    # RxNorm ingredient IDs on a external Synthea export). When omitted or
    # empty, behaviour is unchanged: all eras flow through.
    focus_ids = cfg.get("analysis", {}).get("focus_ingredient_concept_ids")
    if focus_ids:
        focus_ids_clean = sorted({int(x) for x in focus_ids})
        n_before = era_input_df.count()
        era_input_df = era_input_df.filter(
            F.col("ingredient_concept_id").isin(focus_ids_clean))
        print(f"[{label}] focus_ingredient_concept_ids={focus_ids_clean} "
              f"--> {era_input_df.count()} era rows retained (was {n_before}).")

    # ----- Eligibility ----------------------------------------------------
    eligible = build_eligible_from_eras(
        era_df=era_input_df,
        observation_period=observation_period,
        death=death,
        have_death=have_death,
        cfg=cfg
    )
    eligible.cache()

    # ----- Clip eras to follow-up window ---------------------------------
    eras = (
        era_input_df.alias("d")
        .join(
            eligible.select("person_id", "index_date", "censor_date").alias("e"),
            on="person_id", how="inner"
        )
        .filter(
            (F.col("d.drug_era_start_date") <= F.col("e.censor_date")) &
            (F.col("d.drug_era_end_date")   >= F.col("e.index_date"))
        )
        .withColumn("era_start_date",
                    F.greatest(F.col("d.drug_era_start_date"), F.col("e.index_date")))
        .withColumn("clipped_at_censor",
                    F.col("d.drug_era_end_date") > F.col("e.censor_date"))
        .withColumn("era_end_date",
                    F.least(F.col("d.drug_era_end_date"), F.col("e.censor_date")))
        .filter(F.col("era_end_date") >= F.col("era_start_date"))
        .select("person_id", "ingredient_concept_id",
                "era_start_date", "era_end_date",
                "drug_exposure_count", "gap_days", "clipped_at_censor")
    )
    eras.cache()

    # ----- Monthly person-time grid --------------------------------------
    person_months = (
        eligible
        .withColumn("month_index", F.explode(F.sequence(F.lit(1), F.lit(followup_months))))
        .withColumn("month_start", F.add_months(F.col("index_date"), F.col("month_index") - 1))
        .withColumn("month_end",
                    F.least(
                        F.date_sub(F.add_months(F.col("index_date"), F.col("month_index")), 1),
                        F.col("censor_date")
                    ))
        .filter(F.col("month_start") <= F.col("censor_date"))
        .select("person_id", "month_index", "month_start", "month_end")
    )

    month_overlap = (
        person_months.alias("m")
        .join(
            eras.alias("e"),
            (F.col("m.person_id") == F.col("e.person_id")) &
            (F.col("e.era_start_date") <= F.col("m.month_end")) &
            (F.col("e.era_end_date")   >= F.col("m.month_start")),
            "left"
        )
        .select(
            F.col("m.person_id"),
            "month_index",
            F.col("m.month_start"),
            F.col("m.month_end"),
            F.col("e.ingredient_concept_id"),
            F.col("e.clipped_at_censor"),
            F.when(
                (F.col("e.era_start_date") >= F.col("m.month_start")) &
                (F.col("e.era_start_date") <= F.col("m.month_end")), 1
            ).otherwise(0).alias("started_flag"),
            F.when(
                (F.col("e.era_end_date") >= F.col("m.month_start")) &
                (F.col("e.era_end_date") <= F.col("m.month_end")) &
                (~F.coalesce(F.col("e.clipped_at_censor"), F.lit(False))),
                1
            ).otherwise(0).alias("stopped_flag")
        )
    )

    person_month_summary = (
        month_overlap
        .groupBy("person_id", "month_index")
        .agg(
            F.countDistinct("ingredient_concept_id").alias("active_n"),
            F.sum("started_flag").alias("starts_n"),
            F.sum("stopped_flag").alias("stops_n"),
            F.coalesce(
                F.collect_set("ingredient_concept_id"),
                _empty_array
            ).alias("active_set")
        )
    )

    person_month_summary = (
        person_months.select("person_id", "month_index")
        .join(person_month_summary, on=["person_id", "month_index"], how="left")
        .withColumn("active_n",   F.coalesce(F.col("active_n"),   F.lit(0)))
        .withColumn("starts_n",   F.coalesce(F.col("starts_n"),   F.lit(0)))
        .withColumn("stops_n",    F.coalesce(F.col("stops_n"),    F.lit(0)))
        .withColumn("active_set",
                    F.coalesce(F.col("active_set"), _empty_array))
    )

    # ----- Monthly state classification ----------------------------------
    w_month = Window.partitionBy("person_id").orderBy("month_index")

    person_month_summary = (
        person_month_summary
        .withColumn("prev_active_n",   F.coalesce(F.lag("active_n",   1).over(w_month), F.lit(0)))
        .withColumn("prev_active_set", F.lag("active_set", 1).over(w_month))
        .withColumn("prev_active_set",
                    F.coalesce(F.col("prev_active_set"), _empty_array))
        .withColumn("turnover", jaccard_distance(F.col("active_set"), F.col("prev_active_set")))
        .withColumn(
            "state",
            F.when(F.col("active_n") == 0, "NoRx")
             .when((F.col("prev_active_n") == 0) & (F.col("active_n") > 0), "Initiation")
             .when(
                (F.col("active_n") == 1) &
                (F.col("turnover") < turnover_low) &
                (F.col("starts_n") == 0) & (F.col("stops_n") == 0),
                "StableMono"
             )
             .when(
                F.col("active_n").between(2, polypharmacy_threshold - 1) &
                (F.col("turnover") < turnover_low),
                "StableLowPoly"
             )
             .when(
                (F.col("active_n") >= polypharmacy_threshold) &
                (F.col("turnover") < turnover_low),
                "StablePolypharmacy"
             )
             .when(
                (F.col("starts_n") > 0) &
                (F.col("stops_n") > 0) &
                (F.col("turnover") >= turnover_high),
                "HighTurnover"
             )
             .when(
                (F.col("active_n") > F.col("prev_active_n")) & (F.col("starts_n") > 0),
                "Intensifying"
             )
             .when(
                (F.col("active_n") < F.col("prev_active_n")) & (F.col("stops_n") > 0),
                "Deintensifying"
             )
             .otherwise("ModerateFlux")
        )
    )

    # ----- Temporal sub-window features ----------------------------------
    # The follow-up window is divided into ``n_windows`` equally sized
    # sub-windows.  For each one we compute the mean burden and the
    # dominant prescribing state, retaining ordinal temporal information
    # that is otherwise destroyed by global per-person averaging.
    n_windows = 4
    window_size = followup_months // n_windows
    subwindow_dfs = []
    subwindow_numeric_cols = []

    for w in range(n_windows):
        lo = w * window_size + 1
        hi = lo + window_size - 1
        label_w = f"w{w+1}"

        b_feat = (
            person_month_summary
            .filter((F.col("month_index") >= lo) & (F.col("month_index") <= hi))
            .groupBy("person_id")
            .agg(F.mean("active_n").alias(f"mean_burden_{label_w}"))
        )

        s_mode = (
            person_month_summary
            .filter((F.col("month_index") >= lo) & (F.col("month_index") <= hi))
            .groupBy("person_id", "state").count()
            .withColumn("rn", F.row_number().over(Window.partitionBy("person_id").orderBy(F.col("count").desc())))
            .filter(F.col("rn") == 1)
            .select("person_id", F.col("state").alias(f"dominant_state_{label_w}"))
        )

        subwindow_dfs.extend([b_feat, s_mode])
        subwindow_numeric_cols.append(f"mean_burden_{label_w}")

    # ----- Maintenance eligibility ---------------------------------------
    # Tiered eligibility resolves the single-era exclusion bug.
    #
    # Tier A: a single era of sufficient duration is unambiguously a
    #         maintenance prescription regardless of refill count.
    # Tier B: multiple eras with minimum cumulative duration, the original
    #         logic retained for intermittent chronic users.
    #
    # maintenance_tier is carried forward into era_events for auditing
    # but is NOT in the clustering feature vector.
    maintenance = (
        eras
            .withColumn("era_days", F.datediff("era_end_date", "era_start_date") + 1)
            .groupBy("person_id", "ingredient_concept_id")
            .agg(
                F.count("*").alias("n_eras"),
                F.sum("era_days").alias("total_era_days")
            )
            .withColumn(
                "tier_a",
                F.when(
                    (F.col("n_eras") == 1) &
                    (F.col("total_era_days") >= maintenance_single_era_min_days),
                    1
                ).otherwise(0)
            )
            .withColumn(
                "tier_b",
                F.when(
                    (F.col("n_eras") >= maintenance_min_eras) &
                    (F.col("total_era_days") >= maintenance_min_total_days),
                    1
                ).otherwise(0)
            )
            .withColumn(
                "maintenance_eligible",
                F.when(
                    (F.col("tier_a") == 1) | (F.col("tier_b") == 1), 1
                ).otherwise(0)
            )
            .withColumn(
                # String label kept in maintenance and era_events for auditing only.
                # It does NOT survive the disc_summary groupBy aggregation and is
                # therefore never in the feature vector or select_cols.
                "maintenance_tier",
                F.when(F.col("tier_a") == 1, F.lit("A_single_long"))
                .when(F.col("tier_b") == 1, F.lit("B_multi_era"))
                .otherwise(F.lit("ineligible"))
            )
            .drop("tier_a", "tier_b")
    )

    # ----- Discontinuation events (range-filter join) --------------------
    w_era = Window.partitionBy("person_id", "ingredient_concept_id").orderBy("era_start_date", "era_end_date")

    eras_for_events = (
        eras
        .join(eligible.select("person_id", "censor_date"), on="person_id", how="left")
        .withColumn("era_days", F.datediff("era_end_date", "era_start_date") + 1)
        .withColumn("era_number", F.row_number().over(w_era))
        .withColumn("next_same_ingredient_start", F.lead("era_start_date", 1).over(w_era))
        .join(maintenance, on=["person_id", "ingredient_concept_id"], how="left")
        .withColumn(
            "restarted_within_180d",
            F.when(
                F.col("next_same_ingredient_start").isNotNull() &
                (F.col("next_same_ingredient_start") <=
                 F.date_add(F.col("era_end_date"), restart_window_days)),
                1
            ).otherwise(0)
        )
        .withColumn(
            "observed_for_restart_window",
            F.when(
                F.date_add(F.col("era_end_date"), restart_window_days) <= F.col("censor_date"),
                1
            ).otherwise(0)
        )
        .withColumn(
            "observed_for_switch_window",
            F.when(
                F.date_add(F.col("era_end_date"), switch_window_days) <= F.col("censor_date"),
                1
            ).otherwise(0)
        )
    )

    eras_for_events = (
        eras_for_events
        .withColumn(
            "era_row_key",
            F.concat_ws("|",
                F.col("person_id").cast("string"),
                F.col("ingredient_concept_id").cast("string"),
                F.col("era_start_date").cast("string"),
                F.col("era_end_date").cast("string")
            )
        )
    )
    eras_for_events.cache()

    # --- Timestamp-range join to avoid O(N^2) Cartesian product ---------
    # ``SWITCH_SECS`` defines an inclusive upper bound on the join, so the
    # downstream filter is a closed-interval range predicate rather than a
    # full cross-join over all eras within the same person.
    SWITCH_SECS = switch_window_days * 86400

    eras_ts = eras_for_events.withColumn("era_end_ts", F.unix_timestamp("era_end_date"))

    starts = eras.select(
        "person_id",
        F.col("ingredient_concept_id").alias("switch_to_ingredient"),
        F.unix_timestamp("era_start_date").alias("other_start_ts")
    )

    switch_flags = (
        eras_ts.alias("a")
        .join(starts.alias("b"), on="person_id", how="left")
        .filter(
            (F.col("b.other_start_ts") > F.col("a.era_end_ts")) &
            (F.col("b.other_start_ts") <= F.col("a.era_end_ts") + SWITCH_SECS) &
            (F.col("b.switch_to_ingredient") != F.col("a.ingredient_concept_id"))
        )
        .groupBy("a.era_row_key")
        .agg(F.lit(1).alias("switched_within_60d"))
    )

    era_events = (
        eras_for_events
        .join(switch_flags, on="era_row_key", how="left")
        .fillna({"switched_within_60d": 0, "maintenance_eligible": 0})
        .withColumn(
            "early_discontinuation_90d",
            F.when(
                (F.col("era_number") == 1) &
                (F.col("maintenance_eligible") == 1) &
                (F.col("observed_for_restart_window") == 1) &
                (F.col("era_days") < early_discontinuation_days) &
                (F.col("restarted_within_180d") == 0),
                1
            ).otherwise(0)
        )
    )

    # ----- Feature construction ------------------------------------------
    burden = (
        person_month_summary
        .groupBy("person_id")
        .agg(
            F.avg("month_index").alias("mx"),
            F.avg("active_n").alias("my"),
            F.avg(F.col("month_index") * F.col("active_n")).alias("mxy"),
            F.avg(F.col("month_index") * F.col("month_index")).alias("mx2"),
            F.avg("active_n").alias("mean_active_n"),
            F.avg(F.when(F.col("active_n") >= polypharmacy_threshold, 1.0).otherwise(0.0))
             .alias("poly_month_prop"),
            F.avg("turnover").alias("mean_turnover")
        )
        .withColumn(
            "burden_slope",
            F.when(
                (F.col("mx2") - F.col("mx") * F.col("mx")) != 0,
                (F.col("mxy") - F.col("mx") * F.col("my")) /
                (F.col("mx2") - F.col("mx") * F.col("mx"))
            ).otherwise(F.lit(0.0))
        )
        .select("person_id", "mean_active_n", "poly_month_prop", "mean_turnover", "burden_slope")
    )

    state_levels = [
        "NoRx", "Initiation", "StableMono", "StableLowPoly",
        "StablePolypharmacy", "Intensifying", "Deintensifying",
        "HighTurnover", "ModerateFlux"
    ]
    state_counts = (
        person_month_summary
        .groupBy("person_id")
        .pivot("state", state_levels)
        .count()
        .fillna(0)
    )
    n_months = person_month_summary.groupBy("person_id").agg(F.count("*").alias("n_months"))
    state_props = state_counts.join(n_months, on="person_id", how="left")
    for s in state_levels:
        state_props = state_props.withColumn(f"prop_{s}", F.col(s) / F.col("n_months"))
    state_props = state_props.select("person_id", *[f"prop_{s}" for s in state_levels])

    era_counts = (
        eras
        .groupBy("person_id")
        .agg(
            F.count("*").alias("n_ingredient_eras"),
            F.countDistinct("ingredient_concept_id").alias("n_distinct_ingredients")
        )
    )

    disc_summary = (
    era_events
        .filter(F.col("maintenance_eligible") == 1)
        .groupBy("person_id")
        .agg(
            F.count("*").alias("n_maintenance_eras"),
            F.avg(F.col("early_discontinuation_90d").cast("double"))
                 .alias("early_disc_90_rate"),
            F.avg(F.when(F.col("observed_for_restart_window") == 1,
                         F.col("restarted_within_180d").cast("double")))
                 .alias("restart_180_rate"),
            F.avg(F.when(F.col("observed_for_switch_window") == 1,
                         F.col("switched_within_60d").cast("double")))
                 .alias("switch_60_rate"),
            F.expr("percentile_approx(era_days, 0.5)").alias("median_era_days"),
            # Deterministic numeric encoding.
            # F.max() is used deliberately: if ANY of this person's eligible
            # ingredients entered via Tier A, the person receives 1.0.
            # F.first() is explicitly avoided because it is non-deterministic
            # across Spark partitions and would corrupt clustering reproducibility.
            # NOTE: maintenance_tier (the string) does NOT survive this groupBy.
            # It is intentionally absent from select_cols for that reason.
            F.max(
                F.when(F.col("maintenance_tier") == "A_single_long", 1.0)
                 .otherwise(0.0)
            ).alias("is_single_era_maintenance")
        )
    )

    # ----- Person-level features -----------------------------------------
    evaluable = (
        disc_summary.select(
        "person_id",
        F.lit(True).alias("disc_evaluable"),
        "n_maintenance_eras",
        "early_disc_90_rate",
        "restart_180_rate",
        "switch_60_rate",
        "median_era_days",
        "is_single_era_maintenance"    # numeric, computed deterministically above
        )
    )

    features = (
        eligible.select("person_id")
        .join(burden,    on="person_id", how="left")
        .join(state_props, on="person_id", how="left")
        .join(era_counts, on="person_id", how="left")
        .join(evaluable,  on="person_id", how="left")
    )

    # Join the four sub-window feature frames in turn
    for feat_df in subwindow_dfs:
        features = features.join(feat_df, on="person_id", how="left")

    features = (
        features.fillna(0, subset=[
            "mean_active_n", "poly_month_prop", "mean_turnover", "burden_slope",
            "n_ingredient_eras", "n_distinct_ingredients",
            #they are NULL when no observable window exists
            "early_disc_90_rate",
            "restart_180_rate",
            "switch_60_rate",
            "median_era_days",
            *subwindow_numeric_cols,
            *[f"prop_{s}" for s in state_levels]
        ])
        .withColumn("disc_evaluable", F.coalesce(F.col("disc_evaluable"), F.lit(False)))
    )  

    # ----- Rule-based discontinuation phenotype --------------------------
    # Decision order: a participant is first checked against the specific
    # dynamic behaviours (switching, repeated restart, early drop-off) and
    # only then against the generic "stable" patterns. A participant who, for
    # example, switched antidepressants is characterised by that switch rather
    # than by an otherwise low month-to-month turnover; placing the stable
    # rules first would silently absorb switchers and restarters and leave
    # those categories unreachable.
    features = (
        features
        .withColumn(
            "discontinuation_phenotype",
            F.when(
                ~F.col("disc_evaluable"),
                "Insufficient prescribing history"
            )
            .when(
                F.col("switch_60_rate") >= pheno_switch_high,
                "High-turnover switching"
            )
            .when(
                F.col("restart_180_rate") >= pheno_restart_high,
                "Intermittent stop-start"
            )
            .when(
                (F.col("early_disc_90_rate") >= pheno_early_disc_high) |
                (F.col("burden_slope") < pheno_burden_slope_neg),
                "Early drop-off / de-intensification"
            )
            .when(
                (F.col("mean_active_n") >= polypharmacy_threshold) &
                (F.col("mean_turnover") < pheno_stable_turnover_max) &
                (F.col("early_disc_90_rate") < pheno_early_disc_low),
                "Stable polypharmacy"
            )
            .when(
                (F.col("mean_turnover") < pheno_stable_turnover_max) &
                (F.col("early_disc_90_rate") < pheno_early_disc_low),
                "Persistent stable use"
            )
            .otherwise("Mixed transition pattern")
        )
    )

    # ----- K-means clustering with stratified silhouette -----------------
    feature_cols = [
        "mean_active_n",
        "poly_month_prop",
        "mean_turnover",
        "burden_slope",
        "n_ingredient_eras",
        "n_distinct_ingredients",
        "early_disc_90_rate",
        "restart_180_rate",
        "switch_60_rate",
        "median_era_days",
        "is_single_era_maintenance",   
        *subwindow_numeric_cols,
        *[f"prop_{s}" for s in state_levels]
    ]

    cluster_input = features.filter(F.col("disc_evaluable"))
    # Count BEFORE StandardScaler.fit: Spark ML raises
    # ``IllegalArgumentException: Nothing has been added to this summarizer``
    # on empty input; some sensitivity-grid cells temporarily reduce the
    # evaluable subgroup to zero.
    n_evaluable = cluster_input.count()

    scaler = StandardScaler(inputCol="features_raw", outputCol="features_scaled",
                            withMean=True, withStd=True)
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw",
                                handleInvalid="keep")
    # L2 normalisation (Methods §3.11.2): minimising squared Euclidean
    # distance on L2-normalised vectors is equivalent to maximising cosine
    # similarity. K-means then uses the default (Euclidean) distance.
    normalizer = Normalizer(inputCol="features_scaled", outputCol="features_cosine", p=2.0)

    best_model = None
    best_k = None
    best_score = float("nan")

    if n_evaluable == 0:
        print(f"[{label}] SKIPPING K-Means: n_evaluable=0...")
    else:
        assembled = assembler.transform(cluster_input.fillna(0, subset=feature_cols))
        scaled = scaler.fit(assembled).transform(assembled)
        # Apply L2 normalisation to the standardised features (§3.11.2)
        scaled = normalizer.transform(scaled)

        # ----- Cohort-size guard for K-Means feasibility ----------------------
        # PySpark's KMeans raises a non-catchable JVM
        # ArrayIndexOutOfBoundsException when the requested K exceeds the
        # number of distinct feature vectors available for fitting (e.g.
        # tiny external OMOP demos where the strict eligibility filter
        # leaves only a handful of evaluable persons).  We compute the
        # number of distinct evaluable feature vectors up front, and fall
        # back to the unclustered baseline (every person assigned to -1)
        # if even K=2 is infeasible.  This realises the safe-failure
        # contract advertised in the Methods chapter for the small-cohort
        # regime.
        n_distinct  = (
            cluster_input.fillna(0, subset=feature_cols)
                         .dropDuplicates(feature_cols)
                         .count()
        )
        max_feasible_k = max(1, min(n_evaluable, n_distinct))
        print(f"[{label}] n_evaluable={n_evaluable}, "
              f"n_distinct_feature_vectors={n_distinct}, "
              f"max_feasible_k={max_feasible_k}")

        # The silhouette is evaluated with cosine distance so that the reported
        # quality index matches the cosine-equivalent K-means objective
        # (Methods §3.11.2). Spark's ClusteringEvaluator otherwise defaults to
        # ``squaredEuclidean``, which is not the distance the model optimises
        # and inflates the reported silhouette.
        evaluator = ClusteringEvaluator(
            featuresCol="features_cosine",
            predictionCol="trajectory_cluster",
            metricName="silhouette",
            distanceMeasure="cosine"
        )

        sil_records = []

        if fixed_k is not None and fixed_k <= max_feasible_k:
            # ---- Fast path: skip silhouette evaluation entirely ----------------
            km = KMeans(k=fixed_k, seed=seed,
                        featuresCol="features_cosine",
                        predictionCol="trajectory_cluster")
            best_model = km.fit(scaled)
            best_k     = fixed_k
            best_score = float("nan")
            print(f"[{label}] fixed_k={fixed_k} (silhouette evaluation skipped)")
        elif fixed_k is not None and fixed_k > max_feasible_k:
            print(f"[{label}] SKIPPING K-Means: fixed_k={fixed_k} > "
                  f"max_feasible_k={max_feasible_k}; all persons assigned to -1.")
            best_k = -1
            best_score = float("nan")
        else:
            # ---- Full path: silhouette-based K selection ----------------------
            feasible_grid = [k for k in k_grid if k <= max_feasible_k]
            if not feasible_grid:
                print(f"[{label}] SKIPPING K-Means: no k in {k_grid} is <= "
                      f"max_feasible_k={max_feasible_k}; all persons assigned to -1.")
            else:
                best_score = -1.0  # initialise for max search

            for k in feasible_grid:
                km = KMeans(k=k, seed=seed,
                            featuresCol="features_cosine",
                            predictionCol="trajectory_cluster")
                model = km.fit(scaled)
                pred  = model.transform(scaled)

                # --- Stratified silhouette sampling --------------------------
                cluster_counts = pred.groupBy("trajectory_cluster").count().collect()
                total = sum(row["count"] for row in cluster_counts)
                target_sample_fraction = 1.0 if total < 20_000 else 0.15

                if target_sample_fraction == 1.0:
                    sil_input = pred
                else:
                    min_fraction = 0.05
                    fractions = {
                        row["trajectory_cluster"]: max(min_fraction, min(target_sample_fraction * total / row["count"], 1.0))
                        for row in cluster_counts
                    }
                    sil_input = pred.sampleBy("trajectory_cluster", fractions=fractions, seed=seed)

                score = evaluator.evaluate(sil_input)
                sil_records.append({"label": label, "k": k, "silhouette": score})
                print(f"[{label}] k={k}, silhouette={score:.4f}")
                if score > best_score:
                    best_score = score
                    best_k     = k
                    best_model = model

            if feasible_grid:
                best_score = best_score if best_score != -1.0 else float("nan")

            sil_df = pd.DataFrame(sil_records)
            sil_df.to_csv(f"{outdir}/{label}_silhouette_grid.csv", index=False)

            # The silhouette sweep above is the model-selection DIAGNOSTIC.
            # When ``clustering.final_k`` is configured, the operational
            # partition is fixed at that k for phenotypic interpretability
            # (the silhouette optimum favours fewer, coarser clusters; a
            # larger k is retained because it resolves clinically meaningful
            # archetypes). The reported silhouette is the value at final_k.
            if final_k is not None and feasible_grid and final_k <= max_feasible_k:
                if best_k != final_k:
                    km_final = KMeans(k=final_k, seed=seed,
                                      featuresCol="features_cosine",
                                      predictionCol="trajectory_cluster")
                    best_model = km_final.fit(scaled)
                    best_k = final_k
                    match = [r["silhouette"] for r in sil_records if r["k"] == final_k]
                    best_score = match[0] if match else float("nan")
                    print(f"[{label}] final_k override -> k={final_k} "
                          f"(silhouette diagnostic optimum reported in grid CSV)")

    if best_model is None:
        # Safe-failure path (no rows, degenerate feasibility, etc.)
        clustered = (
            cluster_input.select("person_id")
                         .withColumn("trajectory_cluster", F.lit(-1).cast("int"))
        )
    else:
        clustered = best_model.transform(scaled).select("person_id", "trajectory_cluster")

    final_person = (
        features
        .join(clustered, on="person_id", how="left")
        .withColumn("trajectory_cluster",
                    F.coalesce(F.col("trajectory_cluster").cast("int"), F.lit(-1)))
    )

    # Retain all feature_cols and dominant_state strings for downstream analysis
    select_cols = ["person_id", "trajectory_cluster", "discontinuation_phenotype", *feature_cols, "disc_evaluable"]
    
    for w in range(n_windows):
        select_cols.append(f"dominant_state_w{w+1}")

    final_person = final_person.select(*select_cols)

    # ----- Cluster summaries ---------------------------------------------
    cluster_summary = (
        final_person
        .filter(F.col("trajectory_cluster") >= 0)
        .groupBy("trajectory_cluster")
        .agg(
            F.count("*").alias("n_people"),
            F.avg("mean_active_n").alias("mean_active_n"),
            F.avg("poly_month_prop").alias("poly_month_prop"),
            F.avg("mean_turnover").alias("mean_turnover"),
            F.avg("early_disc_90_rate").alias("early_disc_90_rate"),
            F.avg("restart_180_rate").alias("restart_180_rate"),
            F.avg("switch_60_rate").alias("switch_60_rate")
        )
        .orderBy("trajectory_cluster")
    )

    cluster_months = (
        final_person.select("person_id", "trajectory_cluster")
        .join(
            person_month_summary.select("person_id", "month_index", "active_set"),
            on="person_id", how="inner"
        )
    )

    # --- Pre-aggregate arrays to the cluster level before exploding -----
    # ``flatten(collect_list(active_set))`` reduces the array-explode problem
    # from O(n_person_months) to O(n_clusters * distinct_ingredients), which
    # prevents the JVM-side SparkOutOfMemoryError that occurs on naive
    # row-level explodes of long array columns.
    cluster_ingredients = (
        cluster_months
        .groupBy("trajectory_cluster")
        .agg(F.flatten(F.collect_list("active_set")).alias("all_ingredients"))
        .withColumn("ingredient_concept_id", F.explode("all_ingredients"))
        .groupBy("trajectory_cluster", "ingredient_concept_id")
        .agg(F.count("*").alias("cluster_month_count"))
        .join(ingredient_concepts, on="ingredient_concept_id", how="left")
    )

    w_top = Window.partitionBy("trajectory_cluster").orderBy(F.col("cluster_month_count").desc())
    top_ingredients = (
        cluster_ingredients
        .withColumn("rank", F.row_number().over(w_top))
        .filter(F.col("rank") <= top_n)
        .orderBy("trajectory_cluster", "rank")
    )

    # ----- Write outputs --------------------------------------------------
    prefix = f"{outdir}/{label}"
    eligible.write.mode("overwrite").parquet(f"{prefix}_eligible.parquet")
    eras.write.mode("overwrite").parquet(f"{prefix}_eras.parquet")
    person_month_summary.write.mode("overwrite").parquet(f"{prefix}_person_months.parquet")
    era_events.write.mode("overwrite").parquet(f"{prefix}_era_events.parquet")
    final_person.write.mode("overwrite").parquet(f"{prefix}_person_level_phenotypes.parquet")
    cluster_summary.write.mode("overwrite").parquet(f"{prefix}_cluster_summary.parquet")
    top_ingredients.write.mode("overwrite").parquet(f"{prefix}_top_ingredients.parquet")

    # ----- Release cached intermediates ----------------------------------
    eligible.unpersist()
    eras.unpersist()
    eras_for_events.unpersist()

    if isinstance(best_score, float) and best_score != best_score:
        ss = "nan"
    else:
        try:
            ss = f"{float(best_score):.4f}"
        except (TypeError, ValueError):
            ss = "nan"
    print(f"[{label}] selected k = {best_k}, silhouette = {ss}")

    return {
        "eligible":            eligible,
        "eras":                eras,
        "person_month_summary": person_month_summary,
        "era_events":          era_events,
        "final_person":        final_person,
        "cluster_summary":     cluster_summary,
        "top_ingredients":     top_ingredients
    }


# ---------------------------------------------------------------------------
# Standalone validation utilities
# ---------------------------------------------------------------------------

def validate_negative_controls(phenotype_df, acute_person_ids):
    """
    Negative-control check: ensure acute-only patients are not evaluated as
    maintenance.

    Returns a dictionary with the number of acute patients that the
    pipeline incorrectly flagged as maintenance-evaluable
    (``disc_evaluable == True``).  In the thesis a small residual leakage is
    expected (and is reported as a formal heuristic limitation); the pass
    criterion is therefore exposed alongside the raw violation count so the
    caller can decide whether to treat it as fatal.

    Parameters
    ----------
    phenotype_df : pyspark.sql.DataFrame
        Output of :func:`run_trajectory_pipeline` (the ``final_person``
        frame), containing at least ``person_id`` and ``disc_evaluable``.
    acute_person_ids : Iterable[int]
        Person identifiers that were generated as the
        ``acute_antibiotic`` archetype in the synthetic cohort.

    Returns
    -------
    dict
        ``{"passed": bool, "n_violations": int, "violations": DataFrame}``.
    """
    acute_df = phenotype_df.filter(
        F.col("person_id").isin(list(acute_person_ids))
    )

    violations = acute_df.filter(F.col("disc_evaluable") == True)
    n_violations = violations.count()

    return {
        "passed": n_violations == 0,
        "n_violations": n_violations,
        "violations": violations,
    }
