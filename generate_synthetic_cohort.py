"""
generate_synthetic_cohort.py
-----------------------------
Generates an OMOP-compatible synthetic cohort calibrated to the
Danish Medstat distributions from Chapter 2 of the thesis.

The canonical, reproducible regime used throughout the thesis is
``--n_patients 1000 --seed 42`` (now the defaults).

Usage:
    python generate_synthetic_cohort.py --n_patients 1000 --seed 42

The output is a set of Parquet files in OMOP drug_era schema:
    synthetic_drug_era.parquet
    synthetic_observation_period.parquet
    synthetic_person.parquet

These can be read directly by the pipeline with no schema changes.
"""

import argparse
import random
import uuid
from datetime import date, timedelta

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Medstat-calibrated drug distributions (from Chapter 2, Table 2.2)
# These are approximate individual-level parameters derived from the
# population-level Medstat statistics using published persistence data.
# ---------------------------------------------------------------------------

# Format: {archetype_name: {
#   "prevalence":   fraction of patients who receive this drug pattern,
#   "n_eras":       (min, max) number of distinct eras,
#   "era_duration": (min, max) duration per era in days,
#   "gap_days":     (min, max) gap between eras in days,
#   "atc_group":    ATC group label for annotation,
#   "concept_id":   fake OMOP concept_id (integer)
# }}

# Antidepressants (N06A): ~11% of women, ~7% of men use in any year.
# Published persistence data (Garnock-Jones 2010, Hermansen 2018):
#   ~50% discontinue within 3 months, ~30% persist >1 year.
# We model this as a mixture:
#   - "Persistent stable" (30%): 1 era, 300–720 days
#   - "Early drop-off" (35%): 1 era, 14–90 days, no restart
#   - "Intermittent" (35%): 2–4 eras, 60–180 days each, 30–120 day gaps
#
# Three additional archetypes are included so that the pipeline's
# maintenance-aware discontinuation features are exercised by genuinely
# maintenance-eligible patients (rather than being identically zero):
#   - "Early drop-off then late re-initiation": a short (<90 day) first era
#     followed by a >180 day gap and a longer second era of the SAME
#     ingredient. The pair is maintenance-eligible (>=2 eras) and the first
#     era qualifies as an early discontinuation (era < 90 days, no restart
#     within the 180-day window). Drives ``early_disc_90_rate``.
#   - "Therapeutic switch": a maintenance-length first ingredient followed,
#     within 60 days of its end, by a DIFFERENT N06A ingredient (clinically,
#     switching for inefficacy/intolerance; cf. STAR*D). Drives
#     ``switch_60_rate``.
#   - "Severe polypharmacy": >=5 concurrent ingredients (antidepressant +
#     antipsychotic + anxiolytic + mood stabiliser + somatic), as seen in
#     complex severe-mental-illness regimens. Drives ``poly_month_prop`` and
#     the ``StablePolypharmacy`` monthly state (threshold = 5 concurrent).

DRUG_ARCHETYPES = {
    # ----- Antidepressants (N06A) -----
    "N06A_persistent": {
        "prevalence": 0.09,    # ~9% of cohort
        "n_eras": (1, 1),
        "era_duration_days": (300, 730),
        "gap_days": (0, 0),    # no gap (single era)
        "atc_group": "N06A",
        "concept_id": 700001,  # fake — replace with real OMOP ingredient concept
        "concept_name": "sertraline",
    },
    "N06A_early_dropoff": {
        "prevalence": 0.11,
        "n_eras": (1, 1),
        "era_duration_days": (14, 90),
        "gap_days": (0, 0),
        "atc_group": "N06A",
        "concept_id": 700002,
        "concept_name": "escitalopram",
    },
    "N06A_intermittent": {
        "prevalence": 0.10,
        "n_eras": (2, 4),
        "era_duration_days": (60, 180),
        "gap_days": (30, 150),
        "atc_group": "N06A",
        "concept_id": 700001,
        "concept_name": "sertraline",
    },
    # Early discontinuation of the FIRST era (< 90 days) followed by a gap
    # longer than the 180-day restart window and a longer second era of the
    # SAME ingredient. The pair is maintenance-eligible (>=2 eras, cumulative
    # duration >= 56 days), so the early first era is recorded as an early
    # discontinuation. Drives early_disc_90_rate among eligible patients.
    "N06A_early_then_restart": {
        "prevalence": 0.06,
        "n_eras": (2, 2),
        "era_duration_days": (21, 75),          # first era: short (< 90 days)
        "later_era_duration_days": (120, 250),  # subsequent eras: maintenance length
        "gap_days": (200, 300),                 # > 180-day restart window
        "atc_group": "N06A",
        "concept_id": 700001,
        "concept_name": "sertraline",
    },
    # Therapeutic switch: a maintenance-length antidepressant (Tier-A
    # eligible) discontinued and replaced, within 60 days, by a DIFFERENT
    # N06A ingredient. Drives switch_60_rate.
    "N06A_switch": {
        "prevalence": 0.05,
        "n_eras": (1, 1),
        "era_duration_days": (180, 260),        # first AD: single era >= 180 days
        "gap_days": (0, 0),
        "atc_group": "N06A",
        "concept_id": 700001,
        "concept_name": "sertraline",
        "switch_to": {
            "concept_id": 700002,
            "concept_name": "escitalopram",
            "atc_group": "N06A",
            "delay_after_end_days": (10, 55),   # new ingredient within 60 days
            "era_duration_days": (180, 360),
        },
    },
    # ----- Antipsychotics (N05A) -----
    "N05A_stable": {
        "prevalence": 0.03,
        "n_eras": (1, 1),
        "era_duration_days": (270, 730),
        "gap_days": (0, 0),
        "atc_group": "N05A",
        "concept_id": 700010,
        "concept_name": "haloperidol",
    },
    "N05A_intermittent": {
        "prevalence": 0.02,
        "n_eras": (2, 3),
        "era_duration_days": (90, 270),
        "gap_days": (30, 120),
        "atc_group": "N05A",
        "concept_id": 700011,
        "concept_name": "risperidone",
    },
    # ----- Anxiolytics (N05B): declining since 2000, ~2% prevalence -----
    "N05B_acute": {
        "prevalence": 0.04,
        "n_eras": (1, 2),
        "era_duration_days": (7, 30),
        "gap_days": (30, 180),
        "atc_group": "N05B",
        "concept_id": 700020,
        "concept_name": "diazepam",
    },
    # ----- Hypnotics (N05C): post-2020 rebound -----
    "N05C_intermittent": {
        "prevalence": 0.05,
        "n_eras": (1, 3),
        "era_duration_days": (14, 90),
        "gap_days": (30, 180),
        "atc_group": "N05C",
        "concept_id": 700030,
        "concept_name": "zolpidem",
    },
    # ----- Polypharmacy: psychiatric + somatic combination -----
    # Models patients who escalate to polypharmacy (N06A + N05A + somatic)
    "polypharmacy_escalating": {
        "prevalence": 0.06,
        "n_eras": (1, 1),
        "era_duration_days": (270, 730),
        "gap_days": (0, 0),
        "atc_group": "N06A",   # anchor drug; additional drugs added separately
        "concept_id": 700001,
        "concept_name": "sertraline",
        "additional_drugs": [
            {"concept_id": 700010, "concept_name": "haloperidol",
             "start_offset_days": (90, 180), "era_duration_days": (180, 540)},
            {"concept_id": 700040, "concept_name": "metoprolol",
             "start_offset_days": (0, 60), "era_duration_days": (270, 730)},
        ],
    },
    # Severe polypharmacy: >=5 concurrent ingredients sustained across most of
    # follow-up (antidepressant + antipsychotic + anxiolytic + mood stabiliser
    # + somatic). Reaches the polypharmacy threshold (5 concurrent), so it
    # populates poly_month_prop and the StablePolypharmacy monthly state.
    "polypharmacy_severe": {
        "prevalence": 0.04,
        "n_eras": (1, 1),
        # All five ingredients start at index and span the full 24-month
        # follow-up (clipped to the censor date), so the average concurrent
        # ingredient count reaches the polypharmacy threshold (5) rather than
        # being diluted by partial-window exposure.
        "era_duration_days": (720, 760),
        "gap_days": (0, 0),
        "atc_group": "N06A",
        "concept_id": 700001,
        "concept_name": "sertraline",
        "additional_drugs": [
            {"concept_id": 700010, "concept_name": "haloperidol",
             "start_offset_days": (0, 0), "era_duration_days": (720, 760)},
            {"concept_id": 700020, "concept_name": "diazepam",
             "start_offset_days": (0, 0), "era_duration_days": (720, 760)},
            {"concept_id": 700040, "concept_name": "metoprolol",
             "start_offset_days": (0, 0), "era_duration_days": (720, 760)},
            {"concept_id": 700060, "concept_name": "lamotrigine",
             "start_offset_days": (0, 0), "era_duration_days": (720, 760)},
        ],
    },
    # ----- Acute-only (negative control for maintenance logic) -----
    "acute_antibiotic": {
        "prevalence": 0.15,
        "n_eras": (1, 3),
        "era_duration_days": (5, 10),
        "gap_days": (60, 365),   # long gaps — not a maintenance drug
        "atc_group": "J01",
        "concept_id": 700050,
        "concept_name": "amoxicillin",
    },
    # ----- No psychiatric medication (baseline) -----
    "no_psychiatric_rx": {
        "prevalence": 0.22,
        "n_eras": (0, 0),
        "era_duration_days": (0, 0),
        "gap_days": (0, 0),
        "atc_group": None,
        "concept_id": None,
        "concept_name": None,
    },
}


def sample_archetype(rng):
    """Sample an archetype label according to prevalence weights."""
    names = list(DRUG_ARCHETYPES.keys())
    weights = [DRUG_ARCHETYPES[n]["prevalence"] for n in names]
    total = sum(weights)
    weights = [w / total for w in weights]
    return rng.choices(names, weights=weights, k=1)[0]


def generate_eras_for_archetype(person_id, archetype_name, index_date, rng):
    """
    Generate drug_era rows for one patient under one archetype.

    Returns list of dicts compatible with OMOP drug_era schema.
    """
    arch = DRUG_ARCHETYPES[archetype_name]
    rows = []

    if arch["n_eras"] == (0, 0):
        return rows  # no-rx archetype

    n_eras = rng.randint(*arch["n_eras"])
    concept_id = arch["concept_id"]

    current_start = index_date
    last_era_end = index_date

    for era_idx in range(n_eras):
        # Eras after the first may use a distinct (typically longer) duration
        # range, so the first era can be a short early drop-off while the
        # pair as a whole still reaches maintenance eligibility.
        if era_idx > 0 and "later_era_duration_days" in arch:
            era_dur = rng.randint(*arch["later_era_duration_days"])
        else:
            era_dur = rng.randint(*arch["era_duration_days"])
        era_start = current_start
        era_end = era_start + timedelta(days=era_dur)
        last_era_end = era_end

        rows.append({
            "person_id": person_id,
            "drug_concept_id": concept_id,
            "ingredient_concept_id": concept_id,
            "ingredient_name": arch["concept_name"],
            "atc_group": arch["atc_group"],
            "archetype": archetype_name,
            "drug_era_start_date": era_start,
            "drug_era_end_date": era_end,
            "drug_exposure_count": rng.randint(1, 4),
            "gap_days": 0,
        })

        if era_idx < n_eras - 1:
            gap = rng.randint(*arch["gap_days"])
            current_start = era_end + timedelta(days=gap + 1)

    # Therapeutic switch: a DIFFERENT ingredient begins shortly after the
    # primary ingredient's last era ends (within the switch window), modelling
    # a clinical switch for inefficacy/intolerance.
    if "switch_to" in arch:
        sw = arch["switch_to"]
        delay = rng.randint(*sw["delay_after_end_days"])
        sw_start = last_era_end + timedelta(days=delay)
        sw_dur = rng.randint(*sw["era_duration_days"])
        sw_end = sw_start + timedelta(days=sw_dur)
        rows.append({
            "person_id": person_id,
            "drug_concept_id": sw["concept_id"],
            "ingredient_concept_id": sw["concept_id"],
            "ingredient_name": sw["concept_name"],
            "atc_group": sw["atc_group"],
            "archetype": archetype_name + "_switch_target",
            "drug_era_start_date": sw_start,
            "drug_era_end_date": sw_end,
            "drug_exposure_count": rng.randint(1, 4),
            "gap_days": 0,
        })

    # Additional drugs for polypharmacy archetypes
    if "additional_drugs" in arch:
        for add_drug in arch["additional_drugs"]:
            offset = rng.randint(*add_drug["start_offset_days"])
            add_start = index_date + timedelta(days=offset)
            add_dur = rng.randint(*add_drug["era_duration_days"])
            add_end = add_start + timedelta(days=add_dur)
            rows.append({
                "person_id": person_id,
                "drug_concept_id": add_drug["concept_id"],
                "ingredient_concept_id": add_drug["concept_id"],
                "ingredient_name": add_drug["concept_name"],
                "atc_group": "polypharmacy_additional",
                "archetype": archetype_name + "_additional",
                "drug_era_start_date": add_start,
                "drug_era_end_date": add_end,
                "drug_exposure_count": rng.randint(1, 6),
                "gap_days": 0,
            })

    return rows


def generate_cohort(n_patients=1000, seed=42, follow_up_months=24):
    """
    Generate a complete synthetic cohort in OMOP drug_era schema.

    Returns
    -------
    drug_era_df : pd.DataFrame
    observation_period_df : pd.DataFrame
    person_df : pd.DataFrame
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    all_eras = []
    obs_rows = []
    person_rows = []

    # Arbitrary cohort start window: 2020-01-01 to 2020-12-31
    cohort_start = date(2020, 1, 1)

    for pid in range(1, n_patients + 1):
        # Random index date within the cohort window
        days_offset = rng.randint(0, 364)
        index_date = cohort_start + timedelta(days=days_offset)
        censor_date = index_date + timedelta(days=follow_up_months * 30)

        # Demographics (simple)
        year_of_birth = rng.randint(1940, 1990)
        gender = rng.choice(["M", "F"])

        person_rows.append({
            "person_id": pid,
            "year_of_birth": year_of_birth,
            "gender_concept_id": 8507 if gender == "M" else 8532,
            "observation_period_start_date": index_date - timedelta(days=365),
            "observation_period_end_date": censor_date,
        })

        obs_rows.append({
            "person_id": pid,
            "observation_period_start_date": index_date - timedelta(days=365),
            "observation_period_end_date": censor_date,
        })

        # Sample archetype and generate eras
        archetype = sample_archetype(rng)
        eras = generate_eras_for_archetype(pid, archetype, index_date, rng)

        # Clip to follow-up window
        for era in eras:
            if era["drug_era_start_date"] > censor_date:
                continue
            era["drug_era_end_date"] = min(era["drug_era_end_date"], censor_date)
            all_eras.append(era)

    drug_era_df = pd.DataFrame(all_eras)
    obs_df = pd.DataFrame(obs_rows)
    person_df = pd.DataFrame(person_rows)

    return drug_era_df, obs_df, person_df


def main():
    parser = argparse.ArgumentParser(description="Generate Medstat-calibrated synthetic OMOP cohort")
    parser.add_argument("--n_patients", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="data/synthetic_medstat")
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating {args.n_patients} patients with seed {args.seed}...")
    drug_era_df, obs_df, person_df = generate_cohort(
        n_patients=args.n_patients,
        seed=args.seed,
    )

    drug_era_df.to_parquet(f"{args.output_dir}/synthetic_drug_era.parquet", index=False)
    obs_df.to_parquet(f"{args.output_dir}/synthetic_observation_period.parquet", index=False)
    person_df.to_parquet(f"{args.output_dir}/synthetic_person.parquet", index=False)

    # Summary statistics (should roughly match Medstat prevalence rates)
    print("\n=== Archetype distribution ===")
    if not drug_era_df.empty:
        archetype_counts = (
            drug_era_df
            .drop_duplicates(subset=["person_id", "archetype"])
            .groupby("archetype")["person_id"].nunique()
            .reset_index()
        )
        archetype_counts["prevalence_pct"] = (
            archetype_counts["person_id"] / args.n_patients * 100
        ).round(1)
        print(archetype_counts.to_string(index=False))

    print(f"\nTotal drug eras: {len(drug_era_df)}")
    print(f"Patients with any medication: {drug_era_df['person_id'].nunique()}")
    print(f"\nOutput written to: {args.output_dir}/")
    print("Files: synthetic_drug_era.parquet, synthetic_observation_period.parquet, synthetic_person.parquet")
    print("\nTo run the pipeline on this data, update your YAML config:")
    print(f"  drug_era_path: {args.output_dir}/synthetic_drug_era.parquet")
    print(f"  observation_period_path: {args.output_dir}/synthetic_observation_period.parquet")


if __name__ == "__main__":
    main()
