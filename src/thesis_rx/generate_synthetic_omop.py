"""
generate_synthetic_omop.py
==========================
Generates a small but realistic synthetic OMOP dataset to validate the
pipeline end-to-end without access to real data.

Produces:
  out_dir/omop_drug_era.csv.gz
  out_dir/omop_drug_exposure.csv.gz
  out_dir/omop_observation_period.csv.gz
  out_dir/omop_death.csv.gz
  out_dir/CONCEPT.csv
  out_dir/concept_ancestor.csv

Usage:
  python generate_synthetic_omop.py --out-dir ./synthetic_data --n-persons 5000
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

RNG = np.random.default_rng(42)

# Fictional ingredient concept IDs (representing common psychiatric drugs)
INGREDIENTS = {
    1_000_001: "sertraline",
    1_000_002: "fluoxetine",
    1_000_003: "citalopram",
    1_000_004: "quetiapine",
    1_000_005: "risperidone",
    1_000_006: "olanzapine",
    1_000_007: "lithium",
    1_000_008: "valproate",
    1_000_009: "lorazepam",
    1_000_010: "diazepam",
}

# Each ingredient has fictional descendant drug product concept IDs
INGREDIENT_TO_PRODUCTS = {
    ing_id: [ing_id * 100 + i for i in range(1, 4)]
    for ing_id in INGREDIENTS
}


def random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=int(RNG.integers(0, delta)))


def generate_persons(n: int):
    obs_start = date(2000, 1, 1)
    obs_end   = date(2022, 12, 31)
    rows = []
    for pid in range(1, n + 1):
        start = random_date(obs_start, date(2015, 1, 1))
        end   = random_date(start + timedelta(days=730), obs_end)
        rows.append({
            "person_id": pid,
            "observation_period_start_date": start,
            "observation_period_end_date": end
        })
    return pd.DataFrame(rows)


def generate_drug_exposures(persons_df: pd.DataFrame, n_exposures_per_person: tuple = (1, 20)):
    ing_ids  = list(INGREDIENTS.keys())
    rows = []
    for _, p in persons_df.iterrows():
        pid      = int(p["person_id"])
        obs_s    = p["observation_period_start_date"]
        obs_e    = p["observation_period_end_date"]
        span     = (obs_e - obs_s).days
        n_exp    = int(RNG.integers(*n_exposures_per_person))
        # Each person is assigned 1-3 main ingredients
        n_ing    = int(RNG.integers(1, min(4, len(ing_ids))))
        my_ings  = RNG.choice(ing_ids, size=n_ing, replace=False).tolist()

        for _ in range(n_exp):
            ing_id   = int(RNG.choice(my_ings))
            products = INGREDIENT_TO_PRODUCTS[ing_id]
            prod_id  = int(RNG.choice(products))
            start_d  = obs_s + timedelta(days=int(RNG.integers(0, max(1, span - 60))))
            supply   = int(RNG.choice([28, 30, 56, 84, 90]))
            end_d    = start_d + timedelta(days=supply - 1)
            if end_d > obs_e:
                end_d = obs_e
            rows.append({
                "drug_exposure_id":          len(rows) + 1,
                "person_id":                 pid,
                "drug_concept_id":           prod_id,
                "drug_exposure_start_date":  start_d,
                "drug_exposure_end_date":    end_d,
                "drug_type_concept_id":      38000177,
                "days_supply":               supply,
                "quantity":                  float(supply),
            })
    return pd.DataFrame(rows)


def generate_drug_eras(exposures_df: pd.DataFrame, gap_days: int = 30):
    """Merge exposures into eras by ingredient, per person."""
    prod_to_ing = {
        prod: ing
        for ing, prods in INGREDIENT_TO_PRODUCTS.items()
        for prod in prods
    }

    exp = exposures_df.copy()
    exp["ingredient_concept_id"] = exp["drug_concept_id"].map(prod_to_ing)
    exp = exp.dropna(subset=["ingredient_concept_id"])
    exp["ingredient_concept_id"] = exp["ingredient_concept_id"].astype(int)
    exp = exp.sort_values(["person_id", "ingredient_concept_id", "drug_exposure_start_date"])

    eras = []
    era_id = 1
    for (pid, ing_id), grp in exp.groupby(["person_id", "ingredient_concept_id"]):
        cur_start = None
        cur_end   = None
        cur_count = 0
        for _, row in grp.iterrows():
            s = row["drug_exposure_start_date"]
            e = row["drug_exposure_end_date"]
            if cur_start is None:
                cur_start, cur_end, cur_count = s, e, 1
            elif s <= cur_end + timedelta(days=gap_days):
                cur_end   = max(cur_end, e)
                cur_count += 1
            else:
                eras.append({
                    "drug_era_id":              era_id,
                    "person_id":                pid,
                    "ingredient_concept_id":    ing_id,
                    "drug_era_start_date":      cur_start,
                    "drug_era_end_date":        cur_end,
                    "drug_exposure_count":      cur_count,
                    "gap_days":                 gap_days
                })
                era_id   += 1
                cur_start, cur_end, cur_count = s, e, 1
        if cur_start is not None:
            eras.append({
                "drug_era_id":              era_id,
                "person_id":                pid,
                "ingredient_concept_id":    ing_id,
                "drug_era_start_date":      cur_start,
                "drug_era_end_date":        cur_end,
                "drug_exposure_count":      cur_count,
                "gap_days":                 gap_days
            })
            era_id += 1
    return pd.DataFrame(eras)


def generate_deaths(persons_df: pd.DataFrame, death_rate: float = 0.03):
    rows = []
    for _, p in persons_df.iterrows():
        if RNG.random() < death_rate:
            obs_e = p["observation_period_end_date"]
            death_d = obs_e - timedelta(days=int(RNG.integers(0, 365)))
            rows.append({"person_id": int(p["person_id"]), "death_date": death_d})
    return pd.DataFrame(rows)


def generate_concept_tables():
    concepts = []
    ancestors = []

    for ing_id, ing_name in INGREDIENTS.items():
        # Ingredient concept
        concepts.append({
            "concept_id":    ing_id,
            "concept_name":  ing_name,
            "domain_id":     "Drug",
            "vocabulary_id": "RxNorm",
            "concept_class_id": "Ingredient",
            "standard_concept": "S",
            "concept_code":  str(ing_id),
            "valid_start_date": "1970-01-01",
            "valid_end_date":   "2099-12-31",
            "invalid_reason":   None
        })
        # Self-ancestor
        ancestors.append({
            "ancestor_concept_id":   ing_id,
            "descendant_concept_id": ing_id,
            "min_levels_of_separation": 0,
            "max_levels_of_separation": 0
        })
        # Product concepts and their ancestry to ingredient
        for prod_id in INGREDIENT_TO_PRODUCTS[ing_id]:
            concepts.append({
                "concept_id":    prod_id,
                "concept_name":  f"{ing_name} tablet",
                "domain_id":     "Drug",
                "vocabulary_id": "RxNorm",
                "concept_class_id": "Clinical Drug",
                "standard_concept": "S",
                "concept_code":  str(prod_id),
                "valid_start_date": "1970-01-01",
                "valid_end_date":   "2099-12-31",
                "invalid_reason":   None
            })
            ancestors.append({
                "ancestor_concept_id":   ing_id,
                "descendant_concept_id": prod_id,
                "min_levels_of_separation": 1,
                "max_levels_of_separation": 1
            })

    return pd.DataFrame(concepts), pd.DataFrame(ancestors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="./synthetic_data")
    parser.add_argument("--n-persons", type=int, default=5000)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic OMOP data for {args.n_persons} persons...")

    persons   = generate_persons(args.n_persons)
    exposures = generate_drug_exposures(persons)
    eras      = generate_drug_eras(exposures)
    deaths    = generate_deaths(persons)
    concepts, ancestors = generate_concept_tables()

    persons.to_csv(out / "omop_observation_period.csv.gz",  index=False, compression="gzip")
    exposures.to_csv(out / "omop_drug_exposure.csv.gz",     index=False, compression="gzip")
    eras.to_csv(out / "omop_drug_era.csv.gz",               index=False, compression="gzip")
    deaths.to_csv(out / "omop_death.csv.gz",                index=False, compression="gzip")
    concepts.to_csv(out / "CONCEPT.csv",                    index=False)
    ancestors.to_csv(out / "concept_ancestor.csv",          index=False)

    print(f"Done. Files written to {out}/")
    print(f"  Persons:   {len(persons)}")
    print(f"  Exposures: {len(exposures)}")
    print(f"  Eras:      {len(eras)}")
    print(f"  Deaths:    {len(deaths)}")


if __name__ == "__main__":
    main()
