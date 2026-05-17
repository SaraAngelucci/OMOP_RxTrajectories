"""
prepare_external_omop.py
------------------------
Takes external OMOP CDM v5.x dataset (Eunomia, Synthea-OMOP, or any
other OMOP-compatible source) and convert it to the parquet schema
for ``main.py``.

The pipeline itself requires no changes; only need to harmonise the input table layout.

Expected input directory contents (CSV or Parquet):
    DRUG_ERA.csv             (required)  OMOP drug_era table
    OBSERVATION_PERIOD.csv   (required)  OMOP observation_period table
    PERSON.csv               (optional)  OMOP person table (for demographics)
    CONCEPT.csv              (optional)  for human-readable ingredient names

Outputs (written to --output-dir, default data/external_omop_<label>/):
    synthetic_drug_era.parquet
    synthetic_observation_period.parquet
    synthetic_person.parquet (if PERSON was provided)

Plus a config YAML at config/config_external_<label>.yaml pointing at
the new files, so the existing ``main.py`` can run unchanged via:
    SPARK_CONFIG_PATH=config/config_external_<label>.yaml python main.py

Usage:
    # Tier 1  Eunomia (after SQLite has been extracted to CSV)
    python prepare_external_omop.py \\
        --input-dir ~/Downloads/Eunomia_csv \\
        --label eunomia \\
        --format csv

    # Tier 2  Synthea-OMOP (CSV downloads from ftp.ohdsi.org)
    python prepare_external_omop.py \\
        --input-dir ~/Downloads/synthea_omop \\
        --label synthea \\
        --format csv
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import pandas as pd


# OMOP DRUG_ERA columns expected in the source.  CDM v5.3 and v5.4 are
# both supported because the columns read are stable across versions.
DRUG_ERA_REQUIRED = [
    "person_id",
    "drug_concept_id",
    "drug_era_start_date",
    "drug_era_end_date",
]

# Optional columns to map if present; otherwise synthesize a default.
DRUG_ERA_OPTIONAL = {
    "drug_exposure_count": 1,
    "gap_days": 0,
}

OBSPERIOD_REQUIRED = [
    "person_id",
    "observation_period_start_date",
    "observation_period_end_date",
]

PERSON_REQUIRED = [
    "person_id",
    "year_of_birth",
]


def _read(path: Path, fmt: str) -> pd.DataFrame:
    """Format-agnostic loader for a single OMOP table."""
    if fmt == "csv":
        return pd.read_csv(path, low_memory=False)
    elif fmt == "parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unknown format: {fmt}")


def find_table(input_dir: Path, table_name: str, fmt: str) -> Path | None:
    """Locate OMOP table file ``TABLE_NAME.ext`` case-insensitively."""
    target_ext = ".csv" if fmt == "csv" else ".parquet"
    candidates = [
        input_dir / f"{table_name}{target_ext}",
        input_dir / f"{table_name.lower()}{target_ext}",
        input_dir / f"{table_name.upper()}{target_ext}",
    ]
    for c in candidates:
        if c.exists():
            return c
    matches = list(input_dir.glob(f"*{target_ext}"))
    for m in matches:
        if m.stem.lower() == table_name.lower():
            return m
    return None


_DRUG_ERA_PIPELINE_COLS = [
    "person_id",
    "drug_concept_id",
    "ingredient_concept_id",
    "ingredient_name",
    "atc_group",
    "archetype",
    "drug_era_start_date",
    "drug_era_end_date",
    "drug_exposure_count",
    "gap_days",
]


def _project_to_pipeline_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Add the placeholder columns the rest of the pipeline expects and
    project to the canonical column order."""
    df["ingredient_concept_id"] = df["drug_concept_id"]
    df["ingredient_name"] = None
    df["atc_group"] = None
    df["archetype"] = "external_omop"
    df["drug_era_start_date"] = pd.to_datetime(df["drug_era_start_date"]).dt.date
    df["drug_era_end_date"] = pd.to_datetime(df["drug_era_end_date"]).dt.date
    return df[_DRUG_ERA_PIPELINE_COLS]


def eraize_from_drug_exposure(input_dir: Path, fmt: str, gap_days: int = 30) -> pd.DataFrame:
    """Synthesise DRUG_ERA rows from DRUG_EXPOSURE.

    Defensive fallback for OMOP CDM distributions that ship without a
    pre-computed DRUG_ERA table ( OHDSI's ``Synthea27Nj_5.4``
    demo, which carries DRUG_EXPOSURE but an empty DRUG_ERA).
    implement the standard OHDSI gap-rule eraization: rows with identical
    ``(person_id, drug_concept_id)`` are sorted by start date and
    merged into a single era whenever the gap between the previous era's
    end and the next exposure's start does not exceed ``gap_days``
    (default 30 days, the OHDSI canonical value).

    Note:  ``drug_exposure.drug_concept_id`` treated as the era-level
    ingredient. OMOP CDM v5.x expects DRUG_ERA's
    drug_concept_id to be the ingredient concept obtained by ascending
    CONCEPT_ANCESTOR; pre-shipped DRUG_ERA tables (e.g. Eunomia
    GiBleed) do that.  but it is not a replacement for a proper Achilles/ETL-Synthea eraization run.
    """
    path = find_table(input_dir, "drug_exposure", fmt)
    if path is None:
        raise FileNotFoundError(
            f"Neither DRUG_ERA nor DRUG_EXPOSURE found in {input_dir}; "
            "cannot fall back to eraization."
        )
    print(f"  Eraizing from DRUG_EXPOSURE: {path} (gap rule: <= {gap_days} days) ...")
    df = _read(path, fmt)
    df.columns = [c.lower() for c in df.columns]

    required = ["person_id", "drug_concept_id", "drug_exposure_start_date"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"DRUG_EXPOSURE missing required columns: {miss}")

    if "drug_exposure_end_date" not in df.columns:
        df["drug_exposure_end_date"] = df["drug_exposure_start_date"]

    df["drug_exposure_start_date"] = pd.to_datetime(df["drug_exposure_start_date"])
    df["drug_exposure_end_date"] = pd.to_datetime(df["drug_exposure_end_date"])
    df["drug_exposure_end_date"] = df["drug_exposure_end_date"].fillna(df["drug_exposure_start_date"])

    df = df.sort_values(["person_id", "drug_concept_id", "drug_exposure_start_date"]).reset_index(drop=True)

    one_day = pd.Timedelta(days=1)
    eras = []
    for (pid, did), grp in df.groupby(["person_id", "drug_concept_id"], sort=False):
        rows = list(zip(grp["drug_exposure_start_date"], grp["drug_exposure_end_date"]))
        cur_start, cur_end = rows[0]
        cnt = 1
        for s, e in rows[1:]:
            if (s - cur_end) / one_day <= gap_days:
                if e > cur_end:
                    cur_end = e
                cnt += 1
            else:
                eras.append((pid, did, cur_start, cur_end, cnt))
                cur_start, cur_end, cnt = s, e, 1
        eras.append((pid, did, cur_start, cur_end, cnt))

    out = pd.DataFrame(eras, columns=[
        "person_id", "drug_concept_id",
        "drug_era_start_date", "drug_era_end_date", "drug_exposure_count",
    ])
    out["gap_days"] = 0
    print(
        f"  Eraised {len(df):,} DRUG_EXPOSURE rows into {len(out):,} eras "
        f"for {out['person_id'].nunique():,} persons."
    )
    return _project_to_pipeline_schema(out)


def load_drug_era(input_dir: Path, fmt: str) -> pd.DataFrame:
    """Load OMOP DRUG_ERA and project to pipeline-compatible schema.

    If DRUG_ERA is missing or empty (header-only), automatically falls
    back to eraising from DRUG_EXPOSURE using the OHDSI 30-day gap rule
    (see :func:`eraize_from_drug_exposure`).
    """
    path = find_table(input_dir, "drug_era", fmt)
    if path is None:
        print("DRUG_ERA file not found.")
        return eraize_from_drug_exposure(input_dir, fmt)

    print(f"Reading DRUG_ERA from {path} ...")
    df = _read(path, fmt)
    df.columns = [c.lower() for c in df.columns]

    if len(df) == 0:
        print(f"  DRUG_ERA at {path} is empty (header-only).")
        return eraize_from_drug_exposure(input_dir, fmt)

    missing = [c for c in DRUG_ERA_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"DRUG_ERA is missing required columns: {missing}")

    for col, default in DRUG_ERA_OPTIONAL.items():
        if col not in df.columns:
            df[col] = default

    return _project_to_pipeline_schema(df)


def load_observation_period(input_dir: Path, fmt: str) -> pd.DataFrame:
    path = find_table(input_dir, "observation_period", fmt)
    if path is None:
        raise FileNotFoundError(f"OBSERVATION_PERIOD.{fmt} not found in {input_dir}")
    print(f"Reading OBSERVATION_PERIOD from {path} ...")
    df = _read(path, fmt)
    df.columns = [c.lower() for c in df.columns]

    missing = [c for c in OBSPERIOD_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"OBSERVATION_PERIOD missing required cols: {missing}")

    df["observation_period_start_date"] = pd.to_datetime(df["observation_period_start_date"]).dt.date
    df["observation_period_end_date"] = pd.to_datetime(df["observation_period_end_date"]).dt.date

    return df[OBSPERIOD_REQUIRED]


def load_person(input_dir: Path, fmt: str) -> pd.DataFrame | None:
    path = find_table(input_dir, "person", fmt)
    if path is None:
        print("PERSON table not provided; skipping person export.")
        return None
    print(f"Reading PERSON from {path} ...")
    df = _read(path, fmt)
    df.columns = [c.lower() for c in df.columns]
    if "year_of_birth" not in df.columns:
        df["year_of_birth"] = 1970  # placeholder; not used by the pipeline
    if "gender_concept_id" not in df.columns:
        df["gender_concept_id"] = 0  # placeholder; not used by the pipeline
    return df[["person_id", "year_of_birth", "gender_concept_id"]]


def write_config_yaml(label: str, output_dir: Path) -> Path:
    """Emit a config YAML that points the existing main.py at the new data."""
    yaml_text = textwrap.dedent(f"""\
        project:
          name: "trajectory-pipeline-external-{label}"
          output_dir: "./outputs/external_{label}"

        paths:
          raw_dir: "{output_dir.as_posix()}"
          vocab_dir: "./synthetic_data"

        files:
          drug_era:           "synthetic_drug_era.parquet"
          observation_period: "synthetic_observation_period.parquet"
          person:             "synthetic_person.parquet"
          concept:            "CONCEPT.csv"
          concept_ancestor:   "concept_ancestor.csv"
          drug_exposure:      "omop_drug_exposure.csv.gz"
          death:              "omop_death.csv.gz"

        csv:
          header: true
          inferSchema: true
          concept_sep: ","

        analysis:
          washout_days:               365
          followup_months:            24
          exposure_gap_days:          30
          maintenance_min_total_days: 28
          early_discontinuation_days: 90
          restart_window_days:        180
          switch_window_days:         60
          polypharmacy_threshold:     5
          turnover_low:               0.20
          turnover_high:              0.50

        clustering:
          k_grid: [2, 3, 4, 5, 6]
          seed: 42

        run:
          run_exposure_sensitivity:         false
          save_top_ingredients_per_cluster: 10
        """)
    out = Path("config") / f"config_external_{label}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml_text)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="OMOP CDM data dir")
    parser.add_argument(
        "--label",
        required=True,
        help="short label for this dataset (e.g. eunomia, synthea, synpuf)",
    )
    parser.add_argument(
        "--format",
        default="csv",
        choices=["csv", "parquet"],
        help="format of the OMOP source tables",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="output parquet directory (default: data/external_omop_<label>)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        print(f"ERROR: input dir does not exist: {input_dir}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (
        Path("data") / f"external_omop_{args.label}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    drug_era = load_drug_era(input_dir, args.format)
    obs_period = load_observation_period(input_dir, args.format)
    person = load_person(input_dir, args.format)

    drug_era.to_parquet(output_dir / "synthetic_drug_era.parquet", index=False)
    obs_period.to_parquet(output_dir / "synthetic_observation_period.parquet", index=False)
    if person is not None:
        person.to_parquet(output_dir / "synthetic_person.parquet", index=False)

    n_persons = drug_era["person_id"].nunique()
    n_eras = len(drug_era)
    print()
    print(f"Wrote {n_eras} drug eras for {n_persons} persons to {output_dir}")

    cfg_path = write_config_yaml(args.label, output_dir)
    print(f"Wrote pipeline config to {cfg_path}")

    print()
    print("Next steps:")
    print(f"  SPARK_CONFIG_PATH={cfg_path} python main.py 2>&1 | tee outputs/external_{args.label}/run.log")
    print()
    print("Note: Experiment 2 (negative-control validation) will report")
    print("'Found 0 negative control patients' on external data because")
    print("ground-truth archetypes do not exist; this is expected behaviour.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
