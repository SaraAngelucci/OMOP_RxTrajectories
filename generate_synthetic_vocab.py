"""
generate_synthetic_vocab.py
----------------------------
Emit a minimal OMOP CONCEPT.csv stub that matches the ingredient_concept_id
values used by ``generate_synthetic_cohort.py``.

This is purely a presentation aid: without it, the pipeline still runs, but
``baseline_cohort_top_ingredients.parquet`` will contain NULL ingredient
names instead of human-readable strings.

Output: ./synthetic_data/CONCEPT.csv
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from generate_synthetic_cohort import DRUG_ARCHETYPES


def collect_ingredient_concepts() -> list[tuple[int, str]]:
    """Walk the archetype table and return the unique (id, name) pairs."""
    seen: dict[int, str] = {}
    for arch in DRUG_ARCHETYPES.values():
        cid = arch.get("concept_id")
        cname = arch.get("concept_name")
        if cid is not None and cname is not None:
            seen[int(cid)] = cname
        for add in arch.get("additional_drugs", []) or []:
            cid = add.get("concept_id")
            cname = add.get("concept_name")
            if cid is not None and cname is not None:
                seen[int(cid)] = cname
    return sorted(seen.items())


def write_concept_csv(rows: list[tuple[int, str]], out_path: Path) -> None:
    """Write the OMOP CONCEPT schema header plus the synthetic stub rows."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # OMOP v5.4 CONCEPT columns; only concept_id and concept_name are used by
    # the pipeline, but keeping the full schema header keeps the file
    # interpretable by any OMOP-aware tool.
    columns = [
        "concept_id",
        "concept_name",
        "domain_id",
        "vocabulary_id",
        "concept_class_id",
        "standard_concept",
        "concept_code",
        "valid_start_date",
        "valid_end_date",
        "invalid_reason",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for cid, cname in rows:
            writer.writerow([
                cid,
                cname,
                "Drug",
                "Synthetic",
                "Ingredient",
                "S",
                f"SYN_{cid}",
                "1970-01-01",
                "2099-12-31",
                "",
            ])


def main() -> None:
    out_path = Path("synthetic_data") / "CONCEPT.csv"
    rows = collect_ingredient_concepts()
    write_concept_csv(rows, out_path)
    print(f"Wrote {len(rows)} ingredient concepts to {out_path}")
    for cid, cname in rows:
        print(f"  {cid}\t{cname}")


if __name__ == "__main__":
    main()
