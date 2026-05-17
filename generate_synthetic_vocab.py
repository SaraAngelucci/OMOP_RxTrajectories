"""
generate_synthetic_vocab.py
----------------------------
Minimal OMOP vocabulary under ``./synthetic_data/``:

* ``CONCEPT.csv``  ingredient (and optional RxNorm clinical-drug) rows
  referenced by ``generate_synthetic_cohort.py``.
* ``concept_ancestor.csv`` identity rows (ancestor =
  descendant) for every ``concept_id`` in ``CONCEPT.csv``.  Required because
  ``main.py`` always loads ``concept_ancestor`` via ``src/thesis_rx/io.py``
  before pipeline execution.

Without ``CONCEPT.csv``, ingredient names resolve to NULL in cluster
summaries; without ``concept_ancestor.csv``, Spark raises ``Path does not
exist`` on a fresh checkout.
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


def write_concept_ancestor_identity(concept_csv: Path, out_path: Path) -> None:
    """Write one OMOP-style self-ancestor row per concept_id in CONCEPT.csv."""
    concept_ids: list[int] = []
    with concept_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("concept_id")
            if cid is None or cid == "":
                continue
            concept_ids.append(int(cid))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "ancestor_concept_id",
        "descendant_concept_id",
        "min_levels_of_separation",
        "max_levels_of_separation",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for cid in concept_ids:
            writer.writerow([cid, cid, 0, 0])


def main() -> None:
    out_dir = Path("synthetic_data")
    out_path = out_dir / "CONCEPT.csv"
    rows = collect_ingredient_concepts()
    write_concept_csv(rows, out_path)
    ancestor_path = out_dir / "concept_ancestor.csv"
    write_concept_ancestor_identity(out_path, ancestor_path)
    print(f"Wrote {len(rows)} ingredient concepts to {out_path}")
    print(f"Wrote identity ancestor rows for all concepts to {ancestor_path}")
    for cid, cname in rows:
        print(f"  {cid}\t{cname}")


if __name__ == "__main__":
    main()
