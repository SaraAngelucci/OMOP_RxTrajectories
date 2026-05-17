"""
compute_ari_matrix.py
---------------------
Standalone recovery utility.

Recomputes the 9x9 Adjusted Rand Index sensitivity matrix from the
``grid_run_*_person_level_phenotypes.parquet`` artefacts that
``main.py`` writes to ``cfg['project']['output_dir']`` during
Experiment 3.

Running this script is equivalent to re-running Experiment 3 but takes ~5 seconds instead of ~3 hours,
because the per-configuration cluster labels are already on disk.

Usage:
    python compute_ari_matrix.py
    python compute_ari_matrix.py --input-dir /path/to/outputs/synthetic
    python compute_ari_matrix.py --input-dir ... --param-grid \\
        polypharmacy_threshold=4,5,6 maintenance_min_total_days=14,28,56
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score


# The grid order used by ``main.py``.  Must match ``param_grid`` in
# ``main.py::main()`` exactly so labels in the output CSV are correct.
DEFAULT_PARAM_GRID = {
    "polypharmacy_threshold": [4, 5, 6],
    "maintenance_min_total_days": [14, 28, 56],
}


def discover_grid_runs(input_dir: Path) -> list[Path]:
    """Return the 9 person-level-phenotype parquets sorted by mtime.

    Sorting by mtime preserves the original orchestrator creation order
    because :func:`run_sensitivity_grid` writes each configuration's
    artefacts sequentially.
    """
    files = sorted(
        input_dir.glob("grid_run_*_person_level_phenotypes.parquet"),
        key=lambda p: p.stat().st_mtime,
    )
    return files


def build_config_labels(param_grid: dict) -> list[str]:
    """Cartesian-product labels exactly as ``main.py`` constructs them."""
    names = list(param_grid.keys())
    values = list(param_grid.values())
    labels = []
    for combo in itertools.product(*values):
        # Short form: "Poly:4_Maint:14d" (matches main.py)
        parts = []
        for name, value in zip(names, combo):
            if name == "polypharmacy_threshold":
                parts.append(f"Poly:{value}")
            elif name == "maintenance_min_total_days":
                parts.append(f"Maint:{value}d")
            else:
                parts.append(f"{name}:{value}")
        labels.append("_".join(parts))
    return labels


def load_phenotypes(parquet_path: Path) -> pd.DataFrame:
    """Load (person_id, discontinuation_phenotype) from a grid-run parquet."""
    df = pd.read_parquet(parquet_path, columns=["person_id", "discontinuation_phenotype"])
    return df


def pairwise_ari(label_vectors: list[pd.DataFrame], labels: list[str]) -> pd.DataFrame:
    """Compute the n x n pairwise ARI matrix, aligned on person_id."""
    n = len(label_vectors)
    ari_matrix = [[1.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            merged = label_vectors[i].merge(
                label_vectors[j], on="person_id", suffixes=("_i", "_j")
            )
            if merged.empty:
                ari_matrix[i][j] = ari_matrix[j][i] = float("nan")
                continue
            ari = adjusted_rand_score(
                merged["discontinuation_phenotype_i"],
                merged["discontinuation_phenotype_j"],
            )
            ari_matrix[i][j] = ari_matrix[j][i] = ari

    return pd.DataFrame(ari_matrix, index=labels, columns=labels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="outputs/synthetic",
        help="Directory containing grid_run_*_person_level_phenotypes.parquet files",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <input-dir>/ari_sensitivity_matrix.csv)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        print(f"ERROR: input dir does not exist: {input_dir}", file=sys.stderr)
        return 1

    grid_runs = discover_grid_runs(input_dir)
    expected_n = len(list(itertools.product(*DEFAULT_PARAM_GRID.values())))
    print(f"Found {len(grid_runs)} grid-run parquets in {input_dir}")
    for i, p in enumerate(grid_runs):
        print(f"  config {i+1}: {p.name}")

    if len(grid_runs) != expected_n:
        print(
            f"WARNING: expected {expected_n} grid-run parquets (3x3 grid) "
            f"but found {len(grid_runs)}.  Output ARI matrix will be "
            f"{len(grid_runs)}x{len(grid_runs)}.",
            file=sys.stderr,
        )

    labels = build_config_labels(DEFAULT_PARAM_GRID)
    if len(labels) > len(grid_runs):
        labels = labels[: len(grid_runs)]

    print("\nLoading cluster-label vectors...")
    label_vectors = [load_phenotypes(p) for p in grid_runs]
    for lab, vec in zip(labels, label_vectors):
        print(f"  {lab}: {len(vec)} persons")

    print("\nComputing pairwise ARI...")
    ari_df = pairwise_ari(label_vectors, labels)

    output = Path(args.output) if args.output else input_dir / "ari_sensitivity_matrix.csv"
    ari_df.to_csv(output)

    print(f"\nWrote: {output}")
    print("\nARI matrix preview:")
    print(ari_df.round(3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
