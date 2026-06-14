"""
make_plots.py
=============

Generate the silhouette figure referenced in the thesis Results chapter:

* ``fig_silhouette_grid.png`` -- stratified silhouette width as a function
  of the candidate K-means cluster count *k* (Experiment 1).

The input is a CSV file produced by ``main.py``. It defaults to the
primary-cohort artefact:

    outputs/synthetic/baseline_cohort_silhouette_grid.csv

Override the path with ``--silhouette-csv`` to render figures for the 50k
or external runs (e.g. ``outputs/external_eunomia/``).

When clustering is withheld at runtime (e.g. on MIMIC-IV ICU, where the
safe-failure guard skips K-means entirely), the silhouette CSV is empty.
This script then skips the silhouette plot. The silhouette fallback values
embedded below correspond to the thesis-reported primary-cohort numbers and
are used solely when the CSV is missing on a fresh checkout.

Usage
-----
    python make_plots.py
    python make_plots.py --silhouette-csv outputs/synthetic_50k/baseline_cohort_silhouette_grid.csv \\
                        --figures-dir NEWAngelucci_draft_thesis/figures_50k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_OUTPUTS_DIR = Path("outputs/synthetic")
DEFAULT_SILHOUETTE_CSV = DEFAULT_OUTPUTS_DIR / "baseline_cohort_silhouette_grid.csv"

THESIS_FIGURE_DIRS = [
    Path("NEWAngelucci_draft_thesis/figures"),
    Path("Angelucci_draft_thesis/figures"),
]


# ---------------------------------------------------------------------------
# Thesis-reported values (used only when the CSV artefacts are unavailable)
# ---------------------------------------------------------------------------
# Fallback used only when the silhouette CSV is unavailable. These are the
# cosine-distance silhouette values produced by the current pipeline
# (ClusteringEvaluator(distanceMeasure="cosine"), matching the L2-normalised
# K-means objective). The curve increases monotonically across the grid, so
# the silhouette optimum sits at k=6; the operational partition is fixed at
# k=4 for clinical interpretability (see ERRATA_AND_CHANGES.md). Replace these
# by re-running ``python main.py`` and letting this script read the fresh CSV.
SILHOUETTE_FALLBACK = pd.DataFrame(
    {
        "label": ["baseline_cohort"] * 5,
        "k": [2, 3, 4, 5, 6],
        "silhouette": [0.5215, 0.6182, 0.6554, 0.7115, 0.7354],
    }
)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_silhouette(df: pd.DataFrame, out_path: Path) -> None:
    """Render the silhouette-vs-k line plot."""
    df = df.sort_values("k")
    optimal_row = df.loc[df["silhouette"].idxmax()]

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    ax.plot(df["k"], df["silhouette"], marker="o", linewidth=2.0, color="#2b6cb0")
    ax.scatter(
        [optimal_row["k"]],
        [optimal_row["silhouette"]],
        s=140,
        facecolor="white",
        edgecolor="#c53030",
        linewidth=2.5,
        zorder=5,
        label=f"Optimal: k={int(optimal_row['k'])}, silhouette={optimal_row['silhouette']:.4f}",
    )

    ax.set_xlabel("Candidate cluster count, $k$")
    ax.set_ylabel("Stratified silhouette width")
    ax.set_title("K-means silhouette across candidate $k$ on the calibrated synthetic cohort")
    ax.set_xticks(df["k"].astype(int).tolist())
    ax.legend(loc="upper right", frameon=True)

    ax.set_ylim(0, max(0.7, optimal_row["silhouette"] * 1.1))

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> wrote {out_path}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def resolve_figures_dir(override: str | None) -> Path:
    """Find or create the active thesis figures directory."""
    if override:
        target = Path(override)
        target.mkdir(parents=True, exist_ok=True)
        return target
    for candidate in THESIS_FIGURE_DIRS:
        if candidate.is_dir():
            return candidate
    # Last resort: create the canonical name
    fallback = THESIS_FIGURE_DIRS[0]
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def load_silhouette(csv_path: Path) -> tuple[pd.DataFrame | None, str]:
    """Read silhouette CSV, fall back to thesis values, or return None if
    clustering was skipped entirely (empty CSV)."""
    if csv_path.is_file() and csv_path.stat().st_size > 5:
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            return None, f"empty file at {csv_path} (clustering was skipped)"
        if {"k", "silhouette"}.issubset(df.columns) and len(df) > 0:
            return df, f"read from {csv_path}"
        return None, f"no usable rows in {csv_path} (clustering was skipped)"
    if csv_path.is_file():
        return None, f"empty file at {csv_path} (clustering was skipped)"
    print(
        f"!! Could not read a usable silhouette grid from {csv_path}.\n"
        f"   Falling back to the values reported in the thesis manuscript.\n"
        f"   Re-run `python main.py` to overwrite this with measured values."
    )
    return SILHOUETTE_FALLBACK.copy(), "thesis fallback"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silhouette-csv", default=str(DEFAULT_SILHOUETTE_CSV))
    parser.add_argument("--figures-dir", default=None)
    args = parser.parse_args()

    figures_dir = resolve_figures_dir(args.figures_dir)
    print(f"== Figures directory: {figures_dir}")

    silhouette_df, sil_source = load_silhouette(Path(args.silhouette_csv))
    print(f"== Silhouette data: {sil_source}")
    if silhouette_df is not None:
        plot_silhouette(silhouette_df, figures_dir / "fig_silhouette_grid.png")
    else:
        print("   (silhouette plot skipped — no clustering data available)")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
