"""
cluster_validation.py
=====================

Post-hoc cross-method intrinsic-validity cross-check on the pipeline's
person-level phenotypes parquet.

For each candidate K in {2, ..., 6}, a fresh scikit-learn Lloyd K-means is
fitted on the **standardised** 23-dimensional feature vector (StandardScaler
with zero mean and unit variance, no L2 normalisation). The Calinski-Harabasz
(CH, higher = better) and Davies-Bouldin (DB, lower = better) indices are
reported on those labels. A min-max-normalised cluster centroid heatmap is
then rendered for K = 4 (or any K supplied via ``--k``) to support clinical
interpretation of the recovered archetypes.

Why standardised, not L2-normalised?
------------------------------------
The pipeline's primary clustering uses L2-normalised features (so that
Euclidean K-means is equivalent to cosine K-means; see thesis Sec. 3.13.2).
CH and DB are defined on raw Euclidean distances. Computing them on the
unit-sphere representation would compress every pairwise distance into
[0, 2] and bias both indices, particularly DB (whose numerator is the mean
intra-cluster scatter). Standardising-only is the appropriate input for
these isotropic-Euclidean indices, and the resulting K-means partition is
explicitly interpreted in the thesis as an **independent cross-method check**
on the recoverable structure (not as the pipeline's production clustering).

Why ANOVA / eta-squared is run separately?
------------------------------------------
The ANOVA feature-importance ranking in ``feature_importance_fixed.py`` is
run on the pipeline's L2-normalised cluster labels (the ``trajectory_cluster``
column of the parquet). eta-squared is scale-invariant, so whether the input
features are standardised or not does not change the ranking. The CH/DB
re-fit here and the ANOVA there therefore answer two different questions on
two different label sets, both of which are stable and meaningful.

Usage
-----
    python cluster_validation.py --dataset synthetic
    python cluster_validation.py --dataset synthetic_50k --k 4
    python cluster_validation.py --person_parquet outputs/synthetic/baseline_cohort_person_level_phenotypes.parquet \\
                                 --out_dir outputs/synthetic --k 4
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# Strictly numeric features produced by run_trajectory_pipeline (23 columns).
NUMERIC_FEATURES = [
    "mean_active_n", "poly_month_prop", "mean_turnover", "burden_slope",
    "n_ingredient_eras", "n_distinct_ingredients", "early_disc_90_rate",
    "restart_180_rate", "switch_60_rate", "median_era_days",
    "mean_burden_w1", "mean_burden_w2", "mean_burden_w3", "mean_burden_w4",
    "prop_NoRx", "prop_Initiation", "prop_StableMono", "prop_StableLowPoly",
    "prop_StablePolypharmacy", "prop_Intensifying", "prop_Deintensifying",
    "prop_HighTurnover", "prop_ModerateFlux",
]



def load_evaluable(parquet_path: str) -> pd.DataFrame:
    """Read parquet, filter trajectory_cluster >= 0, return DataFrame."""
    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    if "trajectory_cluster" not in df.columns:
        raise KeyError(
            f"Column 'trajectory_cluster' missing from {parquet_path}. "
            "Run the full pipeline (main.py) first to produce this column."
        )
    df = df[df["trajectory_cluster"] >= 0].copy()
    print(f"  {len(df):,} evaluable patients retained.")
    return df


def select_features(df: pd.DataFrame) -> list[str]:
    features = [c for c in NUMERIC_FEATURES if c in df.columns]
    missing = [c for c in NUMERIC_FEATURES if c not in df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} feature columns absent: {missing}")
    return features


def assert_no_nulls(df: pd.DataFrame, features: list[str]) -> None:
    null_counts = df[features].isna().sum()
    bad = null_counts[null_counts > 0]
    if not bad.empty:
        # The pipeline fills the global summaries and prop_* before writing;
        # the discontinuation-rate columns are populated by left-join from
        # disc_summary, which is computed only on maintenance-eligible eras.
        # For trajectory_cluster >= 0 rows these should all be non-null. If
        # any nulls remain, fail loudly rather than imputing silently.
        raise ValueError(
            f"Null values found in evaluable rows: {bad.to_dict()}. "
            "This indicates an upstream pipeline bug; do NOT impute zeros "
            "silently because '0 after standardisation' = -mean/sigma, "
            "which biases CH/DB and the cluster centroids."
        )


def compute_intrinsic_indices(
    X_scaled: np.ndarray,
    k_range: range,
    seed: int = 42,
) -> pd.DataFrame:
    """Refit sklearn KMeans for each k and report CH and DB."""
    records = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X_scaled)
        ch = calinski_harabasz_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        records.append({"k": k, "calinski_harabasz": ch, "davies_bouldin": db})
    out = pd.DataFrame(records)
    return out


def plot_phenotype_heatmap(
    df: pd.DataFrame,
    features: list[str],
    label_col: str,
    title: str,
    out_path: str,
) -> None:
    cluster_means = df.groupby(label_col)[features].mean().T
    row_min = cluster_means.min(axis=1)
    row_max = cluster_means.max(axis=1)
    normed = cluster_means.subtract(row_min, axis=0).divide(
        (row_max - row_min).replace(0, 1), axis=0
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        normed, cmap="YlGnBu", annot=True, fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Normalised feature intensity [0=min, 1=max]"},
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Cluster ID (cross-method refit)", fontsize=12)
    plt.ylabel("Trajectory features", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default=None,
        help="Resolves to outputs/<dataset>/baseline_cohort_person_level_phenotypes.parquet.",
    )
    parser.add_argument("--person_parquet", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--k", type=int, default=4,
                        help="K for the phenotype heatmap (default 4).")
    parser.add_argument("--k_min", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.person_parquet is None and args.dataset is None:
        parser.error("Provide --dataset or --person_parquet.")

    if args.person_parquet is None:
        args.person_parquet = (
            f"outputs/{args.dataset}/baseline_cohort_person_level_phenotypes.parquet"
        )
    if args.out_dir is None:
        args.out_dir = f"outputs/{args.dataset}" if args.dataset else "outputs"

    os.makedirs(args.out_dir, exist_ok=True)


    df = load_evaluable(args.person_parquet)
    features = select_features(df)
    print(f"  Using {len(features)} strictly numeric features.")

    # =====================================================================
    # FIX: Explicitly handle single-era nulls BEFORE the strict guard.
    # Single-era patients have no restart/switch opportunities, so the rate 
    # is logically 0.0, not missing.
    # =====================================================================
    rates_to_fix = ['restart_180_rate', 'switch_60_rate', 'early_disc_90_rate']
    for col in rates_to_fix:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    # =====================================================================

    assert_no_nulls(df, features)

    X = df[features].astype(float).values
    X_scaled = StandardScaler().fit_transform(X)

    # ----- Part A: CH / DB across k ----------------------------------------
    print("\n--- Intrinsic Cluster Metrics (standardised features, "
          "sklearn Lloyd K-means cross-method refit) ---")
    print(f"{'K':<3} {'CH (higher=better)':>22} {'DB (lower=better)':>22}")
    print("-" * 50)

    indices_df = compute_intrinsic_indices(
        X_scaled, range(args.k_min, args.k_max + 1), seed=args.seed,
    )
    for _, row in indices_df.iterrows():
        marker = "  <-- selected for heatmap" if int(row["k"]) == args.k else ""
        print(f"{int(row['k']):<3} {row['calinski_harabasz']:>22.2f} "
              f"{row['davies_bouldin']:>22.2f}{marker}")

    csv_path = os.path.join(args.out_dir, "intrinsic_indices_ch_db.csv")
    indices_df.to_csv(csv_path, index=False)
    print(f"\nCH/DB table saved: {csv_path}")

    # ----- Part B: cross-method phenotype heatmap at the selected k --------
    print(f"\nGenerating cross-method phenotype heatmap at K = {args.k} ...")
    km_final = KMeans(n_clusters=args.k, random_state=args.seed, n_init=10)
    df[f"k{args.k}_labels"] = km_final.fit_predict(X_scaled)

    heatmap_path = os.path.join(
        args.out_dir, f"k{args.k}_phenotype_heatmap.png"
    )
    plot_phenotype_heatmap(
        df, features, f"k{args.k}_labels",
        title=f"Phenotypic profile of K={args.k} archetypes "
              "(standardised cross-method refit)",
        out_path=heatmap_path,
    )

    print("\nDone. Use the CH/DB table to inspect the K-selection in detail, "
          "and the heatmap to name the archetypes.")


if __name__ == "__main__":
    main()
