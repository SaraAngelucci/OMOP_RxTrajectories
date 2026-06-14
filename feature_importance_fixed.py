"""
feature_importance_fixed.py
===========================

One-way ANOVA / eta-squared ranking of trajectory features by cluster label.
Operates on the pipeline's ``baseline_cohort_person_level_phenotypes.parquet``
output and produces (i) a CSV table of η², F, p per feature, (ii) a bar chart
of the top-N features, (iii) a per-cluster violin grid for the top features,
and (iv) a min-max-normalised cluster-centroid heatmap.

What labels does this analyse?
------------------------------
By default (``--cluster_col trajectory_cluster``), this script analyses the
pipeline's **L2-normalised Lloyd K-means** cluster labels persisted in the
parquet. These are the production labels: the K is the value forced via
``clustering.k_grid`` in the YAML config (e.g. ``[4]`` for the production
runs reported in the thesis Section 4.1.1 / 4.4.1).

Pass ``--cluster_col k4_labels`` (or similar) if you want to analyse the
**cross-method** Lloyd K-means refit on standardised (not L2-normalised)
features produced by ``cluster_validation.py``. Both label sets are
meaningful; the thesis reports both.

Mathematical notes
------------------
* η² = SS_between / SS_total is **scale-invariant**: standardising or
  L2-normalising the features does not change the ranking. We therefore
  compute the ANOVA on the **raw** features (as written by the pipeline)
  for full numerical transparency.
* Each feature's ANOVA uses its own row-wise non-null subset. The
  per-feature degrees of freedom are reported in the CSV.
* Cohen's threshold η² ≥ 0.14 marks a "large" effect; the bar chart
  highlights this threshold but should not be interpreted as a p-value test
  (with N in the thousands every η² > 0.001 is significant at p < 0.001;
  the ranking, not the p-value, is the substantive output).

Usage
─────
  python feature_importance_fixed.py \\
      --person_parquet outputs/synthetic/baseline_cohort_person_level_phenotypes.parquet \\
      --cluster_col    trajectory_cluster \\
      --out_dir        outputs/synthetic/feature_importance \\
      --top_n          10
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

PALETTE = sns.color_palette("tab10")

# These are the candidate feature names the pipeline may produce.
# The script auto-detects which ones are actually present AND numeric.

CANDIDATE_FEATURES = [
    "mean_active_n",           # was "mean_burden"
    "poly_month_prop",         # was "polypharmacy_month_prop"
    "mean_turnover",
    "burden_slope",
    "n_ingredient_eras",
    "n_distinct_ingredients",
    "early_disc_90_rate",      # was "early_disc_rate"
    "restart_180_rate",        # was "restart_rate"
    "switch_60_rate",          # was "switch_rate"
    "median_era_days",         # was "median_era_duration_days"
    "is_single_era_maintenance",  # new feature
    "mean_burden_w1",
    "mean_burden_w2",
    "mean_burden_w3",
    "mean_burden_w4",
    "prop_NoRx",
    "prop_Initiation",
    "prop_StableMono",
    "prop_StableLowPoly",
    "prop_StablePolypharmacy",
    "prop_Intensifying",
    "prop_Deintensifying",
    "prop_HighTurnover",
    "prop_ModerateFlux",
]


# ─────────────────────────────────────────────────────────────────────────────
def load_person(path: str, cluster_col: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if cluster_col not in df.columns:
        raise ValueError(
            f"Cluster column '{cluster_col}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )
    df = df[df[cluster_col] >= 0].copy()

    df[cluster_col] = df[cluster_col].astype(int)
    
    print(f"  {len(df):,} patients with cluster assignment kept")
    print(f"  Columns in parquet: {list(df.columns)}")
    return df


def get_numeric_features(df: pd.DataFrame) -> list[str]:
    """
    Return only columns that are (a) in CANDIDATE_FEATURES OR named like
    known pattern columns, AND (b) have a numeric dtype.
    Silently skips any string / categorical column.
    """
    numeric_cols = set(df.select_dtypes(include="number").columns.tolist())

    candidates = set(CANDIDATE_FEATURES)
    # also include any prop_ or mean_burden_w columns in the parquet
    for col in df.columns:
        if col.startswith("prop_") or col.startswith("mean_burden_w"):
            candidates.add(col)

    features = sorted(candidates & numeric_cols)

    skipped = sorted((candidates | numeric_cols) - set(features) - numeric_cols)
    if skipped:
        print(f"  Skipped {len(skipped)} non-numeric candidate columns: {skipped[:8]}")

    print(f"  Using {len(features)} numeric features for ANOVA")
    return features


# ─────────────────────────────────────────────────────────────────────────────
def anova_importance(df: pd.DataFrame,
                     features: list[str],
                     cluster_col: str) -> pd.DataFrame:
    """One-way ANOVA for each feature across clusters → eta-squared ranking."""
    group_sets = {k: g for k, g in df.groupby(cluster_col)}
    records = []

    for feat in features:
        group_vecs = [g[feat].dropna().values for g in group_sets.values()]
        # need at least 2 groups with >0 observations
        if sum(len(v) > 0 for v in group_vecs) < 2:
            continue
        try:
            f, p = stats.f_oneway(*group_vecs)
        except Exception:
            f, p = np.nan, np.nan

        grand_mean = df[feat].mean()
        ss_between = sum(
            len(g[feat].dropna()) * (g[feat].mean() - grand_mean) ** 2
            for g in group_sets.values()
        )
        ss_total = ((df[feat] - grand_mean) ** 2).sum()
        eta2 = float(ss_between / ss_total) if ss_total > 0 else np.nan

        records.append({"feature": feat, "F": f, "p_value": p, "eta_squared": eta2})

    return (
        pd.DataFrame(records)
        .dropna(subset=["eta_squared"])
        .sort_values("eta_squared", ascending=False)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
def plot_importance_bar(importance: pd.DataFrame, top_n: int, out_path: str) -> None:
    top = importance.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(8, top_n * 0.55 + 1.8))
    colors = ["#d62728" if v > 0.14 else "#1f77b4" for v in top["eta_squared"]]
    ax.barh(top["feature"][::-1], top["eta_squared"][::-1], color=colors[::-1])
    ax.axvline(0.14, color="grey", lw=0.9, ls="--",
               label="η²=0.14 (large effect, Cohen)")
    ax.set_xlabel("η²  (fraction of between-cluster variance)")
    ax.set_title(f"Top {top_n} features by cluster-separating power")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_violin(df: pd.DataFrame,
                features: list[str],
                cluster_col: str,
                top_n: int,
                out_path: str) -> None:
    top_feats = features[:top_n]
    
    # Sort the dataframe so clusters appear in order (0, 1, 2, 3) on the x-axis
    df_sorted = df.sort_values(by=cluster_col)

    ncols = 2
    nrows = (len(top_feats) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5, nrows * 3.2),
                             constrained_layout=True)
    axes = np.array(axes).flatten()

    for i, feat in enumerate(top_feats):
        ax = axes[i]
        # Let Seaborn handle the palette natively based on the sorted categories
        sns.violinplot(data=df_sorted, x=cluster_col, y=feat,
                       palette="tab10", inner="box", hue=cluster_col, legend=False,
                       cut=0, ax=ax)
        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("")

    for j in range(len(top_feats), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribution of top discriminating features by cluster",
                 fontsize=11)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_cluster_heatmap(df: pd.DataFrame,
                         features: list[str],
                         cluster_col: str,
                         out_path: str) -> None:
    means = df.groupby(cluster_col)[features].mean().T   # features × clusters
    row_min = means.min(axis=1)
    row_max = means.max(axis=1)
    normed = means.subtract(row_min, axis=0).divide(
        (row_max - row_min).replace(0, 1), axis=0
    )
    height = max(8, len(features) * 0.35)
    width  = max(6, means.shape[1] * 1.8)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(normed, cmap="RdYlGn", center=0.5,
                linewidths=0.3, linecolor="white",
                annot=True, fmt=".2f", annot_kws={"size": 7},
                ax=ax,
                cbar_kws={"label": "Normalised cluster mean [0=min, 1=max]"})
    ax.set_title("Cluster centroid profile (min-max normalised per feature)")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--person_parquet", required=True,
                        help="Path to baseline_cohort_person_level_phenotypes.parquet")
    parser.add_argument("--cluster_col",   default="trajectory_cluster")
    parser.add_argument("--out_dir",       default="outputs/feature_importance")
    parser.add_argument("--top_n",         type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nLoading {args.person_parquet} ...")
    df = load_person(args.person_parquet, args.cluster_col)

    n_clusters = df[args.cluster_col].nunique()
    print(f"  {n_clusters} clusters: {sorted(df[args.cluster_col].unique())}")

    features = get_numeric_features(df)
    if not features:
        raise RuntimeError("No numeric feature columns found. "
                           "Check that the parquet contains the pipeline's output.")

    # ── ANOVA ─────────────────────────────────────────────────────────────────
    print("\nComputing one-way ANOVA (feature × cluster) ...")
    importance = anova_importance(df, features, args.cluster_col)

    csv_path = os.path.join(args.out_dir, "feature_importance_table.csv")
    importance.to_csv(csv_path, index=False)
    print(f"\nTop {args.top_n} features by η² (eta-squared):")
    print(importance.head(args.top_n).to_string(index=False))
    print(f"\nFull table saved: {csv_path}")

    top_features = importance["feature"].head(args.top_n).tolist()

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_importance_bar(
        importance, args.top_n,
        os.path.join(args.out_dir, "feature_importance_bar.png")
    )
    plot_violin(
        df, top_features, args.cluster_col, args.top_n,
        os.path.join(args.out_dir, "cluster_profiles_violin.png")
    )
    plot_cluster_heatmap(
        df, features, args.cluster_col,
        os.path.join(args.out_dir, "cluster_means_heatmap.png")
    )

    print(f"\nAll outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
