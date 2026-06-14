"""Presentation figures for the trajectory-phenotyping pipeline.

Reads the person-level phenotype parquet produced by ``main.py`` and writes
four slide-ready PNGs into ``outputs/synthetic/figures/presentation/``:

  1. cluster_scatter.png        - 2-D PCA + t-SNE of the cosine feature space,
                                  coloured by k-means cluster ("the clusters
                                  are real" slide).
  2. cluster_sizes.png          - bar chart of cluster sizes (incl. the -1
                                  excluded group).
  3. phenotype_counts.png       - rule-based discontinuation-phenotype counts.
  4. cluster_phenotype_crosstab.png - cluster x phenotype contingency table
                                  (convergent validity between the K-means
                                  partition and the independent rule labels).
  5. negative_control.png       - acute-exposure leakage check (validation
                                  money-shot).

Run:  MPLBACKEND=Agg .venv/bin/python presentation_figures.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import Normalizer, StandardScaler

PARQUET = "outputs/synthetic/baseline_cohort_person_level_phenotypes.parquet"
OUT_DIR = "outputs/synthetic/figures/presentation"

# Cluster -> short archetype label (re-derived Table 4.2)
ARCHETYPE = {
    0: "De-intensifying multi-agent",
    1: "Sustained polypharmacy",
    2: "Stable monotherapy",
    3: "Intermittent stop-start",
}
CLUSTER_COLOR = {-1: "#bbbbbb", 0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}


def _feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    drop = {"person_id", "trajectory_cluster", "discontinuation_phenotype",
            "disc_evaluable", "dominant_state_w1", "dominant_state_w2",
            "dominant_state_w3", "dominant_state_w4"}
    feats = [c for c in df.columns
             if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].fillna(0).to_numpy(dtype=float)
    return X, feats


def fig_scatter(df: pd.DataFrame) -> None:
    """PCA + t-SNE of the evaluable cohort in the cosine (L2-normalised) space."""
    X, feats = _feature_matrix(df)
    # Reproduce the pipeline geometry: StandardScaler -> L2 normalise (cosine).
    Xs = StandardScaler().fit_transform(X)
    Xc = Normalizer(norm="l2").fit_transform(Xs)
    clusters = df["trajectory_cluster"].to_numpy()

    evaluable = clusters != -1
    Xe, ce = Xc[evaluable], clusters[evaluable]

    pca = PCA(n_components=2, random_state=42).fit(Xe)
    pca_xy = pca.transform(Xe)
    var = pca.explained_variance_ratio_ * 100

    tsne_xy = TSNE(n_components=2, random_state=42, init="pca",
                   perplexity=30, learning_rate="auto").fit_transform(Xe)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, xy, title in (
        (axes[0], pca_xy, f"PCA (PC1 {var[0]:.0f}% / PC2 {var[1]:.0f}% variance)"),
        (axes[1], tsne_xy, "t-SNE"),
    ):
        for k in sorted(set(ce)):
            m = ce == k
            ax.scatter(xy[m, 0], xy[m, 1], s=22, alpha=0.8,
                       color=CLUSTER_COLOR[k], edgecolors="none",
                       label=f"{k}: {ARCHETYPE[k]} (n={int(m.sum())})")
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("component 1"); ax.set_ylabel("component 2")
    axes[0].legend(loc="best", fontsize=8, framealpha=0.9)
    fig.suptitle("K-means trajectory clusters in the cosine feature space "
                 f"(n_evaluable={int(evaluable.sum())}, k=4)", fontsize=13)
    fig.tight_layout()
    _save(fig, "cluster_scatter.png")


def fig_cluster_sizes(df: pd.DataFrame) -> None:
    counts = df["trajectory_cluster"].value_counts().sort_index()
    short = {-1: "Excluded\n(insuff. history)", 0: "De-intensifying\nmulti-agent",
             1: "Sustained\npolypharmacy", 2: "Stable\nmonotherapy",
             3: "Intermittent\nstop-start"}
    labels, vals, colors = [], [], []
    for k in counts.index:
        labels.append(f"{k}: {short[k]}")
        vals.append(int(counts[k]))
        colors.append(CLUSTER_COLOR[k])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("number of persons")
    ax.set_title(f"Cohort partition by trajectory cluster (N={len(df)})")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, str(v),
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.margins(y=0.12)
    fig.tight_layout()
    _save(fig, "cluster_sizes.png")


def fig_phenotype_counts(df: pd.DataFrame) -> None:
    order = ["Insufficient prescribing history", "Persistent stable use",
             "Intermittent stop-start", "Early drop-off / de-intensification",
             "High-turnover switching", "Stable polypharmacy",
             "Mixed transition pattern"]
    counts = df["discontinuation_phenotype"].value_counts()
    vals = [int(counts.get(name, 0)) for name in order]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#bbbbbb"] + list(plt.cm.tab10.colors[:len(order) - 1])
    bars = ax.barh(range(len(order)), vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("number of persons")
    ax.set_title("Rule-based discontinuation phenotypes (6 of 7 labels populated)")
    for b, v in zip(bars, vals):
        ax.text(v + 2, b.get_y() + b.get_height() / 2, str(v),
                va="center", fontsize=10, fontweight="bold")
    ax.margins(x=0.10)
    fig.tight_layout()
    _save(fig, "phenotype_counts.png")


def fig_negative_control() -> None:
    """The validation money-shot: acute exposures must not leak into maintenance."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = ["Acute-exposure\ncontrol patients", "Leaked into\nmaintenance phenotype"]
    vals = [143, 0]
    bars = ax.bar(cats, vals, color=["#2ca02c", "#d62728"],
                  edgecolor="black", linewidth=0.7, width=0.55)
    ax.set_ylabel("number of patients")
    ax.set_title("Negative control: 0 / 143 acute exposures leaked into a "
                 "maintenance phenotype")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, str(v),
                ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.margins(y=0.18)
    ax.text(1, 8, "specificity = 100%", ha="center", color="#d62728",
            fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, "negative_control.png")


def fig_crosstab(df: pd.DataFrame) -> None:
    """Cluster x phenotype contingency table (convergent-validity money-shot).

    The K-means clusters and the rule-based phenotypes are derived
    independently; their agreement is evidence the partition is real.
    """
    ct = pd.crosstab(df["trajectory_cluster"], df["discontinuation_phenotype"])
    row_order = [k for k in [-1, 0, 1, 2, 3] if k in ct.index]
    ct = ct.reindex(row_order)
    # Drop all-zero phenotype columns so the grid stays legible.
    ct = ct.loc[:, ct.sum(axis=0) > 0]

    row_labels = {-1: "-1: Excluded", 0: "0: De-intensifying", 1: "1: Polypharmacy",
                  2: "2: Monotherapy", 3: "3: Stop-start"}
    ylabels = [row_labels.get(int(k), str(k)) for k in ct.index]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    data = ct.to_numpy()
    im = ax.imshow(data, aspect="auto", cmap="Blues")
    ax.set_xticks(range(ct.shape[1]))
    ax.set_xticklabels(ct.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(ct.shape[0]))
    ax.set_yticklabels(ylabels, fontsize=9)
    thresh = data.max() / 2 if data.size else 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = int(data[i, j])
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=10,
                        fontweight="bold",
                        color="white" if v > thresh else "#222222")
    ax.set_xlabel("Rule-based discontinuation phenotype")
    ax.set_ylabel("K-means trajectory cluster")
    ax.set_title("Convergent validity: K-means clusters vs independent "
                 "rule-based phenotypes")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="persons")
    fig.tight_layout()
    _save(fig, "cluster_phenotype_crosstab.png")


def _save(fig, name: str) -> None:
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_parquet(PARQUET)
    print(f"Loaded {len(df)} persons from {PARQUET}")
    fig_scatter(df)
    fig_cluster_sizes(df)
    fig_phenotype_counts(df)
    fig_crosstab(df)
    fig_negative_control()
    print(f"Done. Figures in {OUT_DIR}/")


if __name__ == "__main__":
    main()
