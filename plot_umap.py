"""UMAP embedding of the person-level feature vector (thesis Figures 4.4-4.5 etc.).

Reproduces the two-panel UMAP figure used throughout the Results chapter: a
2-D UMAP projection of the standardised person-level feature vector, shown
twice side by side - left coloured by the K-means ``trajectory_cluster``, right
coloured by the participant's dominant ingredient ("ingredient-anchored UMAP
island"). The embedding uses ``metric="cosine"`` on the standardised features,
which matches the cosine-equivalent geometry the pipeline actually clusters in
(StandardScaler -> L2 normalise -> Euclidean K-means).

The feature columns are read from the parquet automatically so the projection
always reflects whatever the pipeline actually clustered on (currently the
24-component vector; the submitted thesis text describes 23, before the
``is_single_era_maintenance`` Tier-A flag was added).

Run:  MPLBACKEND=Agg .venv/bin/python plot_umap.py
      MPLBACKEND=Agg .venv/bin/python plot_umap.py --dataset synthetic_50k
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import umap

# Columns that are identifiers / labels / strings, never part of the vector.
NON_FEATURES = {
    "person_id", "trajectory_cluster", "discontinuation_phenotype",
    "disc_evaluable", "ingredient_concept_id", "drug_name", "plot_drug",
    "dominant_state_w1", "dominant_state_w2", "dominant_state_w3",
    "dominant_state_w4",
}

CLUSTER_PALETTE = {
    -1: "#bbbbbb", 0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c",
    3: "#d62728", 4: "#9467bd", 5: "#8c564b", 6: "#e377c2",
}


def _dominant_ingredient(eras_path: str, concept_csv: str) -> pd.DataFrame:
    eras = pd.read_parquet(eras_path)
    eras["era_duration"] = (
        pd.to_datetime(eras["era_end_date"]) - pd.to_datetime(eras["era_start_date"])
    ).dt.days
    dom = eras.loc[eras.groupby("person_id")["era_duration"].idxmax(),
                   ["person_id", "ingredient_concept_id"]].copy()
    try:
        concepts = pd.read_csv(concept_csv, usecols=["concept_id", "concept_name"])
        name = dict(zip(concepts["concept_id"], concepts["concept_name"]))
        dom["drug_name"] = dom["ingredient_concept_id"].map(name).fillna(
            dom["ingredient_concept_id"].astype(str))
    except Exception:
        dom["drug_name"] = dom["ingredient_concept_id"].astype(str)
    return dom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic",
                    help="subfolder under outputs/ (e.g. synthetic, synthetic_50k)")
    ap.add_argument("--run_label", default="baseline_cohort")
    ap.add_argument("--concept_csv", default="synthetic_data/CONCEPT.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    folder = os.path.join("outputs", args.dataset)
    pheno = os.path.join(folder, f"{args.run_label}_person_level_phenotypes.parquet")
    eras = os.path.join(folder, f"{args.run_label}_eras.parquet")
    out = args.out or os.path.join(folder, "figures", "presentation",
                                   f"umap_{args.dataset}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    df = pd.read_parquet(pheno)
    df = df[df["disc_evaluable"] == True].copy()  # noqa: E712 (pandas mask)
    if len(df) > 15000:
        df = df.sample(n=15000, random_state=42)

    feature_cols = [c for c in df.columns
                    if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(df[c])]
    print(f"Loaded {len(df)} evaluable persons; "
          f"{len(feature_cols)}-component feature vector")

    X = df[feature_cols].fillna(0).to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X)

    print("Running UMAP (n_neighbors=15, min_dist=0.1, metric=cosine, seed=42)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                        random_state=42)
    emb = reducer.fit_transform(X_scaled)
    df["UMAP_1"], df["UMAP_2"] = emb[:, 0], emb[:, 1]

    dom = _dominant_ingredient(eras, args.concept_csv)
    df = df.merge(dom, on="person_id", how="left")
    df["drug_name"] = df["drug_name"].fillna("unknown")
    top = df["drug_name"].value_counts().nlargest(8).index
    df["plot_drug"] = df["drug_name"].where(df["drug_name"].isin(top), "Other")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"UMAP of the person-level feature vector "
                 f"({args.dataset}, n_evaluable={len(df)}, "
                 f"{len(feature_cols)} features, cosine metric)",
                 fontsize=14, fontweight="bold")

    for k in sorted(df["trajectory_cluster"].unique()):
        m = df["trajectory_cluster"] == k
        ax1.scatter(df.loc[m, "UMAP_1"], df.loc[m, "UMAP_2"], s=14, alpha=0.75,
                    color=CLUSTER_PALETTE.get(int(k), "#333333"),
                    label=f"cluster {int(k)} (n={int(m.sum())})")
    ax1.set_title("K-means cluster assignment")
    ax1.set_xlabel("UMAP-1"); ax1.set_ylabel("UMAP-2")
    ax1.legend(fontsize=8, framealpha=0.9, loc="best")

    cmap = plt.cm.tab10.colors
    for i, g in enumerate(sorted(df["plot_drug"].unique())):
        m = df["plot_drug"] == g
        ax2.scatter(df.loc[m, "UMAP_1"], df.loc[m, "UMAP_2"], s=14, alpha=0.75,
                    color=cmap[i % len(cmap)], label=g)
    ax2.set_title("Dominant ingredient (UMAP islands)")
    ax2.set_xlabel("UMAP-1"); ax2.set_ylabel("UMAP-2")
    ax2.legend(fontsize=8, framealpha=0.9, loc="best", title="ingredient")

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
