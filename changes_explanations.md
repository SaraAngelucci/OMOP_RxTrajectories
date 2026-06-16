# Post-submission changes

This document records corrections and code changes made **after** the thesis PDF
was submitted.

All numbers below were reproduced with the synthetic cohort generator
(`generate_synthetic_cohort.py`, `N=1000`, `seed=42`) and the pipeline run under
`config/config_synthetic.yaml`. They supersede any conflicting figures in the
submitted PDF.

---

## Summary of canonical reproduced numbers

| Quantity | Value |
|---|---|
| Persons generated | 1000 |
| Persons clusterable / evaluable (maintenance-eligible) | **478** |
| Persons excluded as insufficient history (cluster `-1`) | 310 |
| Negative-control patients found | **143** |
| Acute exposures leaked into maintenance phenotypes | **0 / 143** |
| Operational partition | **k = 4** |

### Cosine silhouette

| k | silhouette |
|---|---|
| 2 | 0.5215 |
| 3 | 0.6182 |
| 4 | 0.6554 |
| 5 | 0.7115 |
| 6 | 0.7354 |

The silhouette is monotonically increasing over the grid, so the
silhouette-optimal value is at the grid edge (k = 6). Therefore **do not**
select k by the silhouette maximum. k = 4 is fixed *a priori* as the operational
partition for phenotypic interpretability (`clustering.final_k: 4`), and the full
sweep is reported only as a model-selection diagnostic (Figure 4.2).

---

## Synthetic generator now exercises the discontinuation features

**Problem.** In the submitted version, `early_disc_90_rate`, `switch_60_rate`,
and `poly_month_prop` were effectively constant (≈0 / never reaching the
polypharmacy threshold) in the synthetic cohort. They contributed no variance to
the feature vector, so the "maintenance-aware discontinuation" capability was
present in code but unsupported by the data.

**Fix** Three calibrated archetypes were added to `generate_synthetic_cohort.py` so that each of the three
features now varies across genuinely eligible patients:

- `N06A_early_then_restart` (prevalence 0.06): a short initial era followed by a
  long gap and a later restart era → drives `restart_180_rate` and early
  discontinuation behaviour.
- `N06A_switch` (prevalence 0.05): a maintenance era on one ingredient followed
  by a switch to a *different* ingredient within the switch window → drives
  `switch_60_rate`.
- `polypharmacy_severe` (prevalence 0.04): five concurrent ingredients spanning
  the full follow-up → drives `poly_month_prop` to 1.0 and `mean_active_n` to
  the polypharmacy threshold (5).

Supporting vocabulary additions: a fifth ingredient
(`700060 lamotrigine`) was added to `synthetic_data/CONCEPT.csv` and an identity
row to `synthetic_data/concept_ancestor.csv` so a 5-drug concurrent regimen is
representable. `no_psychiatric_rx` prevalence was reduced (0.35 → 0.22) to make
room for the new eligible patients.

---

## Table 4.2 re-derived 

Below is the **re-derived** k = 4 table, computed
directly from the cluster-mean feature values of the current run. These
descriptions should be read in place of the submitted Table 4.2.

| Cluster | n | Defining centroid features | Re-derived archetype |
|---|---|---|---|
| 0 | 50 | `mean_active_n` 1.94, `n_distinct_ingredients` 3, `burden_slope` −0.093 (declining), `poly_month_prop` 0 | **De-intensifying multi-agent use** — sustained low-order combination therapy with a downward burden trajectory (tapering). |
| 1 | 40 | `mean_active_n` 5.0, `poly_month_prop` 1.0, `mean_turnover` 0.042 (very low), `n_distinct_ingredients` 5 | **Sustained polypharmacy** — stable high-burden, ≥5 concurrent agents across the whole window. |
| 2 | 177 | `prop_StableMono` 0.625, `mean_active_n` 0.73, `n_distinct_ingredients` 1.23, `switch_60_rate` 0.116 | **Stable monotherapy** — predominantly single-agent maintenance, with a minority undergoing a single ingredient switch. |
| 3 | 211 | `prop_NoRx` 0.511, `restart_180_rate` 0.562, `mean_turnover` 0.198 (highest), `early_disc_90_rate` 0.130 | **Intermittent / stop-start use** — sparse, gappy exposure with frequent restarts. |

(`-1` = 310 persons excluded as insufficient prescribing history; not a cluster.)

---

## 56-vs-180-day single-era 

- **Tier A — single-era chronic use:** one continuous era ≥ **180 days**
  (`maintenance_single_era_min_days: 180`). 180 days is the 6-month chronic-use
  convention and is the intended single-era rule.
- **Tier B — cumulative use across multiple eras:** ≥ 2 distinct eras *and*
  ≥ **56 days** total (`maintenance_min_total_days: 56`, raised from 28 to reduce
  acute-exposure leakage; `maintenance_min_eras: 2`).

---

**6 of 7 labels populated:**

| Phenotype | n |
|---|---|
| Insufficient prescribing history | 310 |
| Persistent stable use | 162 |
| Intermittent stop-start | 156 |
| Early drop-off / de-intensification | 79 |
| High-turnover switching | 41 |
| Stable polypharmacy | 40 |
| Mixed transition pattern | 0 |

"Mixed transition pattern" is a deliberate catch-all for patients matching none of
the specific rules. It is legitimately empty in this synthetic cohort (every
generated eligible patient matches a specific behaviour) and is retained as a
defined residual category rather than removed, so that real-world data with
ambiguous trajectories has a defined bucket.

---

## k-sweep in the default config

The full silhouette sweep grid is in `config/config_synthetic.yaml`:

```yaml
clustering:
  k_grid: [2, 3, 4, 5, 6]   # full sweep for the model-selection diagnostic
  final_k: 4                # operational partition fixed for interpretability
  seed: 42
```

The pipeline computes the silhouette over the entire `k_grid` (rendering
Figure 4.2) but fixes the reported cluster assignment at `final_k`. If
`final_k` is set to `null`, the pipeline falls back to the silhouette-optimal k.
This cleanly separates the diagnostic sweep from the operational partition.

---

## Scope and validity caveats

Two points are stated explicitly here:

1. **Synthetic data demonstrates capability, not real-world prevalence.** the pipeline correctly
   detects these behaviours when they are present — and does **not** estimate how
   common they are in any real population. The phenotype counts in this document
   are properties of the synthetic generator, not epidemiological findings.

2. **The silhouette does not independently select k = 4.** The cosine silhouette
   increases monotonically across the grid (0.5215 → 0.7354), so the
   silhouette-optimal value sits at the grid edge. k = 4 is fixed *a priori* for
   clinical interpretability (it resolves the engineered archetypes into
   distinguishable groups), and the full sweep is reported only as a diagnostic.
   k = 4 is not claimed the statistically optimal partition.
