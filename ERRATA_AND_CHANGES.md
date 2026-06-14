# Errata and post-submission changes

This document records corrections and code changes made **after** the thesis PDF
was submitted. The submitted PDF is unchanged; this file is the authoritative
record of what differs between the manuscript text and the current codebase, and
why each change is scientifically justified.

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

### Cosine silhouette sweep (evaluator distance = cosine, matching the fit)

| k | silhouette |
|---|---|
| 2 | 0.5215 |
| 3 | 0.6182 |
| 4 | 0.6554 |
| 5 | 0.7115 |
| 6 | 0.7354 |

The silhouette is monotonically increasing over the grid, so the
silhouette-optimal value is at the grid edge (k = 6). We therefore **do not**
select k by the silhouette maximum. k = 4 is fixed *a priori* as the operational
partition for phenotypic interpretability (`clustering.final_k: 4`), and the full
sweep is reported only as a model-selection diagnostic (Figure 4.2). This is
stated honestly rather than presenting k = 4 as a silhouette optimum.

---

## Change 1 — Synthetic generator now exercises the discontinuation features

**Problem.** In the submitted version, `early_disc_90_rate`, `switch_60_rate`,
and `poly_month_prop` were effectively constant (≈0 / never reaching the
polypharmacy threshold) in the synthetic cohort. They contributed no variance to
the feature vector, so the "maintenance-aware discontinuation" capability was
present in code but unsupported by the data.

**Fix (chose to *exercise*, not drop, the features).** Three calibrated
archetypes were added to `generate_synthetic_cohort.py` so that each of the three
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
room for the new eligible patients, and a mislabelled additional drug in
`polypharmacy_escalating` was corrected.

**Result (traceable to archetype sizes).** All three features now fire with
non-trivial support: early drop-off / de-intensification n = 79, high-turnover
switching n = 41, stable polypharmacy n = 40. The features carry real variance
into the clustering.

---

## Change 2 — Table 4.2 re-derived from actual cluster centroids

The submitted Table 4.2 archetype descriptions did not match the centroids the
pipeline actually produces. Below is the **re-derived** k = 4 table, computed
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

## Change 3 — Manuscript reconciled to a single pipeline version (n = 478 / k = 4)

The submitted text mixed numbers from more than one pipeline configuration.
The canonical regime is the one in the table above: **n_evaluable = 478, k = 4**,
with the cosine silhouette sweep as reported. §4.1 (cohort/eligibility counts)
and §4.2 (clustering results) should both be read against these numbers. Where
the PDF gives a different evaluable-n or a different silhouette value for k = 4,
the values in this errata are authoritative.

---

## Change 4 — 56-vs-180-day single-era discrepancy resolved

The submitted text described the single continuous-era maintenance threshold
using 56 days, which conflated two distinct rules. They are now kept separate and
documented in `config/config_synthetic.yaml`:

- **Tier A — single-era chronic use:** one continuous era ≥ **180 days**
  (`maintenance_single_era_min_days: 180`). 180 days is the 6-month chronic-use
  convention and is the intended single-era rule.
- **Tier B — cumulative use across multiple eras:** ≥ 2 distinct eras *and*
  ≥ **56 days** total (`maintenance_min_total_days: 56`, raised from 28 to reduce
  acute-exposure leakage; `maintenance_min_eras: 2`).

The 56-day figure belongs to Tier B (the cumulative rule), **not** to the
single-era rule. The submitted text's use of 56 for a single era was an error;
the intended single-era threshold is 180 days. Code and config now agree.

---

## Change 5 — Silhouette evaluator distance matched to the fitting distance

Clustering is performed in an L2-normalised space (StandardScaler →
Normalizer p = 2 → Euclidean K-means), which is cosine-equivalent. The submitted
evaluation used Spark's default `squaredEuclidean` silhouette, which does **not**
match the geometry the model was fit in. The evaluator is now:

```python
ClusteringEvaluator(featuresCol="features_cosine",
                    predictionCol="trajectory_cluster",
                    metricName="silhouette",
                    distanceMeasure="cosine")
```

The reported silhouette values (table above) are the cosine values, reported
honestly including the fact that the curve increases to the grid edge.

---

## Change 6 — Phenotype thresholds externalised to config; reachability justified

The rule-based discontinuation phenotype thresholds (Methods §3.10) were hard-coded.
They are now fully specified in a `phenotype:` block in
`config/config_synthetic.yaml`:

```yaml
phenotype:
  stable_turnover_max:   0.20
  early_disc_low:        0.25
  early_disc_high:       0.50
  restart_high:          0.50
  switch_high:           0.50
  burden_slope_neg:     -0.10
```

The decision rule in `pipeline.py` was also **reordered** so that specific
behaviours are tested before the generic "stable" fallbacks (high-turnover
switching → intermittent stop-start → early drop-off → stable polypharmacy →
persistent stable use → mixed). In the submitted ordering, switchers were
absorbed by "persistent stable use" before the switching rule could fire, leaving
two categories unreachable.

**Reachability after the fix (6 of 7 labels populated):**

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

## Change 7 — k-sweep restored in the default config

The full silhouette sweep grid is restored in `config/config_synthetic.yaml`:

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

## Scope and validity caveats (read before presenting)

Two points are stated explicitly here so they cannot be mistaken for stronger
claims than the evidence supports:

1. **Synthetic data demonstrates capability, not real-world prevalence.** The
   three archetypes in Change 1 were deliberately constructed so that genuinely
   maintenance-eligible patients exhibit early discontinuation, ingredient
   switching, and severe polypharmacy. The fact that the corresponding features
   now fire therefore establishes *construct validity* — the pipeline correctly
   detects these behaviours when they are present — and does **not** estimate how
   common they are in any real population. The phenotype counts in this document
   are properties of the synthetic generator, not epidemiological findings.

2. **The silhouette does not independently select k = 4.** The cosine silhouette
   increases monotonically across the grid (0.5215 → 0.7354), so the
   silhouette-optimal value sits at the grid edge. k = 4 is fixed *a priori* for
   clinical interpretability (it resolves the engineered archetypes into
   distinguishable groups), and the full sweep is reported only as a diagnostic.
   We do not claim k = 4 is the statistically optimal partition.

## Files changed

- `generate_synthetic_cohort.py` — three new archetypes; prevalence rebalancing; per-era duration and switch logic.
- `synthetic_data/CONCEPT.csv`, `synthetic_data/concept_ancestor.csv` — fifth ingredient for severe polypharmacy.
- `src/thesis_rx/pipeline.py` — cosine silhouette evaluator; externalised phenotype thresholds; reordered decision rule; `final_k` override.
- `config/config_synthetic.yaml` — `phenotype:` block; Tier A/B documentation; restored `k_grid` and `final_k`.
- `silhouette_grid_standalone.py` — added `distanceMeasure="cosine"` so the standalone diagnostic matches the main-pipeline evaluator.
- `make_plots.py` — updated the silhouette fallback table (used only when the CSV is absent) to the cosine values and corrected the accompanying comment.
- `README.md` — corrected test-check counts (ten primary / four edge), `maintenance_min_total_days` (56) and `maintenance_single_era_min_days` (180) documentation, and added the `final_k` row.
- `tests/test_validation_cohort.py` — added the Tier A single-era and multi-course-acute edge checks; corrected the printed assertion count.
- `main.py`, `tests/test_validation_cohort.py` — pin `PYSPARK_PYTHON`/`PYSPARK_DRIVER_PYTHON` to `sys.executable` so Spark workers use the driver interpreter. Without this, Spark falls back to PATH `python3` (often 3.9), which crashes on the PEP 604 `X | Y` type-union syntax the pipeline uses.
- `README.md` — corrected the minimum Python version from 3.9 to **3.10** (required by the `X | Y` syntax) and documented the `PYSPARK_PYTHON` requirement.

### Known residual (intentionally not changed)

The external-dataset configs (`config/config_external_eunomia.yaml`,
`config_external_mimic.yaml`, `config_danish.yaml`) and `prepare_external_omop.py`
retain `maintenance_min_total_days: 28`. These drive only the interoperability
vignettes, not the headline synthetic result. They were left at 28 to remain
consistent with any vignette numbers cited in the submitted PDF; align them to 56
only if the vignettes are re-run and re-reported.
