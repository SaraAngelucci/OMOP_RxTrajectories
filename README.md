# OMOP Prescription Trajectory Phenotyping Pipeline

**Author:** Sara Angelucci, University of Copenhagen, Faculty of Science

**Thesis:** *Development and validation of a scalable OMOP-compatible framework for longitudinal prescription trajectory phenotyping*

---

## Overview

This repository contains the complete computational architecture and reproducible PySpark pipeline developed for my Master's thesis. The pipeline is **drug-agnostic** and operates entirely on tables defined by the Observational Medical Outcomes Partnership (OMOP) Common Data Model. It transforms raw, fragmented drug exposure records into:

1. continuous ingredient-level **drug eras**;
2. a strict **monthly person-time grid** with active-ingredient burden, starts, stops, and Jaccard turnover;
3. mutually exclusive **monthly prescribing states** (e.g. `StableMono`, `StablePolypharmacy`, `HighTurnover`);
4. four temporally ordered **sub-window features** that preserve trajectory shape under K-means clustering;
5. **maintenance-aware discontinuation events** (early drop-off, restart within 180 days, switch within 60 days);
6. rule-based **discontinuation phenotypes** and an unsupervised **K-means** trajectory clustering, evaluated with a **stratified silhouette** estimator.

### Project pivot

This project was originally designed to integrate UK Biobank OMOP prescription data with pharmacogenomic variation. Late in the project timeline, access to the secure UK Biobank Research Analysis Platform was unexpectedly revoked. The thesis was therefore strategically refocused on the **methodological development and rigorous validation of the underlying phenotyping framework**, executed on a Medstat-calibrated synthetic cohort. Because the pipeline is implemented strictly against OMOP-standard inputs, the *same* code is portable to UK Biobank, NIH *All of Us*, and OMOP-mapped Danish National Registers without modification (see `src/thesis_rx/danish_register_adapter.py` for the corresponding non-OMOP ATC adapter).

---

## Repository layout

```text
.
├── config/                              YAML configurations (parameter sets)
│   ├── config_synthetic.yaml            Primary $N{=}1{,}000$ cohort
│   ├── config_synthetic_50k.yaml        Scale-up cohort
│   ├── config_external_eunomia.yaml     OHDSI GiBleed (Tier 1)
│   ├── config_external_synthea.yaml     Synthea27Nj vignette (Tier 2a)
│   ├── config_external_mimic.yaml       MIMIC-IV OMOP demo (Tier 2b)
│   └── config_danish.yaml               Danish LMDB adapter configuration
├── data/                                Generated omop parquet / outputs (mostly git-ignored)
├── generate_synthetic_cohort.py         Medstat-calibrated synthetic OMOP generator
├── prepare_external_omop.py             Thin OMOP CDM loader / eraisation fallback
├── main.py                              Three-experiment orchestrator
├── make_plots.py                        Silhouette + ARI heatmap figures for LaTeX
├── compute_ari_matrix.py                Standalone ARI recovery from saved grid parquets
├── notebooks/
│   ├── medstat_analysis.ipynb           Population-level Medstat trend figures
│   └── synthetic_validation_figures.ipynb
├── requirements.txt                     Dependency floor pins
├── requirements.lock                    Exact pins used for thesis reproduction
├── LICENSE                              MIT
├── scripts/bootstrap_clean_repo.sh      Optional: scaffold a minimal public fork
├── src/thesis_rx/                       Core pipeline package (pipeline.py, io.py, …)
├── synthetic_data/CONCEPT.csv           Minimal OMOP vocabulary stub (names)
├── synthetic_data/concept_ancestor.csv  Identity stub (required by ``io.load_tables``)
└── tests/test_validation_cohort.py      Deterministic logic + edge-case suite
```

---

## 1. Install

Python 3.9 or newer and a JVM (for PySpark) are required. Two dependency files are provided:

- `requirements.txt` — minimum-floor pins (`pyspark>=3.4.0`, etc.) suitable for new environments.
- `requirements.lock` — the exact `pip freeze` from the environment that produced the thesis results, for bit-for-bit reproducibility.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# For day-to-day use:
pip install -r requirements.txt

# For exact reproducibility of the thesis numbers:
pip install -r requirements.lock
```

`scikit-learn` is included in both files (required for the Adjusted Rand Index in Experiment 3).

---

## 2. Generate the synthetic Medstat-calibrated cohort

The thesis results are produced on a 1,000-patient synthetic OMOP cohort whose archetype prevalences are mathematically calibrated to Danish primary-care prescribing frequencies derived from the Medstat register (Chapter 2 of the thesis).

```bash
python generate_synthetic_cohort.py --n_patients 1000 --seed 42 \
    --output_dir data/synthetic_medstat
```

This produces three Parquet files under `data/synthetic_medstat/`:

- `synthetic_drug_era.parquet`
- `synthetic_observation_period.parquet`
- `synthetic_person.parquet`

These match the schemas in `src/thesis_rx/io.py` and `config/config_synthetic.yaml` without further transformation.

---

## 3. Run the full thesis pipeline

```bash
export PYTHONPATH=$(pwd)
python main.py
```

Runs **Experiment 1–3** with `config/config_synthetic.yaml`. Output prefix is taken from YAML `project.output_dir` (typically `outputs/synthetic/`).

| Step | Behaviour |
|------|-----------|
| 1 Baseline | **Full stratified silhouette** over `k ∈ {2,…,6}`; selects best `k`; writes `baseline_cohort_*.parquet` and `baseline_cohort_silhouette_grid.csv`. |
| 2 Negative controls | Validates `acute_antibiotic` leak rate (synthetic cohort only). |
| 3 Sensitivity grid | ARI matrix on discontinuation phenotype labels across a 3×3 (poly × maintenance-day) sweep. Uses fixed `k` (`GRID_FIXED_K`, default **`2`**) unless `GRID_SILHOUETTE_IN_GRID=1`. |

Spark resources and config path are overridden without editing code:

```bash
export PYTHONPATH=$(pwd)
SPARK_CONFIG_PATH=config/config_external_synthea.yaml \
SPARK_DRIVER_MEMORY=8g SPARK_SHUFFLE_PARTITIONS=50 \
GRID_FIXED_K=2 ./.venv/bin/python -u main.py
```

Silhouette is **never skipped** on the baseline (Experiment 1). To re-evaluate silhouette **inside each grid cell** (≈9× clustering cost):

```bash
GRID_SILHOUETTE_IN_GRID=1 ./.venv/bin/python main.py
```

Expect minutes on \(N≈10^3\); tens of GPU-free hours may apply at \(N≈50{,}000\). Uses `spark.catalog.clearCache()` between grid cells.

If Experiment 3 fails with `SparkOutOfMemoryError`, increase `SPARK_DRIVER_MEMORY`.

---

## 4. Deterministic logic tests

A separate unit-testing suite executes the pipeline on a hand-crafted 5-patient cohort that encodes every clinically meaningful edge case (stable monotherapy, escalating polypharmacy, intermittent stop-start, acute exposure, and right-censoring):

```bash
export PYTHONPATH=$(pwd)
python tests/test_validation_cohort.py
```

Expected output: nine primary-cohort logic checks (stable mono, polypharmacy, restart, acute exclusion, right-censoring, cluster column typing, acute clustering exclusion, stable feature sanity, poly ordering), three edge-case checks (single-person sentinel cluster $-1$, overlapping eras, boundary-spanning eras), and the message `VALIDATION COMPLETE. Pipeline is theoretically sound.` Optionally run under pytest: `pytest tests/test_validation_cohort.py`.

The same script writes Parquet artefacts to `data/validation_outputs/`, which are then consumed by `notebooks/synthetic_validation_figures.ipynb` to produce the Gantt chart shown in the thesis (`figures/synthetic_validation_gantt.png`).

---

## 5. Deploying against real OMOP databases

The pipeline is intentionally decoupled from any specific source dataset; it reads only OMOP-standard tables.

### UK Biobank Research Analysis Platform

If access to the UK Biobank Research Analysis Platform is available, the same pipeline can be run by pointing `config/config.yaml` to the platform's OMOP tables:

```text
OMOP/raw_data/omop_drug_era.csv.gz
OMOP/raw_data/omop_drug_exposure.csv.gz
OMOP/raw_data/omop_observation_period.csv.gz
OMOP/raw_data/omop_death.csv.gz                  (optional)
ICB/OMOP/maps/athena_omop/CONCEPT.csv
ICB/OMOP/maps/athena_omop/concept_ancestor.csv
```

No code changes are required. The pipeline already handles UK Biobank-specific quirks such as `eid` → `person_id` harmonisation, duplicate death records, and British date formatting (see `src/thesis_rx/io.py`).

### Danish National Prescription Register (Receptregisteret / LMDB)

For Danish register data, which is ATC-coded rather than OMOP-mapped, the helper `src/thesis_rx/danish_register_adapter.py` converts LMDB extracts into the same intermediate era schema consumed by `run_trajectory_pipeline`. A worked example is shown in the module docstring.

### NIH *All of Us* Research Program

The OMOP CDM is the canonical schema in *All of Us*. The pipeline can be executed there by changing input paths in `config/config.yaml`; no code changes are required.

---

## 6. Configuration

All analytic parameters are externalised in YAML (`config/config_synthetic.yaml`). Required keys are validated by `src.thesis_rx.config.validate_config`. The most important parameters are:

| Key | Default | Meaning |
|-----|---------|---------|
| `analysis.washout_days` | 365 | Pre-index observable time required for eligibility |
| `analysis.followup_months` | 24 | Length of the participant-specific monthly grid |
| `analysis.exposure_gap_days` | 30 | Era-merge gap for the `drug_exposure`-derived sensitivity analysis |
| `analysis.maintenance_min_total_days` | 28 | Minimum cumulative era-days for maintenance eligibility |
| `analysis.early_discontinuation_days` | 90 | Maximum duration of a maintenance-eligible first era classified as early drop-off |
| `analysis.restart_window_days` | 180 | Window for restart events |
| `analysis.switch_window_days` | 60 | Window for switch events (timestamp range join) |
| `analysis.polypharmacy_threshold` | 5 | Number of concurrent ingredients defining polypharmacy |
| `analysis.turnover_low` / `turnover_high` | 0.20 / 0.50 | Jaccard cut-points for state classification |
| `clustering.k_grid` | `[2, 3, 4, 5, 6]` | Candidate `k` for K-means |
| `clustering.seed` | 42 | Random seed |
| `analysis.focus_ingredient_concept_ids` | *(omit)* | Optional list of OMOP `concept_id`s to subset `drug_era` rows (interop vignettes); omit for fully drug-agnostic runs |
| `analysis.maintenance_min_eras` | `2` | Minimum distinct eras for maintenance eligibility; may be set to `1` for sparse simulator exports |

---

## 7. Reproducibility checklist

- All parameters externalised in `config/*.yaml`.
- All random seeds fixed (`numpy`, Python `random`, Spark KMeans seed).
- All intermediate outputs persisted as Parquet under `cfg['project']['output_dir']`.
- Deterministic unit-test cohort (`tests/test_validation_cohort.py`) reproducible bit-for-bit.
- The Medstat-calibrated synthetic generator (`generate_synthetic_cohort.py`) is fully deterministic given a seed.

---

## Citation

If you use this code in academic work, please cite the thesis and this repository (`CITATION.cff`).

## Licence

The code is released for academic review and re-use. Refer to `LICENSE` for full terms.
