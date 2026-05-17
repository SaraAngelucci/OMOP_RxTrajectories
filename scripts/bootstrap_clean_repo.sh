#!/usr/bin/env bash
# bootstrap_clean_repo.sh
# ------------------------
# Build a clean, public-ready copy of the thesis pipeline in a new
# directory and initialise a fresh git repository on it.
#
# The new repo contains ONLY the files that belong in a public release:
#   * pipeline source code (src/, main.py, helper scripts)
#   * configuration YAMLs (config/)
#   * tests and the unit-test vocabulary stub
#   * README.md, LICENSE, CITATION.cff, requirements.txt, requirements.lock
# The new repo deliberately omits:
#   * raw OMOP CSV/parquet bundles (eunomia_raw/, mimic_omop_raw/, etc.)
#   * generated outputs (outputs/, data/synthetic_medstat*/)
#   * the local LaTeX thesis source (NEWAngelucci_draft_thesis/)
#   * Python virtualenvs, IDE state, OS junk
#
# Usage:
#   bash scripts/bootstrap_clean_repo.sh                       # uses default name + parent
#   bash scripts/bootstrap_clean_repo.sh <target_dir>          # custom target path
#   bash scripts/bootstrap_clean_repo.sh <target_dir> <name>   # custom name (sets git remote-friendly project name)
#
# Default target: ${HOME}/Desktop/omop-rx-trajectories

set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${1:-${HOME}/Desktop/omop-rx-trajectories}"
PROJECT_NAME="${2:-omop-rx-trajectories}"

if [ -e "${TARGET_DIR}" ]; then
    echo "ERROR: target ${TARGET_DIR} already exists. Move or remove it first." >&2
    exit 1
fi

echo "=============================================================="
echo " Bootstrapping a clean public repository for the thesis pipeline"
echo " Source: ${SOURCE_DIR}"
echo " Target: ${TARGET_DIR}"
echo " Name:   ${PROJECT_NAME}"
echo "=============================================================="

mkdir -p "${TARGET_DIR}"

# ----- copy the curated file set -----
# Top-level files
for f in \
    README.md \
    LICENSE \
    CITATION.cff \
    requirements.txt \
    requirements.lock \
    .gitignore \
    main.py \
    make_plots.py \
    compute_ari_matrix.py \
    generate_synthetic_cohort.py \
    generate_synthetic_vocab.py \
    prepare_external_omop.py
do
    if [ -f "${SOURCE_DIR}/${f}" ]; then
        cp "${SOURCE_DIR}/${f}" "${TARGET_DIR}/${f}"
    else
        echo "WARN: ${f} not found in source; skipping." >&2
    fi
done

# Directory copies (curated)
rsync -a --delete \
    --exclude '__pycache__' --exclude '*.pyc' \
    "${SOURCE_DIR}/src/" "${TARGET_DIR}/src/"

rsync -a --delete \
    --exclude '__pycache__' --exclude '*.pyc' \
    "${SOURCE_DIR}/tests/" "${TARGET_DIR}/tests/"

rsync -a --delete "${SOURCE_DIR}/config/" "${TARGET_DIR}/config/"

# Vocabulary stub: only the small CONCEPT.csv that the tests and external
# configs need; do NOT copy the large CSVs.
mkdir -p "${TARGET_DIR}/synthetic_data"
for vocab in CONCEPT.csv concept_ancestor.csv; do
    if [ -f "${SOURCE_DIR}/synthetic_data/${vocab}" ]; then
        cp "${SOURCE_DIR}/synthetic_data/${vocab}" "${TARGET_DIR}/synthetic_data/${vocab}"
    fi
done

# Notebooks (if they are not data-heavy)
if [ -d "${SOURCE_DIR}/notebooks" ]; then
    rsync -a \
        --exclude 'data' \
        --exclude '.ipynb_checkpoints' \
        "${SOURCE_DIR}/notebooks/" "${TARGET_DIR}/notebooks/"
fi

# Helper scripts directory
mkdir -p "${TARGET_DIR}/scripts"
if [ -f "${SOURCE_DIR}/scripts/run_pipeline.py" ]; then
    cp "${SOURCE_DIR}/scripts/run_pipeline.py" "${TARGET_DIR}/scripts/run_pipeline.py"
fi
cp "${BASH_SOURCE[0]}" "${TARGET_DIR}/scripts/bootstrap_clean_repo.sh"

# ----- placeholders for run-time directories -----
mkdir -p "${TARGET_DIR}/outputs"
mkdir -p "${TARGET_DIR}/data"
touch "${TARGET_DIR}/outputs/.gitkeep"
touch "${TARGET_DIR}/data/.gitkeep"

# ----- initialise a fresh git repo -----
cd "${TARGET_DIR}"
git init -q -b main
git add -A
git -c user.email="sara.angelucci@example.invalid" \
    -c user.name="Sara Angelucci" \
    commit -q -m "Initial public release of ${PROJECT_NAME}

Reproducible PySpark trajectory-phenotyping pipeline for OMOP CDM v5.3 / v5.4
data. See README.md for usage. Author-only repository (no AI co-authorship)."

echo
echo "Bootstrap complete."
echo "  Directory: ${TARGET_DIR}"
echo
echo "Next steps:"
echo "  cd ${TARGET_DIR}"
echo "  gh repo create ${PROJECT_NAME} --public --source=. --remote=origin --push"
echo "  # (or, if you do not use the gh CLI:)"
echo "  # create an empty repo on github.com/<you>/${PROJECT_NAME}"
echo "  # git remote add origin git@github.com:<you>/${PROJECT_NAME}.git"
echo "  # git push -u origin main"
