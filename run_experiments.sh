#!/usr/bin/env bash
# Runs (or resumes) every experiment; finished predictions are cached in results/preds/.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 src/data.py && python3 src/ons_api.py && python3 src/tune.py && python3 src/experiments.py "$@" && python3 src/analysis.py
