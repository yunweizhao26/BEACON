#!/bin/bash
#SBATCH --job-name=beacon-equivalence
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=beacon-test.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the release folder}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
: "${PYTHON:=python}"
"$PYTHON" -B -m tests.test_equivalence --unit
"$PYTHON" -B -m tests.test_evaluation --unit
"$PYTHON" -B -m tests.test_bundle
"$PYTHON" -B -m tests.test_equivalence
"$PYTHON" -B -m tests.test_evaluation
