#!/bin/bash
#SBATCH --job-name=beacon-experiment
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=beacon-experiment.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the release folder}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
: "${PYTHON:=python}"
exec "$PYTHON" -B -m experiments.run "$@"
