#!/bin/bash --login
#SBATCH -n 2
#SBATCH -t 8:00:00
#SBATCH -o logs/evalw_%x_%j.out
#SBATCH -e logs/evalw_%x_%j.err
#
# Greedy-NoV2V comparator: evaluates the Greedy rule with and without V2V on ONE
# traffic window (the run with V2V is a control that must reproduce the
# published Greedy value). Writes results/greedy_nov2v/eval_greedy_nov2v_<tag>.csv.
#
# Usage: sbatch -J gnv_<tag> refactor/slurm/eval_greedy_nov2v.sh <pos_csv> <tag> [experiment] [n_episodes]
#   e.g. ... data/paired/corridor_08-09_seed42.csv main_08-09 a100_peak 20
#        ... data/fcd/corridor_24h_seed42.csv general_24h general_24h 20
set -euo pipefail
POS_CSV="${1:?}"; TAG="${2:?}"; EXP="${3:-a100_peak}"; N="${4:-20}"
# Environment: on a cluster set CONDA_MODULE (the module that provides conda) and
# CONDA_ENV (the environment from requirements.txt); with neither set, the active
# Python environment is used, so the script also runs with plain `bash`.
if [ -n "${CONDA_MODULE:-}" ] && command -v module >/dev/null 2>&1; then module load "$CONDA_MODULE"; fi
if [ -n "${CONDA_ENV:-}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"; mkdir -p logs
echo "window=$POS_CSV tag=$TAG experiment=$EXP episodes=$N"
python refactor/scripts/eval_greedy_nov2v.py --experiment "$EXP" --pos-csv "$POS_CSV" \
    --n-episodes "$N" --out-csv "results/greedy_nov2v/eval_greedy_nov2v_${TAG}.csv"
