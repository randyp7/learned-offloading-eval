#!/bin/bash --login
#SBATCH -n 2
#SBATCH -t 12:00:00
#SBATCH -o logs/evalw_%x_%j.out
#SBATCH -e logs/evalw_%x_%j.err
#
# Peer-availability audit: replays the shared evaluation for the five DQN-Full
# policies and Greedy on ONE traffic window and records, per decision, whether a
# V2V peer is reachable, whether it has a free core, the vehicles within V2V range,
# the action taken and the deadline outcome. Writes
# results/peer_audit/eval_peer_availability_<tag>.csv (published values are
# reproduced as a control column).
#
# Usage: sbatch -J peer_<tag> refactor/slurm/eval_peer_availability.sh <pos_csv> <tag> [experiment] [n_episodes]
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
latest_model () {  # released checkpoints first, then locally trained runs
  if [ -f "checkpoints/$1/model.zip" ]; then echo "checkpoints/$1/model.zip"; return; fi
  ls -dt refactor/results/trained_models/dqn/"$1"/run_*/model.zip 2>/dev/null | head -1; }
M=(); for s in 100 101 102 103 104; do M+=("$(latest_model berlin_general_v2v_seed${s})"); done
echo "window=$POS_CSV tag=$TAG experiment=$EXP episodes=$N models=${M[*]}"
python refactor/scripts/eval_peer_availability.py --experiment "$EXP" --pos-csv "$POS_CSV" \
    --n-episodes "$N" --v2v-models "${M[@]}" --out-csv "results/peer_audit/eval_peer_availability_${TAG}.csv"
