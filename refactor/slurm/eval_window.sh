#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 2
#SBATCH -t 8:00:00
#SBATCH -o logs/evalw_%x_%j.out
#SBATCH -e logs/evalw_%x_%j.err
#
# Evaluate ALL trained models + baselines on ONE specific traffic window
# (position CSV), writing to a caller-named results CSV. Used to build the
# density-response curve (very-low / low / mid / high peak) from the same
# 10 trained models — no retraining.
#
# Usage: sbatch -J eval_<tag> refactor/slurm/eval_window.sh <pos_csv> <out_tag> [n_episodes]
set -euo pipefail

POS_CSV="${1:?usage: $0 <pos_csv> <out_tag> [n_episodes] [scen_prefix]}"
OUT_TAG="${2:?usage: $0 <pos_csv> <out_tag> [n_episodes] [scen_prefix]}"
N_EPISODES="${3:-20}"
# Scenario-name prefix for model resolution — lets us evaluate an archived arm
# (e.g. berlin_energyaware) alongside the default berlin_general models.
SCEN_PREFIX="${4:-berlin_general}"

module load miniforge/24.11.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ito_env
export SUMO_HOME="$(python -c 'import os, sumo; print(os.path.dirname(sumo.__file__))' 2>/dev/null || true)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

latest_model () {  # released checkpoints first, then locally trained runs
  if [ -f "checkpoints/$1/model.zip" ]; then echo "checkpoints/$1/model.zip"; return; fi
  ls -dt refactor/results/trained_models/dqn/"$1"/run_*/model.zip 2>/dev/null | head -1; }
V2V_MODELS=(); LOCAL_MODELS=()
for s in 100 101 102 103 104; do
    v=$(latest_model "${SCEN_PREFIX}_v2v_seed${s}");   [ -n "$v" ] && V2V_MODELS+=("$v")
    l=$(latest_model "${SCEN_PREFIX}_local_seed${s}"); [ -n "$l" ] && LOCAL_MODELS+=("$l")
done
echo "window=$POS_CSV tag=$OUT_TAG episodes=$N_EPISODES prefix=$SCEN_PREFIX models=${#V2V_MODELS[@]}+${#LOCAL_MODELS[@]}"

# --experiment only labels the scenario; with --pos-csv the traffic window
# comes entirely from the replay file.
python refactor/scripts/evaluate_models.py \
    --experiment a100_peak \
    --v2v-models  "${V2V_MODELS[@]}" \
    --local-models "${LOCAL_MODELS[@]}" \
    --baselines greedy random local \
    --pos-csv "$POS_CSV" \
    --n-episodes "$N_EPISODES" \
    --out-csv "results/eval_${OUT_TAG}.csv"

echo "=== done: results/eval_${OUT_TAG}.csv ==="
