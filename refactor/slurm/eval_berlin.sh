#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 2
#SBATCH -t 8:00:00
#SBATCH -o logs/eval_%x_%j.out
#SBATCH -e logs/eval_%x_%j.err
#
# Evaluate the 10 trained models (v2v + local, seeds 100-104) plus baselines
# on ONE shared seeded 50-episode eval set for a given experiment.
#
# Run AFTER all 10 training jobs have finished.
#
# Usage (from repo root):
#   # Live SUMO:
#   sbatch -J eval_peak refactor/slurm/eval_berlin.sh a100_peak
#   # FCD replay (pass the position CSV for this experiment as 2nd arg):
#   sbatch -J eval_peak refactor/slurm/eval_berlin.sh a100_peak data/fcd/a100_peak.csv
set -euo pipefail

EXPERIMENT="${1:?usage: $0 <general_24h|a100_peak> [pos_csv]}"
POS_CSV="${2:-}"   # optional: FCD position CSV/fragment dir for this experiment

POS_ARGS=()
if [[ -n "$POS_CSV" ]]; then
    POS_ARGS=(--pos-csv "$POS_CSV")
    echo "mobility: FCD replay from $POS_CSV"
fi

module load miniforge/24.11.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ito_env
# Best-effort (FCD replay never launches SUMO; empty SUMO_HOME is fine).
export SUMO_HOME="$(python -c 'import os, sumo; print(os.path.dirname(sumo.__file__))' 2>/dev/null || true)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1  # tiny net: 1 thread is fastest

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Resolve the latest run/model.zip for each variant+seed scenario.
latest_model () {  # $1 = scenario name
    ls -dt refactor/results/trained_models/dqn/"$1"/run_*/model.zip 2>/dev/null | head -1
}

V2V_MODELS=()
LOCAL_MODELS=()
for s in 100 101 102 103 104; do
    v=$(latest_model "berlin_general_v2v_seed${s}")
    l=$(latest_model "berlin_general_local_seed${s}")
    [ -n "$v" ] && V2V_MODELS+=("$v")   || echo "WARN: missing v2v seed $s"
    [ -n "$l" ] && LOCAL_MODELS+=("$l") || echo "WARN: missing local seed $s"
done

echo "v2v models  (${#V2V_MODELS[@]}): ${V2V_MODELS[*]}"
echo "local models(${#LOCAL_MODELS[@]}): ${LOCAL_MODELS[*]}"

# N_EPISODES: at full-Berlin density each episode is ~200k task-decisions, so a
# modest count already aggregates a huge number of tasks. Override via 3rd arg.
N_EPISODES="${3:-20}"

python refactor/scripts/evaluate_models.py \
    --experiment "$EXPERIMENT" \
    --v2v-models  "${V2V_MODELS[@]}" \
    --local-models "${LOCAL_MODELS[@]}" \
    --baselines greedy random local \
    "${POS_ARGS[@]}" \
    --n-episodes "$N_EPISODES" \
    --out-csv "refactor/results/eval_${EXPERIMENT}.csv"

echo "=== eval done: refactor/results/eval_${EXPERIMENT}.csv ==="
