#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 2
#SBATCH -t 2-0
#SBATCH -o logs/train_%x_%j.out
#SBATCH -e logs/train_%x_%j.err
#
# Train ONE Berlin V2V DQN model (one variant, one seed).
# SUMO runs in-the-loop (TraCI) so this is CPU-bound -> multicore, NOT gpuA.
#
# Both variants train on the FULL Berlin 24h network (report §4.5); the A100
# peak scenario is an EVALUATION scenario, not a training one.
#
# Usage (from repo root):
#   mkdir -p logs
#   # Live SUMO (slow — reloads the net every episode):
#   sbatch -J v2v_s100   refactor/slurm/train_berlin.sh v2v   100
#   # FCD replay (fast — no live SUMO; pass the position CSV/fragment dir as 3rd arg):
#   sbatch -J v2v_s100   refactor/slurm/train_berlin.sh v2v   100 data/fcd/general
#
# The canonical config below matches training_config/berlin_v2v_scenario.py:
#   state-space v6 (30 feat), task-profile v5, rsu-profile v2 (40 GHz),
#   episode-length 100, lambda 0.5 (see berlin_v2v_scenario.py header for the
#   differences from the original study).
set -euo pipefail

VARIANT="${1:?usage: $0 <v2v|local> <seed> [pos_csv]}"
SEED="${2:?usage: $0 <v2v|local> <seed> [pos_csv]}"
POS_CSV="${3:-}"   # optional: FCD-derived position CSV or fragment dir (replay mode)

V2V_FLAG=""
if [[ "$VARIANT" == "local" ]]; then
    V2V_FLAG="--no-v2v"
elif [[ "$VARIANT" != "v2v" ]]; then
    echo "variant must be 'v2v' or 'local'"; exit 1
fi

# Mobility source: live SUMO (--date) or FCD replay (--pos-csv, no SUMO).
if [[ -n "$POS_CSV" ]]; then
    MOBILITY_ARGS=(--pos-csv "$POS_CSV")
    echo "mobility: FCD replay from $POS_CSV"
else
    MOBILITY_ARGS=(--date 2023-10-18)
    echo "mobility: live SUMO (2023-10-18, full 24h)"
fi

module load miniforge/24.11.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ito_env    # <-- your env (see slurm/README.md); must have eclipse-sumo==1.27.0

# SUMO_HOME is only needed for LIVE SUMO. With FCD replay (--pos-csv) SUMO is
# never launched, so this is best-effort: if the eclipse-sumo `sumo` package
# isn't installed (lean env with just traci+sumolib), SUMO_HOME stays empty and
# that is fine for replay.
export SUMO_HOME="$(python -c 'import os, sumo; print(os.path.dirname(sumo.__file__))' 2>/dev/null || true)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1  # tiny net: 1 thread is fastest
echo "SUMO_HOME=$SUMO_HOME"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

SCEN="berlin_general_${VARIANT}_seed${SEED}"
echo "=== train variant=$VARIANT seed=$SEED scenario=$SCEN node=$(hostname) start=$(date -u +%FT%TZ) ==="

# --reward-type deadline_only = the report's §4.4 reward EXACTLY (binary SLA
# ±1 x category weight 2.0/1.0/0.5). The code default (composite with energy
# weights 0.15/0.05) deviates from the report AND would invalidate the Green IS
# "energy saving as a by-product" claim — energy must be measured, not rewarded.
python refactor/src/train/train_model.py \
    --algorithm dqn \
    --scenario "$SCEN" \
    --timesteps 1500000 \
    --seed "$SEED" \
    "${MOBILITY_ARGS[@]}" \
    --episode-length 100 \
    --state-space-version 6 \
    --task-profile-version 5 \
    --rsu-profile-version 2 \
    --lambda-per-veh 0.5 \
    --reward-type deadline_only \
    $V2V_FLAG \
    --no-register

echo "=== done $(date -u +%FT%TZ) ==="
echo "Model: refactor/results/trained_models/dqn/${SCEN}/run_*/model.zip"
