#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 2
#SBATCH -t 2-0
#SBATCH -o logs/train_%x_%j.out
#SBATCH -e logs/train_%x_%j.err
#
# Cold-trained arm: identical to train_berlin.sh (DQN, state v6, task v5, RSU v2,
# lambda 0.5, deadline_only reward, 1.5M steps) EXCEPT episodes are 20 frames drawn
# from data/coldtrain/ (cold-started fragments) instead of 100 frames from the warm
# 24 h trace. Scenario name prefix berlin_coldtrain_ so eval_window.sh can select it.
# Usage: sbatch -J ct_v2v_s100 refactor/slurm/train_coldtrain.sh v2v 100
set -euo pipefail
VARIANT="${1:?v2v|local}"; SEED="${2:?seed}"
V2V_FLAG=""; [[ "$VARIANT" == "local" ]] && V2V_FLAG="--no-v2v"
module load miniforge/24.11.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ito_env
export SUMO_HOME="$(python -c 'import os, sumo; print(os.path.dirname(sumo.__file__))' 2>/dev/null || true)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
SCEN="berlin_coldtrain_${VARIANT}_seed${SEED}"
echo "=== cold-trained arm variant=$VARIANT seed=$SEED scenario=$SCEN node=$(hostname) start=$(date -u +%FT%TZ) ==="
python refactor/src/train/train_model.py \
    --algorithm dqn --scenario "$SCEN" --timesteps 1500000 --seed "$SEED" \
    --pos-csv data/coldtrain \
    --episode-length 20 \
    --state-space-version 6 --task-profile-version 5 --rsu-profile-version 2 \
    --lambda-per-veh 0.5 --reward-type deadline_only $V2V_FLAG --no-register
echo "=== done $(date -u +%FT%TZ) ==="
