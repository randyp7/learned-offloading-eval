#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 2
#SBATCH -t 8:00:00
#SBATCH -o logs/ctfrag_%x_%j.out
#SBATCH -e logs/ctfrag_%x_%j.err
#
# Cold-start TRAINING fragments (cold-trained arm, CAiSE 2027 RQ1): reinstates the
# original study's protocol — SUMO launched fresh at a random time of day, 20-step
# episodes on an initially empty road. 2,000 start times split over N shards.
# Usage: for i in 0 1 2 3 4; do sbatch -J ctfrag$i refactor/slurm/coldtrain_fragments.sh $i 5; done
set -euo pipefail
SHARD="${1:?shard index}"; NSHARD="${2:?shard count}"
module load miniforge/24.11.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ito_env
SUMO_BIN="$(python -c 'import os, sumo; print(os.path.join(os.path.dirname(sumo.__file__), "bin", "sumo"))')"
export SUMO_HOME="$(dirname "$(dirname "$SUMO_BIN")")"
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
python refactor/scripts/make_paired_fragments.py coldtrain \
    --cfg sumo/berlin_a100_16-18.sumocfg --out data/coldtrain \
    --n 2000 --frames 20 --horizon 120 --seed 4242 --shard "$SHARD/$NSHARD" --sumo "$SUMO_BIN"
echo "=== shard $SHARD/$NSHARD done $(date -u +%FT%TZ) ==="
