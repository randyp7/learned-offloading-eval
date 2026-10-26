#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 2
#SBATCH -t 6:00:00
#SBATCH -o logs/cold_%x_%j.out
#SBATCH -e logs/cold_%x_%j.err
#
# Cold-start arm of the paired fragment experiment (CAiSE 2027, RQ1).
# For each start time in data/paired/starts_<win>.json, run SUMO from an EMPTY
# network at that time (--begin T) and keep the first 100 non-empty frames,
# writing data/paired/cold_<win>/frag_t<T>.csv (same names as warm_<win>/).
#
# Usage: sbatch -J cold_<win> refactor/slurm/cold_fragments.sh <win>
#   <win> in 00-04 | 08-09 | 16-18 | 20-24 | 24h
set -euo pipefail
WIN="${1:?usage: $0 <window>}"

module load miniforge/24.11.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ito_env
SUMO_BIN="$(python -c 'import os, sumo; print(os.path.join(os.path.dirname(sumo.__file__), "bin", "sumo"))')"
export SUMO_HOME="$(dirname "$(dirname "$SUMO_BIN")")"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
echo "=== cold fragments window=$WIN sumo=$SUMO_BIN start=$(date -u +%FT%TZ) ==="
python refactor/scripts/make_paired_fragments.py cold \
    --starts "data/paired/starts_${WIN}.json" \
    --cfg sumo/berlin_a100_16-18.sumocfg \
    --out "data/paired/cold_${WIN}" \
    --sumo "$SUMO_BIN" --seed 42
python refactor/scripts/make_paired_fragments.py stats --dir "data/paired/cold_${WIN}"
python refactor/scripts/make_paired_fragments.py stats --dir "data/paired/warm_${WIN}"
echo "=== done $(date -u +%FT%TZ) ==="
