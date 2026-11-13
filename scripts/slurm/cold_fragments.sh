#!/bin/bash --login
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
# Usage: sbatch -J cold_<win> scripts/slurm/cold_fragments.sh <win>
#   <win> in 00-04 | 08-09 | 16-18 | 20-24 | 24h
set -euo pipefail
WIN="${1:?usage: $0 <window>}"

# Environment: on a cluster set CONDA_MODULE (the module that provides conda) and
# CONDA_ENV (the environment from requirements.txt); with neither set, the active
# Python environment is used, so the script also runs with plain `bash`.
if [ -n "${CONDA_MODULE:-}" ] && command -v module >/dev/null 2>&1; then module load "$CONDA_MODULE"; fi
if [ -n "${CONDA_ENV:-}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi
SUMO_BIN="$(python -c 'import os, sumo; print(os.path.join(os.path.dirname(sumo.__file__), "bin", "sumo"))')"
export SUMO_HOME="$(dirname "$(dirname "$SUMO_BIN")")"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
echo "=== cold fragments window=$WIN sumo=$SUMO_BIN start=$(date -u +%FT%TZ) ==="
python scripts/make_paired_fragments.py cold \
    --starts "data/paired/starts_${WIN}.json" \
    --cfg sumo/berlin_a100_16-18.sumocfg \
    --out "data/paired/cold_${WIN}" \
    --sumo "$SUMO_BIN" --seed 42
python scripts/make_paired_fragments.py stats --dir "data/paired/cold_${WIN}"
python scripts/make_paired_fragments.py stats --dir "data/paired/warm_${WIN}"
echo "=== done $(date -u +%FT%TZ) ==="
