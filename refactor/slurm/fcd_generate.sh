#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 2
#SBATCH -t 12:00:00
#SBATCH -o logs/fcd_%x_%j.out
#SBATCH -e logs/fcd_%x_%j.err
#
# Generate a SUMO FCD trajectory ONCE and convert it to the position CSV that
# VECEnvOffloadCSV replays. After this, training/eval replay the CSV with no
# live SUMO (~100x faster: the 162 MB Berlin net is parsed only here).
#
# This is the reusable primitive — one time window per invocation:
#   sbatch -J fcd_peak refactor/slurm/fcd_generate.sh \
#       sumo/berlin_a100_06-12.sumocfg 28800 32400 42 data/fcd/a100_peak.csv
#
# Args: <sumocfg> <begin_s> <end_s> <seed> <out_csv>
#   begin_s/end_s : seconds of day (8am=28800, 9am=32400, 0=midnight)
#   seed          : SUMO --seed; use 42 to match the env's default seed_val so
#                   the replayed trajectory equals what live SUMO would produce.
#
# For the general_24h training data, generate hourly FRAGMENTS into a directory
# (VECEnvOffloadCSV fragment mode picks a random hour per episode) — see
# slurm/README.md "FCD workflow".
set -euo pipefail

CFG="${1:?usage: $0 <sumocfg> <begin_s> <end_s> <seed> <out_csv>}"
BEGIN="${2:?}"
END="${3:?}"
SEED="${4:?}"
OUT_CSV="${5:?}"

module load miniforge/24.11.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ito_env
export SUMO_HOME="$(python -c 'import os, sumo; print(os.path.dirname(sumo.__file__))')"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs "$(dirname "$OUT_CSV")"
TMP_FCD="$(mktemp --suffix=.xml)"

echo "=== FCD gen cfg=$CFG begin=$BEGIN end=$END seed=$SEED start=$(date -u +%FT%TZ) ==="

# --output-prefix "" cancels any output-prefix baked into the .sumocfg.
sumo -c "$CFG" \
    --seed "$SEED" --step-length 1 --begin "$BEGIN" --end "$END" \
    --output-prefix "" --fcd-output "$TMP_FCD" \
    --no-step-log true --no-warnings true

python refactor/scripts/fcd_to_csv.py "$TMP_FCD" "$OUT_CSV"
rm -f "$TMP_FCD"

echo "=== done $(date -u +%FT%TZ): $OUT_CSV ($(du -h "$OUT_CSV" | cut -f1)) ==="
