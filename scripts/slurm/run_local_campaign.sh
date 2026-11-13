#!/bin/bash
# Local 10-model campaign runner. Single-threaded per model (torch=1);
# concurrency capped to fit RAM. Usage: <pos_csv_or_dir> <steps> <concurrency>
set -uo pipefail
export SUMO_HOME="${SUMO_HOME:?set SUMO_HOME to your SUMO 1.27 install}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$(dirname "$0")/../.."
POS="${1:-data/fcd/general/berlin_t28800.csv}"
STEPS="${2:-1500000}"
CONC="${3:-3}"
mkdir -p local_logs
echo "data=$POS steps=$STEPS concurrency=$CONC start=$(date -u +%FT%TZ)"
pids=()
for s in 100 101 102 103 104; do
  for v in v2v local; do
    while [ "${#pids[@]}" -ge "$CONC" ]; do
      wait -n 2>/dev/null || true
      alive=(); for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive+=("$p"); done
      pids=("${alive[@]}")
    done
    flag=""; [ "$v" = "local" ] && flag="--no-v2v"
    scen="berlin_general_${v}_seed${s}"
    echo "  launch $scen ($(date -u +%T))"
    python3 refactor/src/train/train_model.py --algorithm dqn --scenario "$scen" \
      --timesteps "$STEPS" --seed "$s" --pos-csv "$POS" --episode-length 100 \
      --state-space-version 6 --task-profile-version 5 --rsu-profile-version 2 \
      --lambda-per-veh 0.5 $flag --no-register > "local_logs/${scen}.log" 2>&1 &
    pids+=($!)
  done
done
wait
echo "ALL TRAINING DONE: $(date -u +%FT%TZ)"
echo "models saved: $(ls refactor/results/trained_models/dqn/berlin_general_*/run_*/model.zip 2>/dev/null | wc -l)/10"
