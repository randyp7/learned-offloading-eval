#!/bin/bash
# Launch the whole cold-trained arm with SLURM dependencies (run from the repo root on CSF).
# Fragments (5 shards) -> 10 training jobs (afterok all shards) -> 10 evaluations (afterok both variants).
set -euo pipefail
mkdir -p logs
frag=""
for i in 0 1 2 3 4; do j=$(sbatch --parsable -J ctfrag$i refactor/slurm/coldtrain_fragments.sh $i 5); frag="$frag${frag:+:}$j"; echo "ctfrag$i $j"; done
train=""
for s in 100 101 102 103 104; do for v in v2v local; do
  j=$(sbatch --parsable -J ct_${v}_s$s --dependency=afterok:$frag refactor/slurm/train_coldtrain.sh $v $s); train="$train${train:+:}$j"; echo "ct_${v}_s$s $j"
done; done
for w in 00-04 08-09 16-18 20-24 24h; do
  for arm in warm cold; do
    echo "ct_${arm}_$w $(sbatch --parsable -J ct_${arm}_$w --dependency=afterok:$train refactor/slurm/eval_window.sh data/paired/${arm}_$w coldtrain_paired_${arm}_$w 24 berlin_coldtrain)"
  done
done
for w in 00-04 08-09 16-18 20-24; do
  echo "ct_main_$w $(sbatch --parsable -J ct_main_$w --dependency=afterok:$train refactor/slurm/eval_window.sh data/paired/corridor_${w}_seed42.csv coldtrain_main_$w 20 berlin_coldtrain)"
done
