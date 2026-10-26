# Evaluation design for learned task offloading — artifact

This repository accompanies the paper *Evaluation Design for Learned Components in
Edge-Based Information Systems: Traffic State, Density Coverage and Energy Metering
in Vehicular Task Offloading* (under double-blind review). It contains everything
needed to regenerate the traffic data, retrain and re-evaluate the policies, and
rebuild every table and figure in the paper.

## Contents

| path | what it is |
|---|---|
| `sumo/` | The A100 corridor scenario: road network and corridor demand (gzipped), RSU and bus-stop definitions, and `berlin_a100_16-18.sumocfg`. These are exactly the inputs SUMO resolves for the configuration used in the study. |
| `envs/`, `net/`, `utils/`, `config.py` | The vehicular edge computing environment: task model, radio model (NR-V2X), roadside units and the trajectory-replay environment. |
| `refactor/src/` | Training entry point, environment factory and the scenario configuration (`training_config/berlin_v2v_scenario.py`, which pins every parameter used in the paper). |
| `refactor/scripts/` | `fcd_to_csv.py` (SUMO trajectory to replay CSV), `make_paired_fragments.py` (paired warm/cold evaluation windows), `evaluate_models.py` (seeded evaluation of trained policies and rule-based baselines). |
| `refactor/slurm/` | The job scripts used to run the campaign on a SLURM cluster; each documents its arguments. They also run with `bash` once the `module`/`conda` lines are replaced by your own environment activation. |
| `checkpoints/` | Final trained policies, five training seeds per arm: `berlin_general_{v2v,local}_seed10{0..4}` (warm continuous-replay training, with and without vehicle-to-vehicle offloading) and `berlin_coldtrain_{v2v,local}_seed10{0..4}` (cold-start training arm). Stable-Baselines3 DQN format. |
| `results/` | The evaluation outputs behind every number in the paper (one CSV per evaluation run). |
| `data/paired/` | The seeded start times of the paired windows and the per-window density statistics used in Fig. 1. |
| `data/rsu_data.csv` | Roadside-unit positions. |
| `analysis/make_tables.py` | Rebuilds all result tables, `numbers.json` and both figures from `results/` and `data/paired/`. |

## Environment

Python 3.11. Install the pinned dependencies:

```bash
pip install -r requirements.txt
export SUMO_HOME="$(python -c 'import os, sumo; print(os.path.dirname(sumo.__file__))')"
```

All commands below are run from the repository root.

## Reproducing the paper

**1. Tables and figures from the released results** (seconds):

```bash
python analysis/make_tables.py
```

This writes the LaTeX tables, `numbers.json` and the figures into `analysis/`.

**2. The continuous 24-hour corridor trajectory** (about 1 h; about 0.9 GB of CSV):

```bash
sumo -c sumo/berlin_a100_16-18.sumocfg --seed 42 --step-length 1 \
     --begin 0 --end 86400 --fcd-output corridor_fcd.xml --no-step-log true
python refactor/scripts/fcd_to_csv.py corridor_fcd.xml data/fcd/corridor_24h_seed42.csv
```

On a cluster, `refactor/slurm/fcd_generate.sh sumo/berlin_a100_16-18.sumocfg 0 86400 42 data/fcd/corridor_24h_seed42.csv` does the same.

**3. Evaluation windows.** Warm windows are slices of the continuous trajectory. Cold
windows are separate SUMO runs started on an empty network at the same seeded times:

```bash
python refactor/scripts/make_paired_fragments.py warm --trace data/fcd/corridor_24h_seed42.csv --out data/paired --slices
python refactor/scripts/make_paired_fragments.py cold --starts data/paired/starts_16-18.json \
       --cfg sumo/berlin_a100_16-18.sumocfg --out data/paired/cold_16-18 --seed 42   # repeat per window
```

**4. Evaluation of the released policies** on a window (writes `results/eval_<tag>.csv`):

```bash
bash refactor/slurm/eval_window.sh data/paired/corridor_16-18_seed42.csv main_16-18 20 berlin_general
```

`eval_window.sh` picks up the policies in `checkpoints/` automatically.

**5. Retraining** (optional; about 2 h per policy on one CPU node):

```bash
bash refactor/slurm/train_berlin.sh v2v   100 data/fcd/corridor_24h_seed42.csv
bash refactor/slurm/train_berlin.sh local 100 data/fcd/corridor_24h_seed42.csv
```

Seeds 100 to 104 were used for each arm. The cold-start training arm is launched with
`refactor/slurm/launch_coldtrain_arm.sh`.

## Notes

- Large derived files (the 24-hour trajectory, the evaluation-window fragments and the
  intermediate training checkpoints) are not included. Steps 2 and 3 regenerate them
  deterministically from the released scenario and seeds.
- The demand is calibrated to the public traffic-detector counts of the city of Berlin
  (Digitale Plattform Stadtverkehr Berlin, licence dl-de/by-2.0).
