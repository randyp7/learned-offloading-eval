# Evaluation design for learned task offloading — artefact

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
| `refactor/scripts/` | `fcd_to_csv.py` (SUMO trajectory to replay CSV), `make_paired_fragments.py` (paired warm/cold evaluation windows), `evaluate_models.py` (seeded evaluation of trained policies and rule-based baselines), `eval_greedy_nov2v.py` (Greedy with and without V2V on the same episodes), `eval_peer_availability.py` (per-decision peer-availability audit of the shared evaluation episodes). |
| `refactor/slurm/` | The job scripts used to run the campaign on a SLURM cluster; each documents its arguments. They also run with `bash` once the `module`/`conda` lines are replaced by your own environment activation. |
| `checkpoints/` | Final trained policies, five training seeds per arm: `berlin_general_{v2v,local}_seed10{0..4}` (warm continuous-replay training, with and without vehicle-to-vehicle offloading) and `berlin_coldtrain_{v2v,local}_seed10{0..4}` (cold-start training arm), and `berlin_energyaware_{v2v,local}_seed10{0..4}` (energy-in-reward arm, trained with the composite reward that adds energy terms). Stable-Baselines3 DQN format. |
| `results/` | The evaluation outputs behind every number in the paper (one CSV per evaluation run); see *Paper tables and result files* below. `results/superseded/` holds earlier runs that the paper does not use. |
| `data/paired/` | The seeded start times of the paired windows, the per-window density statistics used in the diurnal density figure, and `eval_sequence_<period>.csv`, the 24 start times actually evaluated in each period (drawn with replacement from 15 distinct times). |
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

### Paper tables and result files

All learned-policy values are computed per training seed (pooled over every task of the
evaluated episodes) and then averaged over the five seeds; Greedy, Local and Random are
single values on the same shared evaluation set (seed 7000, 20 episodes per period, or 24
paired episodes in the warm/cold experiment).

| Paper item | Result files (`results/`) | Generated table |
|---|---|---|
| Service by operating period (Sect. 5, table and density figure) | `eval_main_{00-04,20-24,08-09,16-18}.csv`, full day `deadline_only/eval_general_24h.csv`; the 04–08 check in the text uses `eval_main_04-08.csv` | `tab_service.tex`, `fig_density.pdf` |
| Greedy-NoV2V comparison (Sect. 5) | `greedy_nov2v/eval_greedy_nov2v_{main_<period>,general_24h}.csv` | (text) |
| Peer availability and action mix (Sect. 5) | `peer_audit/eval_peer_availability_{main_<period>,general_24h}.csv`; action-mix rows from `eval_main_<period>.csv` and `deadline_only/eval_general_24h.csv` (columns `frac_local`, `frac_v2v`, `frac_rsu`) | `tab_peer.tex` |
| Warm/cold evaluation (Sect. 6) | `eval_paired_{warm,cold}_{00-04,20-24,08-09,16-18,24h}.csv`; densities from `data/paired/{warm,cold}_<period>/_stats.csv` weighted by `data/paired/eval_sequence_<period>.csv` | `tab_cold.tex` |
| Cold-start training (Sect. 6) | `eval_coldtrain_paired_{warm,cold}_<period>.csv` against the warm/cold evaluation files | `tab_coldtrain.tex` |
| Energy ledgers (Sect. 7) | full day `deadline_only/eval_general_24h.csv`, evening peak `eval_main_16-18.csv`; ledgers for every period are in `eval_main_<period>.csv` (columns `energy_user_j_per_task`, `energy_system_j_per_task`) | `tab_energy.tex` |
| Energy in the reward (Sect. 7) | `eval_main_<period>.csv` (DQN-Full) against `eval_main_ea_<period>.csv` (DQN-Energy) | `tab_reward.tex` |
| Diurnal density figure (Sect. 3) | `data/paired/cold_<period>/_stats.csv` and the 24-hour trajectory | `fig_diurnal.pdf` |

In the result files, `RSU-V2V-Local` is DQN-Full, `RSU-Local` is DQN-NoV2V, and
`greedy`, `local` and `random` are the three rules.

`results/superseded/` contains (i) `eval_paired_*_prefix.csv`, an earlier run of the
warm/cold experiment in which per-window statistics files written into the fragment
directories shifted some episodes onto a neighbouring start time, so the warm and cold
members of a pair were not always matched (the files in `results/` are the corrected
re-run, in which every policy and both arms replay the identical start-time sequence), and
(ii) `*_20260707.csv`, evaluations from an earlier configuration. Neither is used in the paper.

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

`eval_window.sh` picks up the policies in `checkpoints/` automatically. The last argument selects the arm: `berlin_general`, `berlin_coldtrain` or `berlin_energyaware`.

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
- Traffic source: the road network (`sumo/berlin.net.xml.gz`) and the corridor demand
  (`sumo/a100-east.rou.xml.gz`) come from the open Berlin SUMO scenario (BeST v1.1) of
  Schrab et al., "A Large-Scale Traffic Scenario of Berlin for Evaluating Smart Mobility
  Applications" (2023). Its demand is the synthetic MATSim Open Berlin Scenario
  (Ziemke et al., Procedia Computer Science 151, 2019), calibrated against traffic counts
  and re-assigned in SUMO; Schrab et al. report that the simulated volumes on the A100
  closely match public traffic counts for an average working day. The corridor routes were
  cut from the scenario's full-day route file (`berlin-matsim_103.rou.gz`) with SUMO's
  `cutRoutes.py`.
