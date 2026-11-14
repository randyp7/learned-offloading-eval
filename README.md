# Evaluation Design for Learned Task Offloading in Vehicular Edge Computing — Replication Package

This repository accompanies the paper *Designing Credible Evaluations of Learned Components
in Information Systems: A Vehicular Edge Computing Study* (under double-blind review). It contains everything
needed to regenerate the traffic data, retrain and re-evaluate the policies, and
rebuild the result tables of the paper.

## Materials supporting the paper

This replication package contains the executable and supplementary materials supporting the paper's three-dimensional evaluation protocol.

### Core reproduction materials

- continuous-trajectory generation and replay code;
- matched warm/cold window generator;
- policy-training and evaluation scripts;
- executable experiment configurations;
- trained checkpoints for the reported training runs;
- evaluation seeds and window identifiers;
- machine-readable result data;
- scripts generating the paper's principal tables; and
- pinned software dependencies and execution instructions.

### Supplementary analyses

The following documents contain the analyses and details removed from the main paper to satisfy the conference page limit:

- [`docs/supplementary-service-analysis.md`](docs/supplementary-service-analysis.md)
  - complete peer-availability table;
  - executor-load audit;
  - Greedy-NoV2V comparison;
  - morning-ramp robustness analysis.
- [`docs/supplementary-state-continuity.md`](docs/supplementary-state-continuity.md)
  - complete warm/cold executor-state audit;
  - sparse-night diagnostic analysis;
  - exploratory cold-start training experiment;
  - cold-training interaction estimates and confidence intervals.
- [`docs/supplementary-energy-analysis.md`](docs/supplementary-energy-analysis.md)
  - complete DQN-Energy reward-ablation table;
  - seed-level reward results;
  - energy per deadline-satisfying task;
  - analytical energy break-even derivation;
  - task-class-specific RSU transfer thresholds.
- [`docs/statistical-analysis.md`](docs/statistical-analysis.md)
  - aggregation order;
  - matched-run justification;
  - paired and Welch-test sensitivity;
  - treatment of deterministic comparators;
  - limitations of five-run inference.
- [`docs/related-work-evidence-matrix.md`](docs/related-work-evidence-matrix.md)
  - evaluation conditions;
  - initialisation reporting;
  - energy-accounting boundaries; and
  - training replication for the representative studies discussed in the Related Work.

### Traffic-scenario provenance

This README records (see *Reproducing the paper* and *Notes* below):

- the Berlin scenario version;
- the source-route data;
- the transformation used to generate the evaluated corridor;
- the traffic and evaluation seeds;
- the applicable third-party licence; and
- the commands required to regenerate the trajectory.

### Scope of the package

The package supports reproduction of the paper's principal tables and inspection of the supplementary analyses. The paper remains authoritative for the research questions, contribution claims and interpretation; executable configuration files and source code are authoritative for implementation details.

## Related-work evidence matrix

[`docs/related-work-evidence-matrix.md`](docs/related-work-evidence-matrix.md) documents the four representative task-offloading studies discussed in Sect. 2 of the paper. For each study, it records:

- technical method and execution setting;
- operating conditions evaluated;
- simulation or episode initialisation;
- energy-accounting boundary;
- service measures; and
- reported training replication.

Its Sources section gives page, section, figure, table or equation references to the inspected publication for each entry. Where information could not be identified, the matrix records it as "not identified" in the inspected paper rather than inferring that the implementation omitted it.

The matrix supports the paper's bounded statement that the four studies do not consistently report operating-condition coverage, state continuity and accounting scope. It does not support a field-wide claim that these aspects are never reported.

## Reference verification

[`docs/reference-verification.md`](docs/reference-verification.md) records how each reference cited in the paper was verified. For every reference it gives:

- the authoritative bibliographic record (DOI or official source);
- each claim the paper cites it for, checked separately for every use;
- a supporting passage from the full text of the source; and
- the level of support.

## Relationship between the paper and this package

The paper reports the assumptions and parameters required to interpret the experimental
contrasts. To respect the CAiSE page limit, implementation-level details are documented in
this package rather than repeated in the manuscript:

| Document | Contents |
|---|---|
| [`docs/experimental-configuration.md`](docs/experimental-configuration.md) | software environment, corridor, vehicle and RSU computing resources, task generation |
| [`docs/observation-space.md`](docs/observation-space.md) | the 30 observation features with indices and scaling, executor preselection, invalid actions |
| [`docs/rewards.md`](docs/rewards.md) | deadline reward and energy-aware reward with executable coefficients |
| [`docs/scheduling-and-comparators.md`](docs/scheduling-and-comparators.md) | multi-core processor scheduling; Greedy, Local and Random |
| [`docs/radio-and-energy-model.md`](docs/radio-and-energy-model.md) | radio model and the requester and system energy ledgers |
| [`docs/randomisation-and-evaluation.md`](docs/randomisation-and-evaluation.md) | training and evaluation randomisation, aggregation, released matching information |
| [`docs/evaluation-protocol.md`](docs/evaluation-protocol.md) | the three evaluation dimensions (operating-condition coverage, state continuity, accounting scope) and how this study operationalises them |
| [`docs/warm-cold-protocol.md`](docs/warm-cold-protocol.md) | construction of the matched warm/cold window pairs, what each member preserves or resets, released pair files |
| [`docs/statistical-analysis.md`](docs/statistical-analysis.md) | units of analysis, aggregation order, paired and Welch tests |
| [`docs/parameter-provenance.md`](docs/parameter-provenance.md) | source and status of every major parameter |
| [`docs/supplementary-service-analysis.md`](docs/supplementary-service-analysis.md) | RQ1: full peer-availability table, executor-load audit, Greedy-NoV2V, morning-ramp robustness sample |
| [`docs/supplementary-state-continuity.md`](docs/supplementary-state-continuity.md) | RQ2: complete paired-window audit, night condition, exploratory cold-training analysis |
| [`docs/supplementary-energy-analysis.md`](docs/supplementary-energy-analysis.md) | RQ3: ledgers for every operating condition, energy per deadline-satisfying task, DQN-Energy reward ablation, analytical break-even conditions |
| [`docs/supplementary-discussion-evidence.md`](docs/supplementary-discussion-evidence.md) | evidence index mapping each interpretive claim of the Discussion to its supporting files and interpretation boundary |
| [`docs/related-work-evidence-matrix.md`](docs/related-work-evidence-matrix.md) | evidence for the bounded statements about representative vehicular task-offloading studies in the Introduction and Related Work |
| [`docs/reference-verification.md`](docs/reference-verification.md) | bibliographic record, cited claims and supporting source passage for every reference in the paper |
| [`results/supplementary/`](results/supplementary/) | full-day traffic-density profile (`fig_diurnal.pdf`, `.png`) removed from the paper for space |
| [`configs/`](configs/) | one record per training arm (`dqn_full.yaml`, `dqn_nov2v.yaml`, `dqn_energy.yaml`, `cold_training.yaml`) and the evaluation protocol (`evaluation.yaml`) |

The paper remains authoritative for the research questions, experimental contrasts and
interpretation. The released source code and the parameters it pins (`config.py`,
`refactor/src/training_config/`, the job scripts in `scripts/slurm/`) are authoritative for
executable implementation details; the documents and the files in `configs/` describe them
and are not read by the code.

**Recommended reproduction sequence.** (1) Create the documented software environment.
(2) Regenerate the calibrated traffic trajectory and the evaluation windows (steps 2 and 3
below). (3) Evaluate the released checkpoints on the recorded episode start times (step 4).
(4) Rebuild the tables and figures from the new result files (step 1). (5) Compare the new
result files with the released ones in `results/` and the generated tables with the paper. Training from scratch (step 5) is optional because the five
final checkpoints of every reported training arm are included.

## Supplementary evidence supporting the Discussion

The main paper reports the principal evidence for operating-condition coverage, state continuity and accounting scope. To satisfy the conference page limit, detailed diagnostic and exploratory analyses are provided here.

The supplementary material is organised as follows:

- [`docs/supplementary-service-analysis.md`](docs/supplementary-service-analysis.md) documents peer availability, action use, Greedy-NoV2V, executor load and the morning-ramp robustness analysis.
- [`docs/supplementary-state-continuity.md`](docs/supplementary-state-continuity.md) documents warm/cold peer and executor states, the sparse-night audit and the exploratory cold-training analysis.
- [`docs/supplementary-energy-analysis.md`](docs/supplementary-energy-analysis.md) documents the complete DQN-Energy reward ablation, service-normalised energy and analytical break-even calculations.
- [`docs/statistical-analysis.md`](docs/statistical-analysis.md) documents aggregation order, matched-run comparisons, Welch-test sensitivity and the relationship between training-run and evaluation-window variability.
- [`docs/supplementary-discussion-evidence.md`](docs/supplementary-discussion-evidence.md) maps each interpretive claim of the Discussion to the files above.

These analyses support the interpretations in the Discussion but are not required to follow the paper's principal empirical argument.

## Relationship between the Introduction and supplementary evidence

The Introduction reports only the principal contribution and headline findings. Supporting details are distributed as follows:

- literature-claim verification: [`docs/related-work-evidence-matrix.md`](docs/related-work-evidence-matrix.md);
- complete peer and executor audits: [`docs/supplementary-service-analysis.md`](docs/supplementary-service-analysis.md);
- matched warm/cold diagnostics and exploratory cold training: [`docs/supplementary-state-continuity.md`](docs/supplementary-state-continuity.md);
- complete reward-ablation and energy sensitivity analyses: [`docs/supplementary-energy-analysis.md`](docs/supplementary-energy-analysis.md);
- training, evaluation and statistical procedures: [`docs/statistical-analysis.md`](docs/statistical-analysis.md).

The supplementary analyses support the claims made in the Introduction but are not required to follow its principal argument.

## Contents

| path | what it is |
|---|---|
| `docs/`, `configs/` | Implementation-level documentation and configuration records (see above). |
| `sumo/` | The A100 corridor scenario: road network and corridor demand (gzipped), RSU and bus-stop definitions, and `berlin_a100_16-18.sumocfg`. These are exactly the inputs SUMO resolves for the configuration used in the study. |
| `envs/`, `net/`, `utils/`, `config.py` | The vehicular edge computing environment: task model, radio model (NR-V2X), roadside units and the trajectory-replay environment. |
| `refactor/src/` | Training entry point, environment factory and the scenario configuration (`training_config/berlin_v2v_scenario.py`, which pins every parameter used in the paper). |
| `scripts/` | `fcd_to_csv.py` (SUMO trajectory to replay CSV), `make_paired_fragments.py` (paired warm/cold evaluation windows), `evaluate_models.py` (seeded evaluation of trained policies and rule-based baselines), `eval_greedy_nov2v.py` (Greedy with and without V2V on the same episodes), `eval_peer_availability.py` (per-decision peer-availability audit of the shared evaluation episodes), `eval_executor_load.py` (per-decision executor waiting time, busy-core share and backlog, read-only), `coldtrain_density.py` (frame-wise vehicle counts of the cold-start training fragments). |
| `scripts/slurm/` | The job scripts used to run the campaign on a SLURM cluster; each documents its arguments. They run with plain `bash` in the active Python environment; under `sbatch`, set `CONDA_MODULE` (the module that provides conda, if any) and `CONDA_ENV` (the environment built from `requirements.txt`), and add your site's partition option. |
| `checkpoints/` | Final trained policies, five training seeds per arm: `berlin_general_{v2v,local}_seed10{0..4}` (warm continuous-replay training, with and without vehicle-to-vehicle offloading) and `berlin_coldtrain_{v2v,local}_seed10{0..4}` (cold-start training arm), and `berlin_energyaware_{v2v,local}_seed10{0..4}` (energy-in-reward arm, trained with the composite reward that adds energy terms). Stable-Baselines3 DQN format. |
| `results/` | The evaluation outputs behind every number in the paper (one CSV per evaluation run); see *Paper tables and result files* below. `results/superseded/` holds earlier runs that the paper does not use. |
| `data/paired/` | The seeded start times of the paired windows, the per-window density statistics used in the diurnal density figure, and `eval_sequence_<period>.csv`, the 24 start times actually evaluated in each period (drawn with replacement from 15 distinct times). |
| `data/rsu_data.csv` | Roadside-unit positions. |
| `analysis/make_tables.py` | Rebuilds all result tables, `numbers.json` and both figures from `results/` and `data/paired/`. |
| `analysis/make_supplementary.py` | Rebuilds the tables of the three supplementary analysis documents (`docs/supplementary-*.md`) from `results/`; writes `analysis/supplementary_tables.md`. |

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
python analysis/make_supplementary.py   # tables of docs/supplementary-*.md
```

This writes the LaTeX tables, `numbers.json` and the figures into `analysis/`.

### Paper tables and result files

All learned-policy values are computed per training seed (pooled over every task of the
evaluated episodes) and then averaged over the five seeds; Greedy, Local and Random are
single values on the same shared evaluation set (seed 7000, 20 episodes per period, or 24
paired episodes in the warm/cold experiment).

| Paper item | Result files (`results/`) | Generated table |
|---|---|---|
| Service by operating period (Sect. 5.1, table) | `eval_main_{00-04,20-24,08-09,16-18}.csv`, full day `deadline_only/eval_general_24h.csv`; the 04–08 check in the text uses `eval_main_04-08.csv` | `tab_service.tex` (`fig_density.pdf` is no longer in the paper) |
| Greedy-NoV2V comparison (Sect. 5.1 text; table supplementary) | `greedy_nov2v/eval_greedy_nov2v_{main_<period>,general_24h}.csv` | (text) |
| Peer availability and action mix (Sect. 5.1 text; table supplementary) | `peer_audit/eval_peer_availability_{main_<period>,general_24h}.csv`; action-mix rows from `eval_main_<period>.csv` and `deadline_only/eval_general_24h.csv` (columns `frac_local`, `frac_v2v`, `frac_rsu`) | `tab_peer.tex` |
| Warm/cold evaluation (Sect. 5.2) | `eval_paired_{warm,cold}_{00-04,20-24,08-09,16-18,24h}.csv`; densities from `data/paired/{warm,cold}_<period>/_stats.csv` weighted by `data/paired/eval_sequence_<period>.csv` | `tab_cold.tex` |
| Cold-start training (supplementary) | `eval_coldtrain_paired_{warm,cold}_<period>.csv` against the warm/cold evaluation files | `tab_coldtrain.tex` |
| Executor load (Sect. 5.1 and 5.2 text; tables supplementary) | `load_audit/eval_executor_load_{main,paired_warm,paired_cold}_<period>.csv` (DQN-Full, Greedy, Local) | (text) |
| Peer availability on the paired windows (Sect. 5.2 text; tables supplementary) | `peer_audit/eval_peer_availability_paired_{warm,cold}_<period>.csv` (warm-trained) and `peer_audit/eval_peer_availability_coldtrain_paired_{warm,cold}_<period>.csv` (cold-trained) | (text) |
| Cold-start training density by frame (supplementary) | `coldtrain_density/coldtrain_density_frames.csv` | (text) |
| Energy ledgers (Sect. 5.3) | full day `deadline_only/eval_general_24h.csv`, evening peak `eval_main_16-18.csv`; ledgers for every period are in `eval_main_<period>.csv` (columns `energy_user_j_per_task`, `energy_system_j_per_task`) | `tab_energy.tex` |
| Energy in the reward (Sect. 5.3 text; table supplementary) | `eval_main_<period>.csv` (DQN-Full) against `eval_main_ea_<period>.csv` (DQN-Energy) | `tab_reward.tex` |
| Full-day density profile (supplementary) | `data/paired/cold_<period>/_stats.csv`, `data/paired/eval_sequence_<period>.csv` and the 24-hour trajectory | `fig_diurnal.pdf` (copy in `results/supplementary/`) |
| Warm/cold window pairs (Sect. 5.2) | `window-pairs.csv`, `paired_density/paired_density_frames.csv` | (text) |
| Welch sensitivity of the seed-matched tests | `eval_main_<period>.csv`, `eval_main_ea_<period>.csv` | `numbers.json` (`v2v_welch_p_*`, `rw_*.welch_p`) |

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
python scripts/fcd_to_csv.py corridor_fcd.xml data/fcd/corridor_24h_seed42.csv
```

On a cluster, `scripts/slurm/fcd_generate.sh sumo/berlin_a100_16-18.sumocfg 0 86400 42 data/fcd/corridor_24h_seed42.csv` does the same.

**3. Evaluation windows.** Warm windows are slices of the continuous trajectory. Cold
windows are separate SUMO runs started on an empty network at the same seeded times:

```bash
python scripts/make_paired_fragments.py warm --trace data/fcd/corridor_24h_seed42.csv --out data/paired --slices
python scripts/make_paired_fragments.py cold --starts data/paired/starts_16-18.json \
       --cfg sumo/berlin_a100_16-18.sumocfg --out data/paired/cold_16-18 --seed 42   # repeat per window
```

**4. Evaluation of the released policies** on a window (writes `results/eval_<tag>.csv`):

```bash
bash scripts/slurm/eval_window.sh data/paired/corridor_16-18_seed42.csv main_16-18 20 berlin_general
```

`eval_window.sh` picks up the policies in `checkpoints/` automatically. The last argument selects the arm: `berlin_general`, `berlin_coldtrain` or `berlin_energyaware`.

**5. Retraining** (optional; about 2 h per policy on one CPU node):

```bash
bash scripts/slurm/train_berlin.sh v2v   100 data/fcd/corridor_24h_seed42.csv
bash scripts/slurm/train_berlin.sh local 100 data/fcd/corridor_24h_seed42.csv
```

Seeds 100 to 104 were used for each arm. The cold-start training arm is launched with
`scripts/slurm/launch_coldtrain_arm.sh`.

## Notes

- Large derived files (the 24-hour trajectory, the evaluation-window fragments and the
  intermediate training checkpoints) are not included. Steps 2 and 3 regenerate them
  deterministically from the released scenario and seeds.
- Traffic source: the road network (`sumo/berlin.net.xml.gz`) and the corridor demand
  (`sumo/a100-corridor.rou.xml.gz`) come from the open Berlin SUMO Traffic scenario (BeST) of
  Schrab et al., "A Large-Scale Traffic Scenario of Berlin for Evaluating Smart Mobility
  Applications" (2023), https://doi.org/10.1007/978-3-031-23721-8_24,
  https://github.com/mosaic-addons/best-scenario. We used the archive
  distributed by TU Berlin DCAITI (`best-scenario-v2.zip`; network file header: BeST v1.1,
  netedit 1.21.0, 2025-03-26). The BeST demand is the synthetic MATSim Open Berlin Scenario
  (Ziemke et al., Procedia Computer Science 151, 2019), calibrated against traffic counts
  and re-assigned in SUMO; Schrab et al. report that the simulated volumes on the A100
  closely match public traffic counts for an average working day. The corridor routes were
  cut from the scenario's full-day route file (`berlin-matsim_103.rou.gz`) with SUMO's
  `cutRoutes.py`.
- Licence of the scenario files: BeST is licensed under the Creative Commons Attribution
  4.0 International License (https://creativecommons.org/licenses/by/4.0/). The road
  network, the bus-stop definitions and the corridor demand cut from the scenario are
  redistributed in `sumo/` under the same licence, with attribution to Schrab et al. (2023),
  https://doi.org/10.1007/978-3-031-23721-8_24. The corridor demand is a cut of the
  scenario's route file (see above).

## Licence

The code is released under the MIT License (`LICENSE`). Results, trained checkpoints and
data are released under CC BY 4.0, and the BeST-derived files in `sumo/` remain under the
scenario's CC BY 4.0 licence with attribution to Schrab et al. (2023); see `LICENSE-DATA.md`.
