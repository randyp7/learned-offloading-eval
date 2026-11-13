# Experimental configuration

This document provides implementation-level detail omitted from the paper for space. The paper reports the assumptions needed to interpret the experimental contrasts; this document records the executable configuration needed to inspect and reproduce them. Where this document and the code differ, the code is authoritative (`config.py`, `refactor/src/training_config/`, `envs/`).

## Software environment

- Traffic simulator: SUMO 1.27 (`eclipse-sumo==1.27.0`), used only to generate the trajectories; training and evaluation replay the recorded trajectories and do not run SUMO.
- Reinforcement-learning library: Stable-Baselines3 2.3.2 with PyTorch 2.2.0 (CPU build), Gymnasium 0.29.1.
- Learner: Deep Q-Network (DQN).
- Traffic resolution: one second per frame.
- Python 3.11.

`requirements.txt` records the remaining package versions.

## Traffic corridor

The experiment uses a count-calibrated weekday trajectory from the open Berlin SUMO Traffic scenario (BeST), whose demand is derived from the MATSim Open Berlin Scenario (see the README for sources and licence). The evaluated section is the southern A100 motorway between the first and the last roadside unit. A vehicle that traverses the whole section travels approximately 4.6 km (99th percentile of per-vehicle path length in the 16:00–18:00 slice: 4,634 m), in each of the two travel directions, so the section comprises approximately 9.2 km of carriageway in total. The straight-line east–west extent of the section is 4.28 km. Eight roadside units (RSUs) are placed along it at a regular spacing of 610.5 m (`data/rsu_data.csv`).

The 24-hour trajectory is generated with `sumo/berlin_a100_16-18.sumocfg` and traffic seed 42 (README, step 2). The README records the source scenario version, the route cut and the conversion to the replay format.

## Vehicle computing resources

A vehicle's computing class is sampled when the vehicle is first required in an episode (as a requester or as a candidate peer), and is sampled again in every episode:

| Probability | Cores | Per-core clock range |
|---:|---:|---|
| 0.30 | 2 | 0.8–1.5 GHz |
| 0.50 | 4 | 1.5–2.5 GHz |
| 0.15 | 6 | 2.0–3.0 GHz |
| 0.05 | 8 | 2.5–5.0 GHz |

The per-core clock is drawn uniformly within the corresponding range (`config.py`, `VEH_CPU_CLASSES`). Vehicle computation uses 1.0 nJ per CPU cycle.

Each roadside unit provides eight 2.0 GHz cores (16 GHz in total) and uses 0.15 nJ per CPU cycle. These are the defaults in `config.py` (`RSU_CORES_DEFAULT`, `RSU_PER_CORE_HZ_DEFAULT`). The trajectory-replay environment builds the RSUs from `data/rsu_data.csv`, which has no per-RSU core columns, so these defaults apply. The training scripts also pass `--rsu-profile-version 2` (16 cores at 2.5 GHz); in replay mode this flag does not change the RSU cores or clock, and its aggregate value is not used for any reported quantity.

Class proportions hold in expectation rather than being imposed exactly. Training assignments use the environment random-number generator initialised from the training seed. Evaluation assignments use the generator initialised from the evaluation seed 7000 (documented in `configs/evaluation.yaml`).

## Task generation

Each vehicle present in a frame generates a Poisson-distributed number of tasks per second with mean 0.5 (`--lambda-per-veh 0.5`). Task classes are sampled with probabilities:

- safety: 0.30;
- automated driving: 0.35;
- infotainment: 0.35.

Task cycles, input size and deadline are drawn from normal distributions (task profile 5, `refactor/src/training_config/config_loader.py`) and clipped below at one cycle, one byte and one microsecond respectively, so that every value is positive:

| Class | Cycles | Input size | Deadline |
|---|---|---|---|
| Safety | 600 ± 150 Mcycles | 300 ± 100 KB | 300 ± 50 ms |
| Automated driving | 1,200 ± 300 Mcycles | 800 ± 200 KB | 550 ± 90 ms |
| Infotainment | 3,600 ± 1,200 Mcycles | 3 ± 1 MB | 2,000 ± 400 ms |

Values are mean ± standard deviation; KB and MB are decimal (10^3 and 10^6 bytes). Result size is 5% of input size.

Task streams are generated deterministically from a fixed task-seed base (12345), the episode index within the run and the trace time of the frame (`VECEnvOffloadCSV._generate_tasks_seeded`). They do not depend on the policy, so all policies evaluated on the same episode sequence receive identical task arrivals, classes and sampled task parameters.
