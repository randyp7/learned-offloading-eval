# Parameter provenance

This register identifies the source and status of every major experimental parameter. *Inherited* means taken from the earlier task-offloading implementation on which this study builds (the BSc dissertation cited in the paper) with the value that study reports; *inherited from the previous study's shared code* means taken from the code of that implementation, as shared with this study, where the value differs from the one reported in the dissertation; *selected* means chosen for the present study. Values were transcribed from the executable files named in the last column, which remain authoritative.

## Summary by group

| Parameter group | Source / status |
|---|---|
| Traffic demand | Berlin SUMO scenario (BeST) derived from the count-calibrated MATSim Open Berlin Scenario |
| Task classes and numerical task settings | Inherited from the previous study's shared code (task profile 5); class interpretation informed by vehicular edge literature |
| Vehicle compute classes | Class shares (0.30/0.50/0.15/0.05) inherited, informed by German passenger-car stock statistics; core counts and clock ranges are inherited from the previous study's shared code |
| RSU compute | Inherited from the previous study's shared code (`config.py` defaults: 8 × 2.0 GHz) |
| Radio technology | 3GPP V2X and NR-V2X assumptions |
| Radio numerical settings | Inherited from the earlier implementation; individual values below |
| DQN defaults | Stable-Baselines3 2.3.2 |
| DQN overrides | Inherited from the earlier implementation's DQN configuration and kept for the present experiment |
| Corridor and evaluation conditions | Selected for the present study |
| Reward weights | Inherited from the earlier implementation; treated as assumed priorities |

## Register

| Parameter | Value | Unit | Source / status | Location |
|---|---|---|---|---|
| Corridor | Southern A100, between RSU 1 and RSU 8; ≈4.6 per direction | km | Selected | `sumo/`, `data/rsu_data.csv` |
| Traffic demand | BeST full-day routes cut to the corridor | – | BeST (Schrab et al. 2023), MATSim Open Berlin Scenario | `sumo/a100-corridor.rou.xml.gz` |
| Traffic seed | 42 | – | Selected | README, step 2 |
| Frame length | 1 | s | Selected | `config.py` `STEP_LENGTH_S` |
| Number of RSUs; spacing | 8; 610.5 | –; m | Inherited placement | `data/rsu_data.csv` |
| RSU cores × clock | 8 × 2.0 | GHz | Inherited from the previous study's shared code | `config.py` `RSU_CORES_DEFAULT`, `RSU_PER_CORE_HZ_DEFAULT` |
| RSU computation energy | 0.15 | nJ/cycle | Inherited | `config.py` `ENERGY_PER_CYCLE_J_RSU` |
| Vehicle compute classes | 0.30: 2 × 0.8–1.5; 0.50: 4 × 1.5–2.5; 0.15: 6 × 2.0–3.0; 0.05: 8 × 2.5–5.0 | share; cores × GHz | Shares inherited, informed by German passenger-car stock statistics; cores and clocks are inherited from the previous study's shared code | `config.py` `VEH_CPU_CLASSES` |
| Vehicle computation energy | 1.0 | nJ/cycle | Inherited | `config.py` `ENERGY_PER_CYCLE_J_VEH` |
| Task arrival rate | 0.5 | tasks/s per vehicle | Inherited (earlier study's stated value) | `--lambda-per-veh 0.5` |
| Task class mix | 0.30 / 0.35 / 0.35 | safety / automated / infotainment | Inherited | `config.py` `CATEGORY_MIX` |
| Task cycles, input, deadline | Task profile 5 (`experimental-configuration.md`) | Mcycles, KB, ms | Inherited from the previous study's shared code | `config_loader.py` `TASK_PROFILES[5]` |
| Result-size ratio | 0.05 | – | Inherited | `config.py` `RESULT_SIZE_RATIO` |
| Parallel fraction; `k_max` | `scheduling-and-comparators.md` | – | Inherited | `config.py` `PARALLEL` |
| Parallel efficiency | 0.80 vehicle, 0.90 RSU | – | Inherited | `config.py` `PARALLEL_ETA_*` |
| Per-core overhead | 0.2 | ms | Inherited | `config.py` `THREAD_OVERHEAD_S` |
| Carrier | 5.9 | GHz | 3GPP NR-V2X band assumption | `config.py` `NR_CARRIER_HZ` |
| Bandwidth | 20 vehicle, 80 RSU | MHz | Inherited | `envs/vec_env_offload_nrv2x_m_v01_seeded.py` (`NRRadio`) |
| Transmit power | 0.4 vehicle, 2.0 RSU | W | Inherited | `config.py` `NR_P_TX_*` |
| Receive power | 0.8 vehicle, 2.0 RSU | W | Inherited | `config.py` `NR_P_RX_*` |
| Antenna gain | 3 vehicle, 8 RSU | dBi | Inherited | `NRRadio` in the core environment |
| Noise figure | 7 vehicle, 4 RSU | dB | Inherited | `config.py` `NR_NOISE_FIG_*` |
| Path loss | 3GPP TR 36.814 UMi NLOS | dB | 3GPP | `net/nrv2x_lite.py` |
| Rate floor; transfer cap | 2; 5 | Mbit/s; s | Inherited | `net/nrv2x_lite.py` |
| V2V range | 200 (0 for DQN-NoV2V) | m | Inherited | `config.py` `V2V_RANGE_M`, `--no-v2v` |
| Peer ranking uplink rate | 50 | Mbit/s | Inherited | `config.py` `V2V_RATE_MBPS` |
| Observation | 30 features (state-space version 6) | – | Inherited | `envs/offload_task_env.py` |
| Invalid-action penalty | −3 | – | Inherited | `envs/offload_task_env.py` |
| Class weights | 2.0 / 1.0 / 0.5 | – | Inherited; assumed priorities | `config.py` `REWARD_CFG` |
| Principal reward | `deadline_only` | – | Selected among the implemented rewards | `--reward-type deadline_only` |
| Energy-aware reward coefficients | 3.0, 0.3, 0.15, 0.05; miss −0.5; clip [−3, 5] | – | Inherited | `config.py` `REWARD_CFG` |
| DQN network | 2 × 256, ReLU | – | Inherited override (SB3 default 2 × 64) | `config_loader.py` `DQNHyperparameters` |
| DQN replay buffer | 100,000 | transitions | Inherited override (SB3 default 1,000,000) | as above |
| DQN learning starts | 1,000 | steps | Inherited override (SB3 default 100) | as above |
| DQN target update interval | 1,000 | steps | Inherited override (SB3 default 10,000) | as above |
| DQN learning rate, batch, γ, τ, train frequency, gradient steps | 1e-4, 32, 0.99, 1.0, 4, 1 | – | SB3 2.3.2 defaults (set explicitly) | as above |
| Exploration | ε 1.0 → 0.05 over the first 10% of training | – | SB3 2.3.2 defaults (set explicitly) | as above |
| Training budget | 1,500,000 | decisions | Selected | `train_berlin.sh`, `train_coldtrain.sh` |
| Training episode length | 100 warm; 20 cold-start | frames | Selected (20 reproduces the earlier study's cold-start protocol) | as above |
| Training seeds | 100–104 | – | Selected | as above |
| Evaluation seed | 7000 | – | Selected | `berlin_v2v_scenario.py` `EVAL_SEED` |
| Evaluation episodes | 20 per period; 24 per paired window set | – | Selected | `eval_window.sh` arguments |
| Task-seed base | 12345 | – | Inherited | `env_factory.py` `task_seed_base` |

The hyperparameters stored in every released checkpoint (`checkpoints/*/model.zip`, file `data`) match the DQN rows above. Each checkpoint also records 1,500,000 timesteps and its training seed.
