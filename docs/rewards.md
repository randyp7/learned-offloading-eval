# Reward definitions

The reward is computed per task by `_compute_task_reward` in `envs/vec_env_offload_nrv2x_m_v01_seeded.py`, with the weights in `config.py` (`REWARD_CFG`).

## Principal deadline reward

DQN-Full, DQN-NoV2V and the cold-start training arm are trained with `--reward-type deadline_only`. For task `i` of class `c_i`, delay `d_i` and deadline `D_i`:

```
r_i = w_c_i * (2 * I[d_i <= D_i] - 1)
```

The delay `d_i` runs from the decision to the return of the result (uplink, queueing, processing and result return).

Class weights are:

- safety: 2.0;
- automated driving: 1.0;
- infotainment: 0.5.

These weights are inherited from the earlier implementation and encode an assumed, rather than externally validated, safety-first priority.

If the selected offloading class is unavailable, −3 is added to `r_i` (see `observation-space.md`). The discount factor is 0.99.

## Energy-aware reward

The DQN-Energy arm (`checkpoints/berlin_energyaware_*`) is trained with the composite reward that the code uses when no `--reward-type` is given:

```
r_i^E = clip( w_c_i * [ 3.0 * s_i - 0.3 * min(2, d_i / D_i) - 0.15 * min(3, E_user_i / E_ref_i) - 0.05 * min(3, E_sys_i / E_ref_i) ], -3, 5 )
```

where:

- `s_i = 1` if the deadline is met and `-0.5` otherwise;
- `min(2, d_i / D_i)` is set to 2 when the delay is not finite;
- `E_ref_i` is the computation energy of executing the task locally, `cycles × 1.0 nJ`;
- `E_user_i` is the requester energy (requester ledger in `radio-and-energy-model.md`);
- `E_sys_i` is the system energy: requester, communication and executor computation energy.

The PPO-specific shaping terms in `REWARD_CFG` (`ppo_mode`) are disabled. The invalid-action penalty applies as for the principal reward.

Because requester energy is included in both `E_user_i` and `E_sys_i`, the effective coefficient on requester energy is 0.20, compared with 0.05 for executor and communication energy that the requester does not bear (as long as neither ratio reaches its cap of 3).

The released configuration and source code are authoritative if any difference is found between this documentation and generated experiment metadata.
