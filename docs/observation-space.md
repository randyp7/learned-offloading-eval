# DQN observation and action spaces

The DQN receives a 30-dimensional observation (state-space version 6, `envs/offload_task_env.py`, `build_features_v6`). Features use fixed transformations defined in the code rather than sample-based standardisation.

## Task features

| Index | Feature | Transformation |
|---:|---|---|
| 0 | CPU cycles | `(log10(max(1, cycles)) - 8) / 2` |
| 1 | Input size in bytes | `(log10(max(1, bytes)) - 8) / 2` |
| 2 | Deadline in seconds | `deadline_s * 10` |
| 3 | Result-size ratio | Raw value; 0.05 in the reported experiments |
| 4–6 | Task class | One-hot safety / automated driving / infotainment |

An unknown task class is encoded as automated driving.

## Communication features

Each alternative contributes five features, in the order local (indices 7–11), V2V (12–16) and RSU (17–21). The values of an unavailable alternative are encoded as zero.

| Offset | Feature | Transformation |
|---:|---|---|
| +0 | Valid alternative | 1.0 if valid |
| +1 | Distance | `distance_m / 100` |
| +2 | Uplink CQI | `CQI / 15` |
| +3 | Uplink time | Raw seconds |
| +4 | Uplink requester energy | Raw joules |

The local alternative is always valid and has zero distance, CQI, transfer time and transfer energy.

## Context features

| Index | Feature | Transformation |
|---:|---|---|
| 22 | Vehicles within V2V range of the requester | `count / 10` |
| 23 | Requesting-vehicle speed | `speed_m_s / 30` |

The count uses the configured V2V range. In the DQN-NoV2V arm the V2V range is set to 0 m, so feature 22 is always zero for that arm.

## Processing and queue estimates

| Index | Feature | Transformation |
|---:|---|---|
| 24 | Local estimated processing time | `cycles / per-core clock of the requester * 2` |
| 25 | V2V estimated processing time | `cycles / per-core clock of the preselected peer * 2` (0 if no peer) |
| 26 | RSU estimated processing time | `cycles / aggregate RSU clock (8 × 2.0 GHz) * 2` |
| 27 | V2V queue wait | `queue_s * 2` (0 if no peer) |
| 28 | RSU queue wait | `queue_s * 2` |
| 29 | Local queue wait | `queue_s * 2` |

The processing-time estimates ignore multi-core execution. The vehicle estimates use a single core and are therefore pessimistic relative to the executed scheduling model; the RSU estimate uses the aggregate clock and is therefore optimistic.

The queue waits are taken from a trial schedule of the task on a copy of the executor's core queue (`queue_s` in `_evaluate_candidates`). The candidates of all tasks generated in a frame are evaluated when the frame begins, before any of these tasks is committed. The queue features therefore do not include other tasks of the same frame.

## Final numerical handling

- NaN values are replaced by zero.
- Positive and negative infinity are replaced by ±1e6.
- The final observation is clipped to ±1e6.

## Executor preselection

The DQN chooses an executor class (action 0 local, 1 V2V, 2 RSU), not a physical node.

- RSU selects the nearest roadside unit. The trajectory-replay environment applies no coverage limit to the RSU link (the 250 m `RSU_DEFAULT_RANGE_M` in `config.py` is not used in replay mode). The RSU alternative is therefore valid for every task, and link quality falls with distance through the radio model.
- V2V selects, among vehicles within 200 m of the requester, the peer with the earliest estimated processing finish time. For this ranking the uplink time is approximated at a constant 50 Mbit/s, and the full radio model is then applied to the selected peer.
- Local selects the requesting vehicle.

If the DQN selects an unavailable offloading class, execution falls back to Local, and an invalid-action penalty of −3 is added to that task's reward. The fallback applies during both epsilon-greedy training and deterministic evaluation; the penalty affects only training. In practice only the V2V class can be unavailable, because the RSU class is always valid.
