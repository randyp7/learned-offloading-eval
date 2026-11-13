# Radio and energy model

## Radio model

Communication uses the urban-micro non-line-of-sight path-loss model of 3GPP TR 36.814, `PL = 36.7 log10(d) + 22.7 + 26 log10(f_GHz)` with `d` in metres (at least 1 m). Signal-to-noise ratio is mapped to spectral efficiency through a fixed 15-level CQI table (`net/nrv2x_lite.py`, `one_nr_sidelink_hop`). The model has no shadowing or fading, so a link of given length and payload always yields the same rate. The noise power is `-174 dBm/Hz + 10 log10(B) + NF`. A rate floor of 2 Mbit/s applies, and each transfer is capped at 5 s.

Each transfer is computed with the transmitter's bandwidth and power and the receiver's noise figure. The antenna gains of both ends are added.

| Parameter | Vehicle | RSU |
|---|---|---|
| Carrier | 5.9 GHz | 5.9 GHz |
| Bandwidth when transmitting | 20 MHz | 80 MHz |
| Transmit power | 0.4 W (26 dBm) | 2.0 W (33 dBm) |
| Receive power draw | 0.8 W | 2.0 W |
| Antenna gain | 3 dBi | 8 dBi |
| Noise figure | 7 dB | 4 dB |

V2V links in both directions and the vehicle-to-RSU uplink therefore use 20 MHz, while the RSU-to-vehicle result return uses 80 MHz. The V2V range is 200 m. The RSU link has no range limit in the trajectory-replay environment (`observation-space.md`).

Both task upload and result return are modelled. Result size is 5% of input size. The return distance is computed from the requester's position predicted for the start of the return transfer. Radio contention, retransmission and interference among concurrent transfers are not modelled.

## Energy ledgers

For a transfer of duration `t`, the transmitter consumes `t × P_tx` and the receiver `t × P_rx`, using the powers in the table above. Computation consumes `cycles × e`, with `e = 1.0 nJ` per cycle on vehicles and `0.15 nJ` per cycle on RSUs.

| Decision | Requester energy `E_user` | Executor energy `E_exec` |
|---|---|---|
| Local | `cycles × 1.0 nJ` | 0 |
| V2V | `t_up × 0.4 W + t_down × 0.8 W` | `cycles × 1.0 nJ + t_up × 0.8 W + t_down × 0.4 W` |
| RSU | `t_up × 0.4 W + t_down × 0.8 W` | `cycles × 0.15 nJ + t_up × 2.0 W + t_down × 2.0 W` |

Here `t_up` is the upload time and `t_down` the result-return time.

The requester ledger therefore contains the energy attributed to the requesting vehicle for local execution and communication. The system ledger is `E_sys = E_user + E_exec`, the sum of requester, communication and executor computation energy. It does not constitute a lifecycle or carbon-accounting model. Idle, baseline and cooling power are not included.

Every generated task is executed to completion, including tasks that miss their deadline, and its full energy is recorded. The reported energy per task (`energy_user_j_per_task`, `energy_system_j_per_task`) is the mean over all generated tasks of the evaluated episodes, whether or not the deadline was met (`read_step_summary` in `refactor/scripts/evaluate_models.py`). No task is excluded from the denominator.
