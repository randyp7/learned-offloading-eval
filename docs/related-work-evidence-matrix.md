# Evidence matrix for claims about vehicular task-offloading evaluation

The Introduction and Related Work make bounded claims about representative vehicular task-offloading studies. This matrix records the evidence supporting those claims.

| Study | Method | Service measure | Energy measure | Simulation | Operating conditions | Initialisation stated | Training runs |
|---|---|---|---|---|---|---|---|
| Zhan et al. (2020) | Single-agent DRL (PPO); one representative vehicle, V2I only | Average task latency; weighted latency–energy cost; no deadline | Requester-side energy (local computation and upload); server computation excluded | Yes (authors' own simulation) | One synthetic vehicle; sweeps of latency weight (8 values) and task arrival rate (10 values) | Partly: evaluation initial queue and position randomised but identical across algorithms; no warm-up stated | Not identified; one learning curve per architecture; evaluation averaged over episodes; no variance reported |
| Wei et al. (2024) | Multi-agent DRL for many-to-many offloading (V2V, RSU, inter-RSU and cloud tiers, plus local execution) | Completion ratio within deadline; average service latency; utility-based social welfare | Not reported | Yes (authors' own Python simulation) | Synthetic mobility; 3 vehicle counts and 3 load levels | Partly: entry speed and entry process; episode reset state not identified | Not identified; one learning curve per method; box plots of unstated basis |
| Fan et al. (2023) | Joint V2I/V2V optimisation (decomposition algorithm and heuristic; no learning) | Total task processing delay; delay tolerance as a constraint | Not reported | Yes (MATLAB numerical simulation) | Synthetic one-shot snapshot; six one-factor sweeps (e.g. 4–32 vehicles, speed, bandwidth) | Partly: initial positions defined as model variables; generation method and warm-up not identified | Not applicable (no learned policy); number of repetitions not identified |
| Wang et al. (2025) | Hybrid V2V/V2I double-layer Stackelberg game with heuristic search (no learning) | Mean task completion delay; deadline as a constraint | System energy: requester upload, peer and RSU computation; result return ignored | Yes (authors' own simulation; tool not identified) | Synthetic car-following mobility; 20–70 vehicles and 5–30 MB task size (6 levels each) | Partly: random positions and speeds per run; no warm-up or queue state | Not applicable (no learned policy); over 5,000 simulation runs per point; no variance reported |

## Interpretation

The paper does not claim that all vehicular task-offloading studies use one operating condition, omit initialisation or use requester-only energy. The claim is deliberately bounded:

> The four representative studies examined here, two learning-based (Zhan et al.; Wei et al.) and two based on optimisation or game theory (Fan et al.; Wang et al.), do not consistently report three aspects of evaluation design, leaving uncertainty about how these choices affect the evidence: whether the sampled conditions represent the intended deployment envelope, whether evaluation windows or episodes preserve the preceding system state, and whether energy accounting includes remote communication and execution.

The matrix is consistent with this bounded claim:

- every study varies load or traffic parameters, but on synthetic mobility rather than on clock-time conditions of a calibrated trajectory;
- initial conditions are stated only partly, and none states whether a window inherits preceding traffic history or uses a warm-up period;
- energy is not reported in two studies, is limited to the requester in one and covers remote execution in one;
- neither learned-policy study states the number of independent training runs.

The Introduction therefore uses:

> "…the representative studies reviewed in Sect. 2 evaluate offloading policies through simulation."

It does not use stronger terms such as "all", "usually", "none" or "universally" unless the completed matrix supports them.

## Sources

Page numbers are journal pages. Where a property is not stated, the matrix records it as not identified in the inspected paper rather than assuming that it is absent from the implementation.

**Zhan, W., Luo, C., Wang, J., Wang, C., Min, G., Duan, H., Zhu, Q.: Deep-reinforcement-learning-based offloading scheduling for vehicular edge computing. IEEE Internet of Things Journal 7(6), 5449–5465 (2020). https://doi.org/10.1109/JIOT.2020.2978830**

- Method: p. 5449 (Sec. I), p. 5451 (Sec. II-A, one representative vehicle), p. 5458 (Sec. V-B, Algorithm 1).
- Service: p. 5451 (Sec. II-B: "We consider tasks without stringent latency demands or execution priority"); p. 5453 (Sec. II-E, Eqs. 10 and 13); Figs. 5–7, 9, 11–14.
- Energy: pp. 5452–5453 (Sec. II-D, Eqs. 5 and 9: "From VT's perspective, RE of tasks does not consume its energy"); Figs. 8 and 10.
- Simulation and conditions: p. 5459 (Sec. VI, Tables III–IV); p. 5461 (Sec. VI-B, latency weight 0.03–0.10); p. 5462 (Sec. VI-C, arrival rate 0.1–1.0).
- Initialisation: p. 5461 (Sec. VI-B: initial queue and position "randomly chosen, but are kept identical for all the algorithms"); p. 5462 (Sec. VI-C).
- Training runs: p. 5460 (Sec. VI-A, Fig. 4: networks "trained for 1800 epochs"); evaluation over 500 runs per weight (p. 5461) and 200 rounds (p. 5462).

**Wei, Z., Li, B., Zhang, R., Cheng, X., Yang, L.: Many-to-many task offloading in vehicular fog computing: a multi-agent deep reinforcement learning approach. IEEE Transactions on Mobile Computing 23(3), 2107–2122 (2024). https://doi.org/10.1109/TMC.2023.3250495**

- Method: p. 2107 (Abstract); pp. 2112–2114 (Sec. III); p. 2117 (Sec. IV-B, Algorithm 3).
- Service: p. 2119 (Sec. V-B: "Task completion ratio is defined as the percent of the tasks completed within their deadlines"); Figs. 8–10.
- Energy: no energy term in the latency model, utility, objective, reward or results (Eqs. 4–14, pp. 2110–2112; Eqs. 27–29, p. 2116); transmit power appears only in the rate equation (Eq. 5, p. 2110).
- Simulation and conditions: p. 2117 (Sec. V-A, Python environment); p. 2118 (Table I); pp. 2119–2121 (Figs. 8–12: 30/50/100 vehicles; lower, medium and higher load).
- Initialisation: p. 2110 (Eq. 1, entry speed); p. 2117 (Algorithm 3, "Reset environment").
- Training runs: p. 2118 (Sec. V-A, 3,000 training episodes; Fig. 5); p. 2119 (Fig. 7, "the same random seeds"; number not given).

**Fan, W., Su, Y., Liu, J., Li, S., Huang, W., Wu, F., Liu, Y.: Joint task offloading and resource allocation for vehicular edge computing based on V2I and V2V modes. IEEE Transactions on Intelligent Transportation Systems 24(4), 4277–4292 (2023). https://doi.org/10.1109/TITS.2022.3230430**

- Method: p. 4277 (Abstract, Generalized Benders Decomposition and Reformulation-Linearization); pp. 4283–4286 (Sec. IV, Algorithms 1–2).
- Service: p. 4282 (Sec. III-F, Eq. 19: cost "the sum of the total delays of vehicles"); Figs. 3–8.
- Energy: transmit power only as an SINR input (Eq. 6, p. 4281; Table II, p. 4287); no energy in the objective or results.
- Simulation and conditions: p. 4287 (Sec. V: "numerical-based simulations via the Matlab simulator"; Table II); pp. 4288–4289 (Sec. V-A to V-F, Figs. 3–8).
- Initialisation: p. 4280 (Sec. III-A: "we define x_i as its initial position").
- Repetitions: p. 4290 (Table III, solver iterations per figure; number of runs not stated).

**Wang, S., Peng, H., Li, H.: Hybrid V2V/V2I task offloading in vehicular edge computing: a double-layer Stackelberg game approach. IEEE Internet of Things Journal 12(15), 31026–31041 (2025). https://doi.org/10.1109/JIOT.2025.3571937**

- Method: p. 31032 (Sec. V-A); pp. 31034–31035 (Sec. V-C, Algorithms 1–3).
- Service: p. 31032 (Eq. 28i, constraint C6 on maximum tolerant delay); Figs. 3 and 9; Tables III–IV.
- Energy: p. 31031 (Sec. III-C, Eqs. 19–22: "the energy consumed to return the result can be ignored"); Figs. 8 and 14; Tables III–IV.
- Simulation and conditions: p. 31029 (Sec. III, Eqs. 1–2); pp. 31035–31036 (Sec. VI-A, Table II); pp. 31036–31039 (Figs. 3–14).
- Initialisation: pp. 31035–31036 (Sec. VI-A: "In each simulation run, vehicles are distributed randomly in the experimental area").
- Repetitions: p. 31036 (Sec. VI-A: "all data points are based on the average of over 5000 simulation runs").
