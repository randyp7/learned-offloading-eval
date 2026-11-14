# Reference verification register

This register records how each reference cited in the paper was verified. For every reference it gives the authoritative bibliographic record, each claim the paper cites it for, a passage from the source supporting that claim, and the level of evidence on which the check rests.

The check was carried out in November 2026.

**Bibliographic records.** Each entry was compared with an authoritative record:

- the Crossref record and DOI resolution;
- the publisher or proceedings page (JMLR, NeurIPS, AAAI);
- the official 3GPP archive, for TR 36.814;
- the LNCS volume printed on the chapter, for the CAiSE chapters.

**Claim-to-source checks.** Each citation was read in its context in the paper and compared with the full text of the source, not only its title or abstract. Where a reference is cited more than once, each use was checked separately.

Every claim listed is fully supported: the source directly states or demonstrates it, as worded in the paper.

Quotations are taken from the full texts. Section, equation and figure numbers refer to the cited publication.

## Vehicular task offloading

**Meneguette, R., De Grande, R., Ueyama, J., Rocha Filho, G.P., Madeira, E.: Vehicular Edge Computing: Architecture, Resource Management, Security, and Challenges. ACM Computing Surveys 55(1), 1–46 (2023).** https://doi.org/10.1145/3485129

| Where cited | Claim | Support | Evidence |
|---|---|---|---|
| Sect. 1 | A task may run on board, at a roadside unit or on a neighbouring vehicle with spare capacity | Full | Sect. 7.1.1: VEC "can use the idle computing resources embedded in each vehicle … can be allocated both in vehicles and in servers"; Sect. 1: the "collaborative participation of neighbouring vehicles sets VEC apart" |
| Sect. 2 | Offloading assigns computation among onboard, peer and roadside resources under mobility, capacity and latency constraints | Full | Sect. 7.4: resource management must avoid failures from "not meeting QoS requirements or high vehicle mobility" |

**Zhang, S., Yi, N., Ma, Y.: A Survey of Computation Offloading With Task Types. IEEE Transactions on Intelligent Transportation Systems 25(8), 8313–8333 (2024).** https://doi.org/10.1109/TITS.2024.3410896

| Where cited | Claim | Support | Evidence |
|---|---|---|---|
| Sect. 1 | Execution on board or on remote nodes, including other vehicles | Full | Covers local versus remote execution; "UE computing collaboration … often occurs in vehicle scenario" |
| Sect. 1 | Deep reinforcement learning policies are commonly used for task-allocation decisions | Full | "Reinforcement learning (RL) is commonly used in computation offloading … In addition, there are Q-learning, deep reinforcement learning (DRL) …" |
| Sect. 2 | Mobility, capacity and latency constraints | Full | "Mobility is one of the important features that cannot be ignored in computation offloading", especially for vehicles |

**Wei, Z., Li, B., Zhang, R., Cheng, X., Yang, L.: Many-to-Many Task Offloading in Vehicular Fog Computing: A Multi-Agent Deep Reinforcement Learning Approach. IEEE Transactions on Mobile Computing 23(3), 2107–2122 (2024).** https://doi.org/10.1109/TMC.2023.3250495

| Where cited | Claim | Support | Evidence |
|---|---|---|---|
| Sect. 1 | Example of DRL used for task allocation | Full | Multi-agent DRL (MA-GAC) for many-to-many offloading |
| Sect. 1 | Representative evaluations report latency or deadline satisfaction | Full | "Task completion ratio is defined as the percent of the tasks completed within their deadlines"; no energy measure |
| Sect. 2 | Multi-agent reinforcement learning approach; one of the two learning-based representative studies | Full | MA-GAC across V2V, RSU, inter-RSU and cloud tiers, plus local execution; evaluated in the authors' Python simulation |

**Zhan, W., Luo, C., Wang, J., Wang, C., Min, G., Duan, H., Zhu, Q.: Deep-Reinforcement-Learning-Based Offloading Scheduling for Vehicular Edge Computing. IEEE Internet of Things Journal 7(6), 5449–5465 (2020).** https://doi.org/10.1109/JIOT.2020.2978830

| Where cited | Claim | Support | Evidence |
|---|---|---|---|
| Sect. 1 | Representative evaluations report time-based measures and, in some cases, energy | Full | Average latency and a weighted latency–energy cost; energy is "only associated with the task local computation and input data transmission" (requester side) |
| Sect. 2 | Example of a requester-only energy ledger | Full | Sect. II-D: "From VT's perspective, RE of tasks does not consume its energy" (VT = the requesting vehicle; RE = remote execution) |
| Sect. 2 | Single-agent reinforcement learning approach; one of the two learning-based representative studies | Full | PPO for one representative vehicle, V2I only |

**Fan, W., Su, Y., Liu, J., Li, S., Huang, W., Wu, F., Liu, Y.: Joint Task Offloading and Resource Allocation for Vehicular Edge Computing Based on V2I and V2V Modes. IEEE Transactions on Intelligent Transportation Systems 24(4), 4277–4292 (2023).** https://doi.org/10.1109/TITS.2022.3230430

| Where cited | Claim | Support | Evidence |
|---|---|---|---|
| Sect. 1 | Representative evaluations report time-based service measures | Full | Minimises "the total task processing delay of all the vehicles"; delay tolerance is a constraint; no energy measure |
| Sect. 2 | Optimisation formulation; one of the two non-learning representative studies | Full | Generalized Benders Decomposition plus a heuristic, with local, V2I and V2V execution; MATLAB numerical simulation |

**Wang, S., Peng, H., Li, H.: Hybrid V2V/V2I Task Offloading in Vehicular Edge Computing: A Double-Layer Stackelberg Game Approach. IEEE Internet of Things Journal 12(15), 31026–31041 (2025).** https://doi.org/10.1109/JIOT.2025.3571937

| Where cited | Claim | Support | Evidence |
|---|---|---|---|
| Sect. 1 | Representative evaluations report time-based measures and, in some cases, energy | Full | Completion delay with the deadline as constraint C6; Eqs. 19–22 and Figs. 8 and 14 report "Total Energy Consumption of System" |
| Sect. 2 | Example of a broader operational (system-wide) energy ledger | Full | System energy covers requester upload and peer and RSU computation (Sect. III-C, Eqs. 19–22) |
| Sect. 2 | Game-theoretic formulation; one of the two non-learning representative studies | Full | "double-layer Stackelberg game … in the hybrid V2V/V2I offloading scenario" |

The detailed comparison of these four studies is in [`related-work-evidence-matrix.md`](related-work-evidence-matrix.md).

## Information systems engineering and evaluation

**Pastor, O., Krogstie, J., Nurcan, S., Alter, S., Eder, J., Franch, X., Snoeck, M., Ralyté, J.: Articulating Information Systems Engineering Research Contributions. In: Retrospective and Prospective Views on Information Systems Engineering, 15–31. Springer (2026).** https://doi.org/10.1007/978-3-032-19578-4_2

| Where cited | Claim | Support | Evidence |
|---|---|---|---|
| Sect. 1, Sect. 2 | The contribution lies in the methodological dimension of information systems engineering: approaches for evaluating an information-system artefact | Full | Lists "Methodological approaches to represent, to evaluate, or to make the artifact evolve" as a dimension of an ISE contribution |

**Alter, S.: Can Information System Engineering Make Cyber-Human Systems Smarter? CAiSE 2023, LNCS 13901, 16–32.** https://doi.org/10.1007/978-3-031-34560-9_2

| Where cited | Claim | Support | Evidence |
|---|---|---|---|
| Sect. 2 | Following Alter's definition of a digital agent, the learned policy is treated as a digital agent within a wider work system | Full | Sect. 2.4: "A digital agent is a totally automated IS that is an algorithmic agent serving a specific purpose." The paper uses this definition only; Alter's own analysis sets aside agents embedded in devices such as network controllers |

**Wais, B., Rinderle-Ma, S.: Towards a Comprehensive Evaluation of Decision Rules and Decision Mining Algorithms Beyond Accuracy. CAiSE 2024, LNCS 14663, 403–419.** https://doi.org/10.1007/978-3-031-61057-8_24

| Where cited | Claim | Support | Evidence |
|---|---|---|---|
| Sect. 2 | Predictive performance does not by itself establish the value of a decision rule | Full | "performance alone cannot reflect if the discovered decision rules produce value" |

**Ghasemi, A., Kulms, L., Lichtenstein, T., Weske, M.: Agent-Based Simulation of Flexible Process Interactions. CAiSE 2026, LNCS 16559, 75–93.** https://doi.org/10.1007/978-3-032-28117-3_5

| Where cited | Claim | Support | Evidence |
|---|---|---|---|
| Sect. 2 | Simulation supports the exploration of operational alternatives before implementation | Full | Agent-based simulation used to compare alternative process interactions |
| Sect. 2 | Simulation conclusions rest on the assumed input parameters and resource models | Full | The authors state that their conclusions rest on assumed input parameters and simplified resource models, and call for data-driven parameterisation, sufficient replications and statistical analysis |

**Schnellmann, M., Proper, H.A.: From Gut Feeling to Data-Driven Decisions: Exploring Digital Twin Supported Decision-Making for Real Estate Management. CAiSE 2026, LNCS 16558, 3–21.** https://doi.org/10.1007/978-3-032-28110-4_1

| Where cited | Claim | Support | Evidence |
|---|---|---|---|
| Sect. 2 | Practitioners expect digital twins to support what-if comparison of alternative interventions | Full | Interview study of 13 practitioners: decision support should combine "scenario and simulation capabilities for comparing alternative intervention paths"; cited for practitioners' expectations, not for demonstrated use |

**Stein, H., Janzen, S., Agnes, C.K., Ninh, D.T., Maass, W.: Towards Decision Support Systems for Cost-Effective and Energy-Efficient AI Operations in Data Centers. CAiSE 2026, LNCS 16558, 394–413.** https://doi.org/10.1007/978-3-032-28110-4_22

| Where cited | Claim | Support | Evidence |
|---|---|---|---|
| Sect. 2 | For AI operations in data centres, Stein et al. identify a lack of end-to-end transparency in performance–energy trade-offs and propose energy modelling across system layers | Full | Problem statement and design principles of the paper |
| Sect. 7 | Design principles for energy decision support call for multi-layer energy modelling and for making the assumptions behind estimated energy outcomes explicit | Full | Design principles of the paper |

## Reinforcement learning

**Mnih, V., Kavukcuoglu, K., Silver, D., et al.: Human-level control through deep reinforcement learning. Nature 518(7540), 529–533 (2015).** https://doi.org/10.1038/nature14236

| Where cited | Claim | Support | Evidence |
|---|---|---|---|
| Sect. 1, Sect. 4 | DQN is a deep reinforcement-learning method for selecting among discrete actions | Full | The network has "a separate output unit for each possible action" |

**Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., Dormann, N.: Stable-Baselines3: Reliable Reinforcement Learning Implementations. Journal of Machine Learning Research 22(268), 1–8 (2021).** https://jmlr.org/papers/v22/20-1364.html

| Where cited | Claim | Support | Evidence |
|---|---|---|---|
| Sect. 4 | The DQN implementation used | Full | Primary reference for the library; version 2.3.2 is pinned in this package |

**Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., Meger, D.: Deep Reinforcement Learning That Matters. AAAI-18, 3207–3214 (2018).** https://doi.org/10.1609/aaai.v32i1.11694

| Where cited | Claim | Support | Evidence |
|---|---|---|---|
| Sect. 2 | DRL results are sensitive to variation among training runs and to how that variation is analysed | Full | "the variance between runs is enough to create statistically different distributions just from varying random seeds"; calls for proper significance analysis |

**Agarwal, R., Schwarzer, M., Castro, P.S., Courville, A., Bellemare, M.G.: Deep Reinforcement Learning at the Edge of the Statistical Precipice. Advances in Neural Information Processing Systems 34, 29304–29320 (2021).** https://arxiv.org/abs/2108.13264

| Where cited | Claim | Support | Evidence |
|---|---|---|---|
| Sect. 2 | As above | Full | Published comparisons ignore "the statistical uncertainty implied by the use of a finite number of training runs"; 3–10 runs are prevalent |

## Traffic scenario and simulation

**Schrab, K., Protzmann, R., Radusch, I.: A Large-Scale Traffic Scenario of Berlin for Evaluating Smart Mobility Applications. In: Smart Energy for Smart Transport, 276–287. Springer (2023).** https://doi.org/10.1007/978-3-031-23721-8_24

| Where cited | Claim | Support | Evidence |
|---|---|---|---|
| Sect. 4 | The trajectory is cut from the open Berlin SUMO scenario (BeST) | Full | The network and corridor routes in [`sumo/`](../sumo/) are cut from `best-scenario-v2` (see the README) |

**Ziemke, D., Kaddoura, I., Nagel, K.: The MATSim Open Berlin Scenario: A multimodal agent-based transport simulation scenario based on synthetic demand modeling and open data. Procedia Computer Science 151, 870–877 (2019).** https://doi.org/10.1016/j.procs.2019.04.120

| Where cited | Claim | Support | Evidence |
|---|---|---|---|
| Sect. 4, abstract, Table 1 | The demand underlying BeST is calibrated against observed traffic counts | Full | Sect. 2.3: calibration with CaDyTS against "8,304 hourly count values for 346 count stations", 96 of them on motorways |

**Lopez, P.A., Behrisch, M., Bieker-Walz, L., Erdmann, J., Flötteröd, Y.-P., Hilbrich, R., Lücken, L., Rummel, J., Wagner, P., Wiessner, E.: Microscopic Traffic Simulation using SUMO. In: 21st International Conference on Intelligent Transportation Systems (ITSC), 2575–2582 (2018).** https://doi.org/10.1109/ITSC.2018.8569938

| Where cited | Claim | Support | Evidence |
|---|---|---|---|
| Sect. 4 | The trajectory is generated with SUMO 1.27 | Full | The reference the SUMO project asks users to cite; the version is pinned in `requirements.txt` |

**3GPP TR 36.814 V9.2.0: Evolved Universal Terrestrial Radio Access (E-UTRA); Further advancements for E-UTRA physical layer aspects (2017).**

| Where cited | Claim | Support | Evidence |
|---|---|---|---|
| Sect. 4 | The deterministic urban micro (UMi) non-line-of-sight path-loss model | Full | Table B.1.2.1-1 (UMi NLOS, hexagonal layout) gives the formula implemented in `net/nrv2x_lite.py`. The SNR-to-spectral-efficiency table is part of the implementation and is documented in [`radio-and-energy-model.md`](radio-and-energy-model.md) |

## Previous study

**Johnston, L.: Quantifying the Benefit of V2V Cooperative Offloading in Vehicular Edge Computing via Deep Reinforcement Learning. BSc dissertation, Department of Computer Science, The University of Manchester (2026).**

| Where cited | Claim | Support | Evidence |
|---|---|---|---|
| Sect. 4 | Class-weighted deadline reward with weights 2.0 / 1.0 / 0.5 | Full | The dissertation's reward formulation uses the same reward and weights |
| Sect. 4 | Task, compute and radio configurations are inherited from the shared code of the previous study | Full | See [`parameter-provenance.md`](parameter-provenance.md) for the status of each parameter |
