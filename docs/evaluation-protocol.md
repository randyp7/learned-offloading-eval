# Three-dimensional evaluation protocol

The paper evaluates learned operational components along three dimensions: operating-condition coverage, state continuity and accounting scope.

## 1. Operating-condition coverage

### Validity question

Does the evaluation represent materially different conditions within the component's intended deployment envelope?

### Required reporting

An evaluation should document:

- the intended deployment envelope;
- the conditions selected for testing;
- the variables used to distinguish those conditions;
- the rationale for their selection;
- conditions that are not represented;
- whether an aggregate combines heterogeneous conditions; and
- whether the selected variables are observational or experimentally controlled.

### Present operationalisation

The vehicular study uses:

| Condition | Clock time | Whole-period mean vehicles/frame |
|---|---|---:|
| Night | 00:00–04:00 | 4 |
| Morning peak | 08:00–09:00 | 375 |
| Evening peak | 16:00–18:00 | 459 |
| Late evening | 20:00–24:00 | 135 |
| Full-day aggregate | all hours | 274 |

The values are means over all non-empty frames of the period in the continuous 24-hour trajectory (`data/fcd/corridor_24h_seed42.csv`; non-empty frames are the frames that can enter an episode, see `warm-cold-protocol.md`). Only the night contains empty seconds (2,711 of 14,400). Averaged over all seconds of the period instead, the night mean is 3.4 and the full-day mean 265; the other periods are unchanged. These values differ from the warm-start means in the warm/cold experiment (378, 475, 169, 304 and 3.7 vehicles per frame for the morning peak, evening peak, late evening, full day and night). Those means average the sampled windows that were actually evaluated, not every frame of the period (`results/window-pairs.csv`).

The targeted periods span the observed numerical density range but do not experimentally isolate density. Speed, spatial configuration, connectivity, workload and eligible peer capacity may co-vary.

The repository additionally contains the morning-ramp robustness period (04:00–08:00, mean 220 vehicles per frame; `results/eval_main_04-08.csv`) and the full hourly density plot (`results/supplementary/fig_diurnal.pdf`).

## 2. State continuity

### Validity question

Does an evaluation window preserve the system history associated with the scenario it claims to represent?

### Required reporting

An evaluation should document:

- which state variables are preserved;
- which state variables are reset;
- whether a snapshot, continuous replay or warm-up is used;
- any warm-up convergence criterion;
- matching between alternative initialisation regimes;
- frame-selection and episode-termination rules; and
- whether the target scenario is an already-running condition or a deliberate restart/recovery condition.

### Present operationalisation

- Warm windows inherit the population, positions and speeds from the continuous 24-hour trajectory.
- Cold windows begin from an empty road at the same nominal start time.
- Each condition uses 24 seeded nominal start times; the evaluation draws 24 episodes from them with replacement, covering 15 distinct start times, and the warm and cold members replay the identical sequence.
- Both members are evaluated with the same policy checkpoints, the same evaluation seed and the same task-generation rule; the realised tasks differ because the vehicles present differ (`warm-cold-protocol.md`).
- A frame counts towards the 100-frame episode only when at least one vehicle is present.
- Warm/cold pairing identifiers and start times are released in `results/window-pairs.csv`.

Executor queues are empty at the start of every episode in both members; only the traffic state differs.

## 3. Accounting scope

### Validity question

Which resources, activities and service outcomes are included in the claimed value?

### Required reporting

An evaluation should document:

- the stakeholder or system question supported by each ledger;
- resources and activities included in each boundary;
- exclusions from the boundary;
- the service measure reported alongside resource use;
- the energy or cost model used;
- whether the result concerns operational, lifecycle, financial or carbon cost; and
- sensitivity to materially different model parameters where available.

### Present operationalisation

- Requester ledger: task-attributable requester computation and communication.
- Executor ledger: task-attributable computation, reception and result transmission at the selected peer or RSU.
- Modelled system ledger: requester and executor ledgers together, i.e. requester, executor and modelled communication energy.
- Service criterion: proportion of tasks completed by their deadline.

The equations are given in `radio-and-energy-model.md`. The ledgers do not include policy-training energy, embodied energy, idle or cooling energy, non-task-attributable infrastructure energy or carbon intensity. No sensitivity analysis over the energy parameters is included.

## Interpretation rule

The protocol does not prescribe a universally correct operating condition, initial state or accounting boundary. It requires those choices to be specified and aligned with the deployment claim being evaluated.
