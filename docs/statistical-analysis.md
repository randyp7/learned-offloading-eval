# Statistical analysis

All tests are computed by `analysis/make_tables.py` and written to `analysis/numbers.json`.

## Units of analysis

The unit of learned-policy replication is one independently trained policy instance. Each variant is trained five times (seeds 100–104).

Evaluation episodes are repeated observations for a trained instance; they are not treated as independent policy-training replicates.

## Aggregation order

For each trained instance:

1. evaluate all shared episodes for the relevant condition;
2. pool every task of those episodes into one condition-level value (deadline satisfaction = tasks meeting their deadline / all tasks; energy = mean per task);
3. summarise the five trained-instance values using the mean and the sample standard deviation.

## Matched comparisons

DQN-Full and DQN-NoV2V, and DQN-Full and DQN-Energy, are trained with the same five seeds. For a given seed the variants share:

- the initial network weights (same seed, same network shape; checked directly with Stable-Baselines3 2.3.2);
- the sequence of training-episode windows (episode generator seeded with the training seed + 999);
- the task-generation rule and seed base;
- the evaluation episodes and task streams.

Seed identifiers therefore define a common-random-number match, and paired seed-level *t*-tests are used for these comparisons (`v2v_p_<period>` and `rw_<metric>_<period>.p` in `numbers.json`).

The match is partial. Actions, and with them the resulting training trajectories, diverge once interaction begins. Vehicle computing classes are drawn from a seeded generator in order of first use, and candidate peers are drawn only when V2V is enabled, so a given vehicle can receive a different computing class under DQN-Full and DQN-NoV2V. For this reason, Welch's unpaired test is also reported as a sensitivity analysis for the matched comparisons (`v2v_welch_p_<period>` and `rw_<metric>_<period>.welch_p`). For DQN-Full versus DQN-NoV2V it gives the same conclusion as the paired test in every period: p < 0.001 under the loaded conditions and the full day, p = 0.89 at night.

The warm/cold evaluation compares the same trained checkpoint on the two members of each window pair. These comparisons are paired by checkpoint (`cold_<period>.p_full` in `numbers.json`).

## Independent-protocol comparisons

Policies trained under different protocols, warm-start and cold-start training, have no natural seed-level pairing: the episode sources and lengths differ, so equal seed numbers do not produce matched training. They are compared using Welch's test (`coldtrain_<period>.welch_p_<warm|cold>_eval` and `coldtrain_loaded_*.welch_p`). `numbers.json` also lists seed-paired p-values for this comparison (`p_train_*_eval`); the paper's inference for this comparison uses the Welch tests.

## Deterministic comparators

Greedy, Local and Random are evaluated on the shared evaluation set. Greedy and Local are deterministic. Random draws its choices from a generator seeded with the evaluation seed. These comparators are reported as single values rather than being treated as five trained-policy observations.

## Interpretation

With five trained instances per policy, statistical inference is necessarily limited. The paper therefore reports and interprets:

- absolute effect sizes;
- between-run standard deviations;
- directional consistency;
- seed-level overlap or separation where relevant; and
- p-values as supporting rather than decisive evidence.

Statistical significance is not interpreted as operational importance.
