# Matched warm/cold evaluation protocol

## Purpose

This experiment isolates the effect associated with preserving or resetting pre-window traffic history when evaluating a policy at the same nominal clock time.

## Pair construction

For each operating condition (`scripts/make_paired_fragments.py`):

1. Draw 24 nominal start times with seed `7000 + sum(ord(c) for c in period)`, uniformly within the period and at least 2,000 s before its end (`data/paired/starts_<period>.json`).
2. Construct the warm member from the first 100 non-empty frames of the continuous 24-hour trajectory at or after that nominal time.
3. Construct the cold member by starting SUMO on an empty road at the same nominal time (`--begin T`, SUMO seed 42), keeping the first 100 non-empty frames within at most 1,800 simulated seconds.
4. Apply the same frame-eligibility rule to both members.
5. Give both members the same fragment file name, `frag_t<T>.csv`, in `data/paired/warm_<period>/` and `data/paired/cold_<period>/`.
6. Evaluate every policy on both members with the same checkpoints and the same evaluation seed. The replay environment selects the fragment of each episode by name from its seeded episode generator, so both members replay the identical sequence of start times (`data/paired/eval_sequence_<period>.csv`).

## State preserved or reset

### Warm member

Preserves:

- vehicle population inherited from earlier traffic history;
- vehicle position;
- vehicle speed;
- traffic evolution from the continuous trajectory.

### Cold member

Resets:

- vehicles that entered before the nominal start time;
- the traffic population and spatial configuration inherited from prior time.

Vehicles entering after the nominal time are inserted according to the cold simulation's route schedule.

In both members the executor queues start empty and vehicle computing classes are drawn afresh in each episode.

## Episode definition

Each episode contains 100 eligible frames. A frame is eligible when at least one vehicle is present on the corridor. The rule is applied identically to warm and cold members. In the loaded periods every second contains a vehicle, so a member spans exactly 100 s. At night a member can span up to 477 s because empty seconds are skipped.

The first eligible frame of a cold member usually lies after the nominal time, because the empty road needs time for the first vehicle to enter: by 1 s (median) in the loaded periods and by 29 s (median) at night.

## Workload

Both members use the same task-generation rule, task-seed base and episode index. Task arrivals are drawn per vehicle and keyed on the trace time of each frame. The realised task streams therefore differ between members: their vehicles differ, and their frame timestamps are usually offset. The pairing controls the nominal clock time, the policy, the evaluation seed and the episode sequence. It does not control the workload, which follows from the traffic state that the experiment varies.

## Released files

- `results/window-pairs.csv`: one row per pair, giving period, nominal start time, fragment name, number of times evaluated, and mean vehicles per frame and time span of each member. It also serves as the warm and cold window identifier list.
- `results/paired_density/paired_density_frames.csv`: vehicles per frame by pair, member and frame position (1–100), with the trace time of each frame.
- `data/paired/starts_<period>.json`, `data/paired/eval_sequence_<period>.csv`, `data/paired/{warm,cold}_<period>/_stats.csv`.
- `configs/evaluation.yaml`.
- `scripts/make_paired_fragments.py` and `scripts/slurm/cold_fragments.sh`, which regenerate both members of every pair (the fragment files themselves are not included; see the README).
- Evaluation results: `results/eval_paired_{warm,cold}_<period>.csv` (warm-trained policies) and `results/eval_coldtrain_paired_{warm,cold}_<period>.csv` (cold-trained policies).

## Interpretation

The experiment compares two state trajectories associated with the same nominal start time. It does not assume that a cold-start window is inherently invalid. A cold start is appropriate for a restart or recovery scenario; without an adequate warm-up, it does not represent a condition that is already running at that clock time.
