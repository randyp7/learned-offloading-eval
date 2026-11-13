# Training, evaluation and aggregation

## Training

Each policy is trained for 1.5 million task-level decisions (`--timesteps 1500000`). One decision assigns one generated task to an executor class. All tasks generated in a one-second traffic frame are decided one after another before the environment advances to the next frame. Their candidates are evaluated when the frame begins (`scheduling-and-comparators.md`).

The warm-start arms (DQN-Full, DQN-NoV2V and DQN-Energy) train on 100-frame episodes cut from the continuous 24-hour trajectory (`scripts/slurm/train_berlin.sh`). The replay environment counts frames as the timesteps present in the trajectory, which are the timesteps with at least one vehicle. Each episode starts at a frame drawn uniformly, with replacement, from all frames that leave room for a complete episode. The resulting training distribution follows the temporal frequency of traffic conditions rather than balancing operating periods. The first frame of an episode initialises the vehicle positions; tasks are generated on the remaining 99 frames.

The cold-start training arm trains on 20-frame episodes. Each episode replays one of the cold-start fragments (`scripts/slurm/train_coldtrain.sh`, `configs/cold_training.yaml`).

Training seeds are 100, 101, 102, 103 and 104. Each seed sets the DQN initialisation and exploration, the episode start sampling (environment seed + 999) and the vehicle computing-class assignments.

The final training checkpoint is evaluated. Intermediate checkpoints, written every 50,000 steps, are not used, and evaluation data do not influence checkpoint selection.

## Evaluation

Exploration is disabled: the DQN selects the action with the highest estimated Q-value (`deterministic=True`).

The period and full-day evaluations use 20 sampled 100-frame episodes per condition, each drawn from the trajectory slice of that period. The warm/cold experiment uses 24 episodes per condition. Each episode replays one of 24 seeded start times per period, drawn with replacement, which yields 15 distinct start times per period. The warm and the cold arm replay the identical sequence of start times.

Evaluation randomisation is based on seed 7000 and controls:

- episode start-time selection (generator seed 7999);
- onboard computing-class assignment;
- Random-policy actions.

The paired start times are drawn separately by `make_paired_fragments.py` (seed 7000 plus a per-period offset).

Vehicle positions are taken from the recorded trajectory. Task streams are generated deterministically from the episode index and trace time (`experimental-configuration.md`). Communication, queue scheduling and learned-policy action selection are deterministic. None of these draws depends on the policy, so every policy and every training seed is evaluated on the same episodes with the same tasks. Vehicle computing classes are drawn in order of first use, and candidate peers are drawn only when V2V is enabled. Assignments are therefore identical for all policies with V2V enabled (DQN-Full, DQN-Energy, Greedy, Local, Random), but a given vehicle can receive a different class under DQN-NoV2V (`statistical-analysis.md`).

## Aggregation

For a learned policy:

1. metrics are pooled over every task of the shared evaluation episodes within one training seed (deadline satisfaction = tasks meeting their deadline / all tasks; energy = mean over all tasks);
2. this gives one value per training seed;
3. the five seed-level values are summarised by their mean and sample standard deviation.

Greedy, Local and Random are deterministic given the evaluation set and are reported as single values on the same episodes. Operating conditions are not treated as independent policy-training replicates.

## Released matching information

The repository includes:

- the seeded start times of the paired windows (`data/paired/starts_<period>.json`) and the 24 start times actually evaluated in each period (`data/paired/eval_sequence_<period>.csv`);
- the warm/cold pairing: the warm and cold fragments of a pair share the file name `frag_t<start>.csv`; `results/window-pairs.csv` lists every pair with its density, and `results/paired_density/paired_density_frames.csv` gives vehicles per frame by pair and frame position (`warm-cold-protocol.md`);
- the training seeds (100–104), the evaluation seed (7000) and the task-seed base (12345);
- the final policy checkpoints, named by arm, variant and training seed (`checkpoints/berlin_<arm>_<variant>_seed<seed>/model.zip`);
- the scripts that regenerate the evaluation windows (`scripts/make_paired_fragments.py`), the evaluations (`scripts/evaluate_models.py`, `scripts/slurm/eval_window.sh`) and the tables (`analysis/make_tables.py`).

The episode start times of the period and full-day evaluations, and the task streams, are not stored as files. Running the evaluation with the seeds above regenerates them.
