"""
Canonical scenario definition for the Berlin V2V study (UKAIS 2027 re-run).

WHY THIS FILE EXISTS
--------------------
The original study results (Tables 6.1/6.2) were produced by manual `train_model.py`
invocations whose exact flags were NOT recorded in the repo (no run scripts, no
model registry, no trained checkpoints were committed). This module reconstructs
that configuration from the report + code and pins EVERY parameter explicitly, so
the 5-seed re-run is fully reproducible and self-documenting.

RECONSTRUCTED CONFIG (evidence in comments):
  - state_space_version = 6   -> 30 features (matches report Table 4.1 exactly)
  - task_profile_version = 5   -> cpu_cycles 600M/1200M/3600M and data 300KB/800KB/3MB
                                  match report §3.4 exactly
  - rsu_profile_version  = 2   -> 16 cores @ 2.5 GHz = 40 GHz (report §3.3.2)
  - episode_length_steps = 100 -> "100s episodes" (report §5.3.1), step_length_s=1.0

RESOLVED / OPEN parameter decisions:
  (1) Autonomous deadline: RESOLVED = 550 ms (decision, 2026-07). This is what
      task-profile v5 already uses, so no code change is needed. The report's
      Table 3.3 value of 150 ms is a documentation error — the PAPER TEXT must
      report 550 ms for the autonomous task category.
  (2) Task arrival rate: RESOLVED = 0.5 tasks/s/veh (decision, 2026-07),
      matching report Table 3.3. LAMBDA below is 0.5 (config.py's 0.3 default is
      overridden by the scenario / --lambda-per-veh 0.5 in the SLURM scripts).
  (3) Fleet compute (KBA-based): the runs use config.py VEH_CPU_CLASSES —
      KBA market shares match report Table 3.2 EXACTLY (30/50/15/5), but the
      per-category hardware is a variant: cores fixed at 2/4/6/8 (report:
      1-4 / 2-4 / 6-8 / 10-12) and clocks 0.8-1.5 / 1.5-2.5 / 2.0-3.0 /
      2.5-5.0 GHz (report caps at 2.3). The PAPER TEXT must quote the CODE
      values, not copy the original study's Table 3.2.
  (4) Reward: RESOLVED = --reward-type deadline_only (decision, 2026-07).
      This reproduces report §4.4 exactly (binary SLA ±1 x category weight
      2.0/1.0/0.5). The code DEFAULT (reward_type=None) is a composite with
      energy weights 0.15/0.05 — another report/code divergence; models
      trained with the default are archived as the "energy-aware reward"
      ablation arm. deadline_only also makes energy a measured-not-rewarded
      quantity, enabling the Green IS by-product analysis.

Parameters are pinned as named constants so each choice is explicit. The
deadline (550 ms), lambda (0.5) and fleet table (code values) are settled.

Differences from the original study (report-vs-code provenance):
  - "16-18 peak" pre-gen file actually contained 13:00-15:00 (mislabeled
    sumocfg); we regenerated a true 16-18 trace.
  - Exp1 geographic scope: report says full Berlin; re-run uses the A100
    corridor at warm density (see EXPERIMENTS note).
  - Cold-start: original episodes launched SUMO at the window start (empty
    network, density far below the labelled hour); re-run replays warm
    continuous traffic. Report does not mention this artifact.
  - Fleet spec table: report Table 3.2 vs code VEH_CPU_CLASSES (above).
"""

# ---- Locked structural config (high confidence) ----
STATE_SPACE_VERSION = 6
TASK_PROFILE_VERSION = 5
RSU_PROFILE_VERSION = 2
EPISODE_LENGTH_STEPS = 100

# Autonomous-task deadline is 550 ms (task-profile v5, decision 2026-07) — no
# constant here; it lives in TASK_PROFILES[5]. Report text must say 550 ms.

# ---- Task arrival rate (original report 0.5; config.py default 0.3) ----
LAMBDA_PER_VEH_PER_SEC = 0.5   # report Table 3.3 (config.py default is 0.3)

# ---- Seed conventions (aligned with the Paper 2A campaign) ----
TRAINING_SEEDS = [100, 101, 102, 103, 104]   # 5 training seeds carry the DRL variance
EVAL_SEED = 7000                              # single shared, seeded evaluation set
N_EVAL_EPISODES = 50                          # matches the original study's 50-episode eval

# ---- Scenario data (Berlin) ----
# Experiment 1 (general effectiveness): A100 corridor, full 24h day, random
#   continuous 100s windows (single-file replay). This keeps the original study's
#   Exp1 CONCEPT (uniformly sampled episode windows across a 24h
#   detector-calibrated day) while removing two artifacts of the original:
#   (a) cold-start — original episodes launched SUMO fresh at the window start,
#       so the network began EMPTY (density a small fraction of steady state);
#       replaying the continuous 24h trace gives warm, realistic density
#       (diurnal profile verified: ~4 veh/frame at night -> 364 morning ->
#       459 evening peak -> 85 late evening);
#   (b) 20s windows — original training episodes were 20 SUMO steps; we use
#       100 (matching the original study's own evaluation episode length).
#   Geographic scope note: original study Exp1 claimed the FULL Berlin network;
#   we deliberately train on the A100 corridor (where the 8 RSUs are, at
#   warm density).
# Experiment 2 (peak congestion): A100 motorway, 08-09 (as the original study).
EXPERIMENTS = {
    "general_24h": dict(date="2023-10-18", timeframe=None,
                        sumocfg="sumo/berlin_2023-10-18.sumocfg"),
    "a100_peak":   dict(date="2023-10-18", timeframe=(8, 9),
                        sumocfg="sumo/berlin_a100_06-12.sumocfg"),
}

# FCD-replay position data (see slurm/fcd_generate.sh + slurm/README.md).
# Pass these as pos_csv to create_offloading_env to skip live SUMO. Not read
# automatically — the SLURM scripts pass them explicitly.
# Regenerate the 24h trace with:
#   fcd_generate.sh sumo/berlin_a100_16-18.sumocfg 0 86400 42 <out>
FCD_DATA = {
    "general_24h": "data/fcd/corridor_24h_seed42.csv",  # continuous 24h corridor day
    "a100_peak":   "data/fcd/a100_peak.csv",            # 08-09 peak hour
}


def build_scenario(create_scenario_config, *, experiment, seed, no_v2v=False,
                   name=None):
    """
    Return a scenario config for the Berlin V2V study with all study parameters
    pinned. `create_scenario_config` is passed in to avoid a hard import cycle.

    Args:
        experiment: key in EXPERIMENTS ("general_24h" | "a100_peak")
        seed: training seed (100..104) or EVAL_SEED for evaluation
        no_v2v: True for the RSU-Local (V2V-disabled) DRL variant
        name: optional label
    """
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment!r}; valid: {list(EXPERIMENTS)}")
    exp = EXPERIMENTS[experiment]
    return create_scenario_config(
        name=name or f"berlin_{experiment}_seed{seed}",
        seed=seed,
        date=exp["date"],
        timeframe=exp["timeframe"],
        episode_length_steps=EPISODE_LENGTH_STEPS,
        state_space_version=STATE_SPACE_VERSION,
        task_profile_version=TASK_PROFILE_VERSION,
        rsu_profile_version=RSU_PROFILE_VERSION,
        lambda_per_veh_per_sec=LAMBDA_PER_VEH_PER_SEC,
        v2v_range_m=0.0 if no_v2v else None,
    )
