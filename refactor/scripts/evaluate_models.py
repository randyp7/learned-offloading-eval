#!/usr/bin/env python3
"""
Reproducible multi-model evaluation harness for the Berlin V2V study.

Evaluates any mix of trained DRL checkpoints and non-learning baselines on ONE
shared, seeded evaluation set (EVAL_SEED, N_EVAL_EPISODES) and reports deadline
satisfaction with statistics OVER TRAINING SEEDS — the correct inferential unit
for DRL ("5 training seeds + a single shared,
seeded 50-episode evaluation").

Why this exists: the original study reported t-tests over 50 episodes of a single
trained model (pseudo-replication). Here, each learned method contributes ONE
number per training seed (aggregated over the 50 shared episodes); mean ± SD is
taken across the 5 seeds. Baselines are deterministic given the eval set and are
reported as single reference values.

Metric source: env writes step_summary.csv (one row per sim-tick) with per-tick
deadline_met / deadline_miss counts. Episode-aggregated, task-weighted:
    deadline_satisfaction = sum(deadline_met) / sum(deadline_met + deadline_miss)

Usage (from refactor/):
    python scripts/evaluate_models.py --experiment general_24h \
        --v2v-models  results/.../v2v_seed100/model.zip ... (5 paths) \
        --local-models results/.../local_seed100/model.zip ... (5 paths) \
        --baselines greedy random local \
        --out-csv results/eval_general_24h.csv

All 5 seeds of a method + all baselines are evaluated on the identical episode
sequence, so differences are attributable to the policy, not the episodes.
"""
import os
# Single-thread BLAS/OMP: the policy net is tiny; multi-threading it is much
# slower on many-core nodes (see train_model.py). Set before torch import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import sys
import csv
from pathlib import Path

import numpy as np
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

REFACTOR_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = REFACTOR_ROOT.parent
sys.path.insert(0, str(REFACTOR_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from training_config.config_loader import create_scenario_config
from training_config.berlin_v2v_scenario import (
    build_scenario, EVAL_SEED, N_EVAL_EPISODES, TRAINING_SEEDS,
)
from environment.env_factory import create_offloading_env


# ----------------------------------------------------------------------------
# Baseline policies (evaluated through the SAME wrapper as the DRL agent, so the
# episode dynamics and logging are identical). Each returns an action in {0,1,2}
# = {local, v2v, rsu}.
# ----------------------------------------------------------------------------
def greedy_action(uenv):
    """Minimum-estimated-delay candidate — matches the core's internal greedy
    (min over candidate delay_s). Reproduces the original study's Greedy."""
    cur = uenv._current
    if cur is None or cur[1] is None:
        return 0
    cands = cur[1]
    best_i, best_delay = 0, float(cands["local"]["delay_s"])
    for i, key in ((1, "v2v"), (2, "rsu")):
        c = cands.get(key)
        if c is not None:
            d = float(c.get("delay_s", np.inf))
            if d < best_delay:
                best_delay, best_i = d, i
    return best_i


def random_action(uenv, rng):
    """Uniform over VALID actions (respects the action mask)."""
    cur = uenv._current
    if cur is None:
        return 0
    mask = cur[3]
    valid = [i for i in range(3) if mask[i] == 1]
    return int(rng.choice(valid)) if valid else 0


def local_action(uenv):
    """Always compute locally."""
    return 0


# ----------------------------------------------------------------------------
# Metric extraction
# ----------------------------------------------------------------------------
def read_step_summary(core):
    """Aggregate step_summary.csv over the whole run (all shared episodes)."""
    path = Path(core.log_dir) / "step_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No step_summary.csv at {path}. Did the env write any ticks? "
            "(needs the _write_step_summary_full patch in the core env.)"
        )
    met = miss = 0
    tasks_tot = 0
    delay_weighted = 0.0
    e_user_weighted = 0.0
    e_sys_weighted = 0.0
    dec = {"local": 0, "v2v": 0, "rsu": 0}
    with open(path) as f:
        for r in csv.DictReader(f):
            n = int(float(r["tasks"] or 0))
            if n == 0:
                continue
            met += int(float(r["deadline_met"] or 0))
            miss += int(float(r["deadline_miss"] or 0))
            tasks_tot += n
            delay_weighted += float(r["mean_delay_s"] or 0.0) * n
            e_user_weighted += float(r.get("mean_user_energy_j") or 0.0) * n
            e_sys_weighted += float(r.get("mean_system_energy_j") or 0.0) * n
            dec["local"] += int(float(r["dec_local"] or 0))
            dec["v2v"] += int(float(r["dec_v2v"] or 0))
            dec["rsu"] += int(float(r["dec_rsu"] or 0))
    denom = met + miss
    return {
        "deadline_satisfaction": (met / denom) if denom else 0.0,
        "mean_delay_s": (delay_weighted / tasks_tot) if tasks_tot else 0.0,
        # Task-weighted mean energy per task (J). This env's energy model is
        # simpler than Paper 2A's tier-TDP model — report RELATIVE savings
        # between policies, not absolute J comparability across studies.
        "energy_user_j_per_task": (e_user_weighted / tasks_tot) if tasks_tot else 0.0,
        "energy_system_j_per_task": (e_sys_weighted / tasks_tot) if tasks_tot else 0.0,
        "n_tasks": tasks_tot,
        "frac_local": (dec["local"] / tasks_tot) if tasks_tot else 0.0,
        "frac_v2v": (dec["v2v"] / tasks_tot) if tasks_tot else 0.0,
        "frac_rsu": (dec["rsu"] / tasks_tot) if tasks_tot else 0.0,
    }


# ----------------------------------------------------------------------------
# Run one model/policy over the shared eval set
# ----------------------------------------------------------------------------
def evaluate_one(experiment, *, policy, model_path=None, no_v2v=False, seed=EVAL_SEED,
                 pos_csv=None, n_episodes=N_EVAL_EPISODES):
    """Run N_EVAL_EPISODES on the shared seeded eval set and return metrics.

    policy: "drl" | "greedy" | "random" | "local"
    model_path: required if policy == "drl"
    no_v2v: True disables V2V (for the RSU-Local DRL variant / matching its training)
    pos_csv: if set, replay this FCD-derived position CSV instead of running live
             SUMO (much faster; same shared-seed episode sampling).
    """
    scenario = build_scenario(create_scenario_config, experiment=experiment,
                              seed=seed, no_v2v=no_v2v,
                              name=f"eval_{experiment}_{policy}")
    venv = create_offloading_env(scenario, pos_csv=pos_csv)
    uenv = venv.envs[0].unwrapped          # OffloadDecisionEnv
    core = uenv.e                          # CoreEnv (has .log_dir, writes step_summary.csv)

    model = None
    if policy == "drl":
        from stable_baselines3 import DQN
        model = DQN.load(model_path)
    rng = np.random.default_rng(seed)      # seeded RNG for the random baseline

    obs = venv.reset()
    episodes = 0
    while episodes < n_episodes:
        if policy == "drl":
            action, _ = model.predict(obs, deterministic=True)
            a = int(action[0])
        elif policy == "greedy":
            a = greedy_action(uenv)
        elif policy == "random":
            a = random_action(uenv, rng)
        elif policy == "local":
            a = local_action(uenv)
        else:
            raise ValueError(f"unknown policy {policy!r}")
        obs, _, dones, _ = venv.step([a])
        if dones[0]:
            episodes += 1

    metrics = read_step_summary(core)
    venv.close()
    return metrics


# ----------------------------------------------------------------------------
# Aggregation across training seeds
# ----------------------------------------------------------------------------
def summarise_seeds(per_seed):
    """per_seed: list of deadline_satisfaction values (one per training seed)."""
    a = np.array(per_seed, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, choices=["general_24h", "a100_peak"])
    ap.add_argument("--v2v-models", nargs="*", default=[],
                    help="RSU-V2V-Local checkpoints, one per training seed")
    ap.add_argument("--local-models", nargs="*", default=[],
                    help="RSU-Local (V2V-disabled) checkpoints, one per training seed")
    ap.add_argument("--baselines", nargs="*", default=["greedy", "random", "local"],
                    choices=["greedy", "random", "local"])
    ap.add_argument("--pos-csv", default=None,
                    help="Replay this FCD-derived position CSV (or fragment dir) "
                         "instead of running live SUMO. Should correspond to the "
                         "chosen --experiment.")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--n-episodes", type=int, default=N_EVAL_EPISODES,
                    help=f"episodes on the shared eval set (default {N_EVAL_EPISODES})")
    args = ap.parse_args()

    rows = []  # (method, seed_or_'-', deadline_sat, mean_delay, frac_local/v2v/rsu)

    def run_drl_set(label, paths, no_v2v):
        per_seed = []
        per_seed_eu = []
        per_seed_es = []
        for i, mp in enumerate(paths):
            seed_tag = TRAINING_SEEDS[i] if i < len(TRAINING_SEEDS) else i
            print(f"[{label}] seed {seed_tag}: {mp}")
            m = evaluate_one(args.experiment, policy="drl", model_path=mp,
                             no_v2v=no_v2v, pos_csv=args.pos_csv, n_episodes=args.n_episodes)
            per_seed.append(m["deadline_satisfaction"])
            per_seed_eu.append(m["energy_user_j_per_task"])
            per_seed_es.append(m["energy_system_j_per_task"])
            rows.append((label, seed_tag, m["deadline_satisfaction"], m["mean_delay_s"],
                         m["frac_local"], m["frac_v2v"], m["frac_rsu"],
                         m["energy_user_j_per_task"], m["energy_system_j_per_task"]))
        if per_seed:
            mean, sd = summarise_seeds(per_seed)
            eu_mean = float(np.mean(per_seed_eu))
            es_mean = float(np.mean(per_seed_es))
            print(f"  -> {label}: {mean:.4f} ± {sd:.4f} (n={len(per_seed)} seeds)  "
                  f"E/task user {eu_mean:.3f} J, system {es_mean:.3f} J")
            rows.append((f"{label} (mean±SD)", f"n={len(per_seed)}", mean, sd, "", "", "",
                         eu_mean, es_mean))

    if args.v2v_models:
        run_drl_set("RSU-V2V-Local", args.v2v_models, no_v2v=False)
    if args.local_models:
        run_drl_set("RSU-Local", args.local_models, no_v2v=True)

    for b in args.baselines:
        print(f"[baseline] {b}")
        m = evaluate_one(args.experiment, policy=b, no_v2v=False, pos_csv=args.pos_csv, n_episodes=args.n_episodes)
        print(f"  -> {b}: {m['deadline_satisfaction']:.4f} "
              f"(delay {m['mean_delay_s']*1000:.0f} ms, "
              f"L/V/R {m['frac_local']:.2f}/{m['frac_v2v']:.2f}/{m['frac_rsu']:.2f}, "
              f"E/task {m['energy_user_j_per_task']:.3f}/{m['energy_system_j_per_task']:.3f} J)")
        rows.append((b, "-", m["deadline_satisfaction"], m["mean_delay_s"],
                     m["frac_local"], m["frac_v2v"], m["frac_rsu"],
                     m["energy_user_j_per_task"], m["energy_system_j_per_task"]))

    header = ["method", "seed", "deadline_satisfaction", "mean_delay_s_or_SD",
              "frac_local", "frac_v2v", "frac_rsu",
              "energy_user_j_per_task", "energy_system_j_per_task"]
    print("\n" + ",".join(header))
    for r in rows:
        print(",".join(str(x) for x in r))

    if args.out_csv:
        outp = Path(args.out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"\nWrote {outp}")


if __name__ == "__main__":
    main()
