"""Executor-load audit of the shared evaluation episodes.

Replays the shared seeded evaluation exactly as evaluate_models.evaluate_one does
(same scenario, seed 7000, episodes, deterministic DQN / Greedy / Local), and
records, for every task decision, read-only from the wrapper state before the
action is committed:
  - for each candidate executor (requesting vehicle, preselected peer, nearest
    RSU): whether it is available (action mask), its predicted waiting time
    (candidate queue_s), the share of its cores busy at decision time and its
    backlog in core-seconds (sum over cores of max(0, free_at - now));
  - whether every available executor is already loaded (queue_s > 0);
  - the executed action and whether the committed task met its deadline.
The decision time is taken from the candidate itself (finish_s - delay_s), so no
simulator method is called. No simulator code is modified; the overall deadline
satisfaction is written as a control that must reproduce the published value.
Writes a new summary CSV.
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_models as em

ap = argparse.ArgumentParser()
ap.add_argument("--experiment", default="a100_peak", choices=["general_24h", "a100_peak"])
ap.add_argument("--pos-csv", required=True)
ap.add_argument("--n-episodes", type=int, default=20)
ap.add_argument("--v2v-models", nargs="*", default=[])
ap.add_argument("--policies", nargs="*", default=["drl", "greedy", "local"],
                choices=["drl", "greedy", "local"])
ap.add_argument("--out-csv", required=True)
a = ap.parse_args()
if Path(a.out_csv).exists():
    sys.exit(f"refusing to overwrite {a.out_csv}")

KINDS = ("local", "v2v", "rsu")          # action index 0, 1, 2


def heap_state(spec, now):
    """Busy-core share and backlog (core-seconds) of one executor; read-only."""
    heap = spec.get("heap") if spec is not None else None
    if not heap:
        return np.nan, np.nan
    free_at = np.array([float(h[0]) for h in heap])
    return float((free_at > now + 1e-9).mean()), float(np.clip(free_at - now, 0, None).sum())


def run(policy, model_path=None):
    scenario = em.build_scenario(em.create_scenario_config, experiment=a.experiment, seed=em.EVAL_SEED,
                                 no_v2v=False, name=f"loadaudit_{a.experiment}_{policy}")
    venv = em.create_offloading_env(scenario, pos_csv=a.pos_csv)
    uenv = venv.envs[0].unwrapped
    core = uenv.e
    committed = []
    orig_commit = core._commit_candidate
    def commit_wrap(cand, task, *args, **kw):
        out = orig_commit(cand, task, *args, **kw); committed.append(out); return out
    core._commit_candidate = commit_wrap
    model = None
    if policy == "drl":
        from stable_baselines3 import DQN
        model = DQN.load(model_path)
    recs = []
    obs = venv.reset(); episodes = 0
    while episodes < a.n_episodes:
        cur = uenv._current
        if policy == "drl":
            act, _ = model.predict(obs, deterministic=True); act = int(act[0])
        elif policy == "greedy":
            act = em.greedy_action(uenv)
        else:
            act = 0
        if cur is not None and cur[1] is not None:
            task, cands, feats, mask = cur
            row = []
            loaded_all = True
            for i, k in enumerate(KINDS):
                c = cands.get(k)
                avail = int(mask[i] == 1 and c is not None)
                if avail:
                    now = float(c["finish_s"]) - float(c["delay_s"])
                    spec = (core.rsu_spec if k == "rsu" else core.veh_spec).get(c["node_id"])
                    busy, backlog = heap_state(spec, now)
                    q = float(c["queue_s"])
                    loaded_all = loaded_all and q > 1e-9
                else:
                    busy = backlog = q = np.nan
                row += [avail, q, busy, backlog]
            executed = act if mask[act] == 1 else 0
            n_before = len(committed)
        else:
            task = None
        obs, _, dones, _ = venv.step([act])
        if task is not None and len(committed) > n_before:
            out = committed[n_before]
            recs.append(row + [int(loaded_all), executed, int(bool(out.get("deadline_met", False)))])
        if dones[0]:
            episodes += 1
    m = em.read_step_summary(core)
    venv.close()
    r = np.array(recs, dtype=float)
    act = r[:, 13]; met = r[:, 14] == 1; all_loaded = r[:, 12] == 1
    def rate(x, sel): return float(x[sel].mean()) if sel.any() else float("nan")
    res = dict(published_control_ds=m["deadline_satisfaction"], n_tasks=len(r), audit_ds=float(met.mean()),
               share_all_available_loaded=float(all_loaded.mean()))
    for i, k in enumerate(KINDS):
        av, q, busy, backlog = (r[:, 4 * i + j] for j in range(4))
        sel = av == 1
        res.update({
            f"{k}_available": float(sel.mean()),
            f"{k}_wait_mean_s": float(np.nanmean(q[sel])) if sel.any() else np.nan,
            f"{k}_wait_pos_share": float((q[sel] > 1e-9).mean()) if sel.any() else np.nan,
            f"{k}_busy_core_share": float(np.nanmean(busy[sel])) if sel.any() else np.nan,
            f"{k}_backlog_core_s": float(np.nanmean(backlog[sel])) if sel.any() else np.nan,
            f"{k}_action_share": float((act == i).mean()),
            f"met_{k}": rate(met, act == i),
            f"missed_share_{k}": float(((act == i) & ~met).sum() / max((~met).sum(), 1)),
        })
    return res

rows = []
if "drl" in a.policies:
    for i, mp in enumerate(a.v2v_models):
        s = em.TRAINING_SEEDS[i]; print(f"[DQN-Full] seed {s}", flush=True)
        rows.append({"policy": "DQN-Full", "seed": s, **run("drl", mp)}); print(rows[-1], flush=True)
for p, name in (("greedy", "Greedy"), ("local", "Local")):
    if p in a.policies:
        print(f"[{name}]", flush=True)
        rows.append({"policy": name, "seed": "-", **run(p)}); print(rows[-1], flush=True)
Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
with open(a.out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote", a.out_csv)
