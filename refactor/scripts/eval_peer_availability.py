"""Peer-availability audit of the shared evaluation episodes.

Replays the shared seeded evaluation exactly as evaluate_models.evaluate_one does
(same scenario, seed 7000, episodes, deterministic DQN / Greedy), and records,
for every task decision, read-only from the wrapper state:
  - whether a V2V candidate exists (action mask; peer within 200 m),
  - the selected candidate peer's queue wait (0 = a core is free),
  - the number of vehicles within V2V range (observation feature 22 x 10),
  - the executed action and whether the committed task met its deadline.
No simulator code is modified; the overall deadline satisfaction is written as
a control that must reproduce the published value. Writes a new summary CSV.
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
ap.add_argument("--out-csv", required=True)
a = ap.parse_args()
if Path(a.out_csv).exists():
    sys.exit(f"refusing to overwrite {a.out_csv}")


def run(policy, model_path=None):
    scenario = em.build_scenario(em.create_scenario_config, experiment=a.experiment, seed=em.EVAL_SEED,
                                 no_v2v=False, name=f"peeraudit_{a.experiment}_{policy}")
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
        else:
            act = em.greedy_action(uenv)
        if cur is not None and cur[1] is not None:
            task, cands, feats, mask = cur
            v = cands.get("v2v")
            pq = float(v.get("queue_s", np.nan)) if v is not None else np.nan
            executed = act if mask[act] == 1 else 0
            n_before = len(committed)
        else:
            task = None
        obs, _, dones, _ = venv.step([act])
        if task is not None and len(committed) > n_before:
            out = committed[n_before]
            recs.append((int(mask[1]), pq, float(feats[22]) * 10.0, executed, int(bool(out.get("deadline_met", False)))))
        if dones[0]:
            episodes += 1
    m = em.read_step_summary(core)
    venv.close()
    r = np.array(recs, dtype=float)
    peer = r[:, 0] == 1; free = peer & (r[:, 1] <= 1e-9); busy = peer & (r[:, 1] > 1e-9)
    near = r[:, 2]; act = r[:, 3]; met = r[:, 4] == 1
    def rate(x, sel): return float(x[sel].mean()) if sel.any() else float("nan")
    return dict(
        published_control_ds=m["deadline_satisfaction"], n_tasks=len(r), audit_ds=float(met.mean()),
        p_peer=float(peer.mean()), p_peer_free=float(free.mean()), p_peer_busy=float(busy.mean()),
        nearby_mean=float(near.mean()), nearby_median=float(np.median(near)),
        nearby_q25=float(np.quantile(near, .25)), nearby_q75=float(np.quantile(near, .75)),
        v2v_share=float((act == 1).mean()), v2v_share_given_peer=rate(act == 1, peer),
        v2v_share_given_free=rate(act == 1, free),
        met_v2v=rate(met, act == 1), met_local=rate(met, act == 0), met_rsu=rate(met, act == 2),
        met_given_peer=rate(met, peer), met_given_nopeer=rate(met, ~peer))

rows = []
for i, mp in enumerate(a.v2v_models):
    s = em.TRAINING_SEEDS[i]; print(f"[DQN-Full] seed {s}", flush=True)
    rows.append({"policy": "DQN-Full", "seed": s, **run("drl", mp)}); print(rows[-1], flush=True)
print("[Greedy]", flush=True)
rows.append({"policy": "Greedy", "seed": "-", **run("greedy")}); print(rows[-1], flush=True)
Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
with open(a.out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote", a.out_csv)
