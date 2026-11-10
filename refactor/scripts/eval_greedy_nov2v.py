"""Greedy-NoV2V comparator: Greedy with and without V2V on the same evaluation episodes.

Runs the Greedy rule twice on one evaluation window with the shared seeded
evaluation set: once as published (V2V available; control that must reproduce
the existing Greedy value) and once with V2V disabled (V2V range 0, so Greedy
chooses between local execution and the roadside unit). Writes a new CSV; no
existing result file is touched.
"""
import argparse, csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_models import evaluate_one

ap = argparse.ArgumentParser()
ap.add_argument("--experiment", default="a100_peak", choices=["general_24h", "a100_peak"])
ap.add_argument("--pos-csv", required=True)
ap.add_argument("--n-episodes", type=int, default=20)
ap.add_argument("--out-csv", required=True)
a = ap.parse_args()
if Path(a.out_csv).exists():
    sys.exit(f"refusing to overwrite {a.out_csv}")
hdr = ["method", "seed", "deadline_satisfaction", "mean_delay_s_or_SD", "frac_local", "frac_v2v",
       "frac_rsu", "energy_user_j_per_task", "energy_system_j_per_task"]
rows = []
for label, nov in (("greedy", False), ("greedy_nov2v", True)):
    m = evaluate_one(a.experiment, policy="greedy", no_v2v=nov, pos_csv=a.pos_csv, n_episodes=a.n_episodes)
    print(f"{label}: {m['deadline_satisfaction']:.4f} L/V/R {m['frac_local']:.3f}/{m['frac_v2v']:.3f}/{m['frac_rsu']:.3f}", flush=True)
    rows.append([label, "-", m["deadline_satisfaction"], m["mean_delay_s"], m["frac_local"], m["frac_v2v"],
                 m["frac_rsu"], m["energy_user_j_per_task"], m["energy_system_j_per_task"]])
Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
with open(a.out_csv, "w", newline="") as f:
    w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
print("wrote", a.out_csv)
