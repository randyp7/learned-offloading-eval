#!/usr/bin/env python3
"""
Build the CAiSE 2027 tables and figures from the evaluation CSVs.
Every number in the paper comes from here; re-run after new results land.

  python3 build/make_tables.py            # writes build/tab_*.tex, build/fig_*.pdf, build/numbers.json

Result files (edit RES to repoint at re-evaluated windows).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = Path(__file__).resolve().parent
WIN = [("00-04", "Night 00--04"), ("20-24", "Evening 20--24"), ("08-09", "Peak 08--09"),
       ("16-18", "Peak 16--18"), ("24h", "Full day")]
# main-results files per window: (deadline-only arm, energy-in-reward arm)
RES = {
    "00-04": (R / "deadline_only/eval_corridor_00-04.csv", R / "energyaware_evals/eval_energyaware_00-04.csv"),
    "20-24": (R / "deadline_only/eval_corridor_20-24.csv", R / "energyaware_evals/eval_energyaware_20-24.csv"),
    "08-09": (R / "deadline_only/eval_a100_peak.csv",      R / "energyaware_evals/eval_energyaware_a100.csv"),
    "16-18": (R / "deadline_only/eval_corridor_16-18.csv", None),
    "24h":   (R / "deadline_only/eval_general_24h.csv",    None),
}
# re-evaluations on continuous-trace slices (used when present)
for w in ["00-04", "08-09", "16-18", "20-24"]:
    m, e = R / f"eval_main_{w}.csv", R / f"eval_main_ea_{w}.csv"
    if m.exists():
        RES[w] = (m, e if e.exists() else RES[w][1])
PAIRED = {w: (R / f"eval_paired_warm_{w}.csv", R / f"eval_paired_cold_{w}.csv") for w, _ in WIN}
DENS = {"00-04": 4.2, "20-24": 135, "08-09": 375, "16-18": 459, "24h": 274}  # veh/frame, measured from the trace
LABEL = {"RSU-V2V-Local": "Learned, full action set", "RSU-Local": "Learned, no V2V (ablation)",
         "greedy": "Greedy rule", "local": "Local-only", "random": "Random"}
ORDER = ["RSU-V2V-Local", "RSU-Local", "greedy", "local", "random"]


def load(f):
    df = pd.read_csv(f); df["seed"] = df["seed"].astype(str)
    return df[~df["seed"].str.startswith("n=")]


def per_method(df):
    g = df.groupby("method")
    return g.agg(ds=("deadline_satisfaction", "mean"), sd=("deadline_satisfaction", "std"), n=("deadline_satisfaction", "size"),
                 eu=("energy_user_j_per_task", "mean"), es=("energy_system_j_per_task", "mean"),
                 fl=("frac_local", "mean"), fv=("frac_v2v", "mean"), fr=("frac_rsu", "mean"))


def fmt(m, sd, n):
    return f"{m:.3f}" if n <= 1 or np.isnan(sd) else f"{m:.3f} $\\pm$ {sd:.3f}"


num = {}
# ---------------- Table 1: service by window ----------------
rows = {k: [] for k in ORDER}; diffs = {}
for w, lab in WIN:
    df = load(RES[w][0]); pm = per_method(df)
    for k in ORDER:
        r = pm.loc[k]; rows[k].append(fmt(r.ds, r.sd, r.n))
        num[f"ds_{k}_{w}"] = round(float(r.ds), 4)
    a = df[df.method == "RSU-V2V-Local"].sort_values("seed").deadline_satisfaction.to_numpy()
    b = df[df.method == "RSU-Local"].sort_values("seed").deadline_satisfaction.to_numpy()
    d = a - b; t = stats.ttest_rel(a, b)
    diffs[w] = dict(mean=float(d.mean()), sd=float(d.std(ddof=1)), p=float(t.pvalue), allgt=bool(a.min() > b.max()),
                    vs_greedy=float(a.mean() - pm.loc["greedy"].ds))
    num[f"v2v_delta_{w}"] = round(float(d.mean()), 4); num[f"v2v_p_{w}"] = round(float(t.pvalue), 4)
    num[f"v2v_allgt_{w}"] = diffs[w]["allgt"]; num[f"v2v_vs_greedy_{w}"] = round(diffs[w]["vs_greedy"], 4)
with open(OUT / "tab_service.tex", "w") as f:
    f.write("\\begin{tabular}{@{}l" + "c" * len(WIN) + "@{}}\n\\toprule\n")
    f.write("Policy & " + " & ".join(lab for _, lab in WIN) + " \\\\\n")
    f.write("Vehicles per frame & " + " & ".join(f"{DENS[w]:g}" for w, _ in WIN) + " \\\\\n\\midrule\n")
    for k in ORDER:
        f.write(LABEL[k] + " & " + " & ".join(rows[k]) + " \\\\\n")
    f.write("\\midrule\nV2V benefit (paired) & " + " & ".join(
        f"{diffs[w]['mean']*100:+.1f}~pp{'$^{*}$' if diffs[w]['p']<0.001 else ''}" for w, _ in WIN) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------------- Figure 1: density response ----------------
fig, ax = plt.subplots(figsize=(5.2, 3.3))
xs = [DENS[w] for w, _ in WIN if w != "24h"]; order_w = [w for w, _ in WIN if w != "24h"]
sty = {"RSU-V2V-Local": ("o-", "tab:blue"), "RSU-Local": ("s--", "tab:orange"), "greedy": ("^-", "tab:green"),
       "local": ("v-", "tab:gray"), "random": ("x:", "tab:red")}
for k in ORDER:
    ys = [num[f"ds_{k}_{w}"] for w in order_w]
    ax.plot(xs, ys, sty[k][0], color=sty[k][1], label=LABEL[k], ms=5)
    if k in ("RSU-V2V-Local", "RSU-Local"):
        for w, x in zip(order_w, xs):
            df = load(RES[w][0]); s = df[df.method == k].deadline_satisfaction
            ax.scatter([x] * len(s), s, color=sty[k][1], s=8, alpha=.45, zorder=3)
ax.set_xscale("log"); ax.set_xlabel("Vehicles per frame on the corridor (log scale)")
ax.set_ylabel("Deadline satisfaction"); ax.set_ylim(0.3, 0.8); ax.grid(alpha=.3)
ax.legend(fontsize=7, loc="lower left", ncol=2); fig.tight_layout(); fig.savefig(OUT / "fig_density.pdf"); plt.close(fig)

# ---------------- Table 3: energy ledgers (full day + peak) ----------------
with open(OUT / "tab_energy.tex", "w") as f:
    f.write("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n")
    f.write(" & \\multicolumn{3}{c}{Full day} & \\multicolumn{3}{c}{Evening peak (16--18)} \\\\\n")
    f.write("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nPolicy & Deadline & User J & System J & Deadline & User J & System J \\\\\n\\midrule\n")
    for k in ORDER:
        cells = []
        for w in ("24h", "16-18"):
            r = per_method(load(RES[w][0])).loc[k]; cells += [f"{r.ds:.3f}", f"{r.eu:.2f}", f"{r.es:.2f}"]
            num[f"eu_{k}_{w}"] = round(float(r.eu), 3); num[f"es_{k}_{w}"] = round(float(r.es), 3)
        f.write(LABEL[k] + " & " + " & ".join(cells) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
for w in ("24h", "16-18"):
    num[f"user_vs_local_{w}"] = round(100 * (num[f"eu_RSU-V2V-Local_{w}"] / num[f"eu_local_{w}"] - 1), 1)
    num[f"sys_vs_local_{w}"] = round(100 * (num[f"es_RSU-V2V-Local_{w}"] / num[f"es_local_{w}"] - 1), 1)
    num[f"sys_vs_greedy_{w}"] = round(100 * (num[f"es_RSU-V2V-Local_{w}"] / num[f"es_greedy_{w}"] - 1), 1)

# ---------------- Table 4: energy-in-reward ablation ----------------
rw_wins = [(w, lab) for w, lab in WIN if RES[w][1] is not None and Path(RES[w][1]).exists()]
cells = {("ds", 0): [], ("ds", 1): [], ("eu", 0): [], ("eu", 1): [], ("es", 0): [], ("es", 1): []}
for w, lab in rw_wins:
    d0 = load(RES[w][0]); e = load(RES[w][1])
    a = d0[d0.method == "RSU-V2V-Local"].sort_values("seed"); b = e[e.method == "RSU-V2V-Local"].sort_values("seed")
    for col, key in (("deadline_satisfaction", "ds"), ("energy_user_j_per_task", "eu"), ("energy_system_j_per_task", "es")):
        x, y = a[col].to_numpy(), b[col].to_numpy(); t = stats.ttest_rel(y, x); dm = y.mean() - x.mean()
        if key == "ds":
            cells[(key, 0)].append(f"{x.mean():.3f} $\\to$ {y.mean():.3f}"); cells[(key, 1)].append(f"{dm*100:+.1f}~pp ($p$={t.pvalue:.2f})")
        else:
            cells[(key, 0)].append(f"{x.mean():.2f} $\\to$ {y.mean():.2f}"); cells[(key, 1)].append(f"{100*dm/x.mean():+.0f}\\% ($p$={t.pvalue:.2f})")
        num[f"rw_{key}_{w}"] = dict(base=round(float(x.mean()), 4), ea=round(float(y.mean()), 4), p=round(float(t.pvalue), 3))
with open(OUT / "tab_reward.tex", "w") as f:
    f.write("\\begin{tabular}{@{}l" + "c" * len(rw_wins) + "@{}}\n\\toprule\n")
    f.write("Metric & " + " & ".join(lab for _, lab in rw_wins) + " \\\\\n\\midrule\n")
    for key, name in (("ds", "Deadline satisfaction"), ("eu", "User energy, J/task"), ("es", "System energy, J/task")):
        f.write(name + " & " + " & ".join(cells[(key, 0)]) + " \\\\\n")
        f.write("\\quad change (paired $t$-test) & " + " & ".join(cells[(key, 1)]) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------------- Table 2: paired warm/cold (when available) ----------------
have = {w: (PAIRED[w][0].exists() and PAIRED[w][1].exists()) for w, _ in WIN}
num["cold_available"] = [w for w, v in have.items() if v]
with open(OUT / "tab_cold.tex", "w") as f:
    f.write("\\begin{tabular}{@{}lcccccccccc@{}}\n\\toprule\n")
    f.write("Window & \\multicolumn{2}{c}{Veh./frame} & \\multicolumn{2}{c}{Learned, full set} & \\multicolumn{2}{c}{Greedy rule} & \\multicolumn{2}{c}{V2V benefit (pp)} & \\multicolumn{2}{c}{Learned $-$ greedy (pp)} \\\\\n")
    f.write("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}\n & warm & cold & warm & cold & warm & cold & warm & cold & warm & cold \\\\\n\\midrule\n")
    for w, lab in WIN:
        if not have[w]:
            f.write(lab + " & \\multicolumn{10}{c}{\\textit{pending}} \\\\\n"); continue
        dw, dc = load(PAIRED[w][0]), load(PAIRED[w][1])
        def st(kind):
            q = ROOT / f"data/paired/{kind}_{w}/_stats.csv"
            return f"{pd.read_csv(q).veh_per_frame_mean.mean():.0f}" if q.exists() else "--"
        def parts(df):
            a = df[df.method == "RSU-V2V-Local"].sort_values("seed").deadline_satisfaction.to_numpy()
            b = df[df.method == "RSU-Local"].sort_values("seed").deadline_satisfaction.to_numpy()
            g = float(df[df.method == "greedy"].deadline_satisfaction.mean()); return a, b, g
        aw, bw, gw = parts(dw); ac, bc, gc = parts(dc)
        f.write(f"{lab} & {st('warm')} & {st('cold')} & {aw.mean():.3f} & {ac.mean():.3f} & {gw:.3f} & {gc:.3f} & "
                f"{100*(aw-bw).mean():+.1f} & {100*(ac-bc).mean():+.1f} & {100*(aw.mean()-gw):+.1f} & {100*(ac.mean()-gc):+.1f} \\\\\n")
        num[f"cold_{w}"] = dict(warm_full=round(float(aw.mean()), 4), cold_full=round(float(ac.mean()), 4), warm_greedy=round(gw, 4), cold_greedy=round(gc, 4),
                                warm_ben=round(float((aw - bw).mean()), 4), cold_ben=round(float((ac - bc).mean()), 4),
                                warm_gap=round(float(aw.mean() - gw), 4), cold_gap=round(float(ac.mean() - gc), 4),
                                p_full=round(float(stats.ttest_rel(aw, ac).pvalue), 4), warm_veh=st("warm"), cold_veh=st("cold"))
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------------- Table 2b: cold-TRAINED policies vs warm-trained (when available) ----------------
# eval_coldtrain_paired_{warm,cold}_<w>.csv: policies trained on 20-frame cold-start episodes (the original
# study's protocol), evaluated on the SAME paired windows as the warm-trained policies above.
CT = {w: (R / f"eval_coldtrain_paired_warm_{w}.csv", R / f"eval_coldtrain_paired_cold_{w}.csv") for w, _ in WIN}
have_ct = {w: (CT[w][0].exists() and CT[w][1].exists() and have[w]) for w, _ in WIN}
num["coldtrain_available"] = [w for w, v in have_ct.items() if v]
with open(OUT / "tab_coldtrain.tex", "w") as f:
    f.write("\\begin{tabular}{@{}llcccccc@{}}\n\\toprule\n")
    f.write("Window & Trained on & \\multicolumn{2}{c}{Learned, full set} & \\multicolumn{2}{c}{V2V benefit (pp)} & \\multicolumn{2}{c}{Learned $-$ greedy (pp)} \\\\\n")
    f.write("\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n & & warm & cold & warm & cold & warm & cold \\\\\n\\midrule\n")
    for w, lab in WIN:
        if not have_ct[w]:
            f.write(lab + " & \\multicolumn{7}{c}{\\textit{pending}} \\\\\n"); continue
        def parts(df):
            a = df[df.method == "RSU-V2V-Local"].sort_values("seed").deadline_satisfaction.to_numpy()
            b = df[df.method == "RSU-Local"].sort_values("seed").deadline_satisfaction.to_numpy()
            g = float(df[df.method == "greedy"].deadline_satisfaction.mean()); return a, b, g
        res = {}
        for tr, files in (("warm", PAIRED[w]), ("cold", CT[w])):
            aw, bw, gw = parts(load(files[0])); ac, bc, gc = parts(load(files[1]))
            res[tr] = dict(aw=aw, bw=bw, gw=gw, ac=ac, bc=bc, gc=gc)
            f.write(f"{lab if tr == 'warm' else ''} & {tr} episodes & {aw.mean():.3f} & {ac.mean():.3f} & "
                    f"{100*(aw-bw).mean():+.1f} & {100*(ac-bc).mean():+.1f} & {100*(aw.mean()-gw):+.1f} & {100*(ac.mean()-gc):+.1f} \\\\\n")
        W_, C_ = res["warm"], res["cold"]
        num[f"coldtrain_{w}"] = dict(
            ct_warm_full=round(float(C_["aw"].mean()), 4), ct_cold_full=round(float(C_["ac"].mean()), 4),
            ct_warm_ben=round(float((C_["aw"] - C_["bw"]).mean()), 4), ct_cold_ben=round(float((C_["ac"] - C_["bc"]).mean()), 4),
            ct_warm_gap=round(float(C_["aw"].mean() - C_["gw"]), 4), ct_cold_gap=round(float(C_["ac"].mean() - C_["gc"]), 4),
            # warm-trained minus cold-trained, same windows, paired by seed
            train_effect_warm_eval=round(float((W_["aw"] - C_["aw"]).mean()), 4), p_train_warm_eval=round(float(stats.ttest_rel(W_["aw"], C_["aw"]).pvalue), 4),
            train_effect_cold_eval=round(float((W_["ac"] - C_["ac"]).mean()), 4), p_train_cold_eval=round(float(stats.ttest_rel(W_["ac"], C_["ac"]).pvalue), 4),
            greedy_check=round(float(C_["gw"] - W_["gw"]), 5), ct_sd_warm=round(float(C_["aw"].std(ddof=1)), 4), ct_sd_cold=round(float(C_["ac"].std(ddof=1)), 4))
    f.write("\\bottomrule\n\\end{tabular}\n")
# Welch tests (independent training runs) and per-seed averages over the loaded windows
loaded = [w for w in ["08-09", "16-18", "20-24", "24h"] if have_ct.get(w)]
if loaded:
    def fullset(f, m="RSU-V2V-Local"):
        d = load(f); return d[d.method == m].sort_values("seed").deadline_satisfaction.to_numpy()
    acc = {}
    for w in loaded:
        for ev, idx in (("warm", 0), ("cold", 1)):
            for tr, files in (("wt", PAIRED[w]), ("ct", CT[w])):
                a, b = fullset(files[idx]), fullset(files[idx], "RSU-Local")
                acc.setdefault((ev, tr, "full"), []).append(a); acc.setdefault((ev, tr, "ben"), []).append(a - b)
                num[f"coldtrain_{w}"][f"welch_p_{ev}_eval"] = round(float(stats.ttest_ind(fullset(PAIRED[w][idx]), fullset(CT[w][idx]), equal_var=False).pvalue), 4)
                num[f"coldtrain_{w}"][f"ben_p_{tr}_{ev}"] = round(float(stats.ttest_rel(a, b).pvalue), 5)
    for ev in ("warm", "cold"):
        for q in ("full", "ben"):
            wt = np.mean(acc[(ev, "wt", q)], axis=0); ct = np.mean(acc[(ev, "ct", q)], axis=0)
            num[f"coldtrain_loaded_{q}_{ev}_eval"] = dict(warm_trained=round(float(wt.mean()), 4), cold_trained=round(float(ct.mean()), 4),
                                                        diff=round(float(wt.mean() - ct.mean()), 4), welch_p=round(float(stats.ttest_ind(wt, ct, equal_var=False).pvalue), 4),
                                                        sd_wt=round(float(wt.std(ddof=1)), 4), sd_ct=round(float(ct.std(ddof=1)), 4), windows=loaded)
# cold-trained policies on the main trace slices (per-window full-set and gap), for the text
for w in ["00-04", "08-09", "16-18", "20-24"]:
    m = R / f"eval_coldtrain_main_{w}.csv"
    if m.exists():
        d = per_method(load(m))
        num[f"coldtrain_main_{w}"] = {k: round(float(d.loc[k, "ds"]), 4) for k in ORDER if k in d.index}
        num[f"coldtrain_main_{w}"]["sd_full"] = round(float(d.loc["RSU-V2V-Local", "sd"]), 4)

# ---------------- Figure 2: diurnal density with windows ----------------
hourly = {0: 3, 1: 3, 2: 4, 3: 5, 4: 15, 5: 148, 6: 347, 7: 370, 8: 375, 9: 368, 10: 383, 11: 385, 12: 358, 13: 393,
          14: 413, 15: 460, 16: 463, 17: 455, 18: 457, 19: 417, 20: 287, 21: 141, 22: 73, 23: 41}
fig, ax = plt.subplots(figsize=(5.2, 2.6))
ax.step(list(hourly) + [24], list(hourly.values()) + [hourly[23]], where="post", color="k", lw=1.2, label="Warm trace (hourly mean)")
for (h0, h1, lab) in [(0, 4, "00--04"), (8, 9, "08--09"), (16, 18, "16--18"), (20, 24, "20--24")]:
    ax.axvspan(h0, h1, alpha=.15, color="tab:blue"); ax.text((h0 + h1) / 2, 480, lab.replace("--", "–"), ha="center", fontsize=7)
for w, _ in WIN:
    sc = ROOT / f"data/paired/cold_{w}/_stats.csv"
    if sc.exists() and w != "24h":
        s = pd.read_csv(sc); ax.scatter(s["T"] / 3600, s.veh_per_frame_mean, s=10, color="tab:red", zorder=3,
                                       label="Cold-start windows" if w == "00-04" else None)
ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 4)); ax.set_xlabel("Hour of day"); ax.set_ylabel("Vehicles per frame")
ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT / "fig_diurnal.pdf"); plt.close(fig)

(OUT / "numbers.json").write_text(json.dumps(num, indent=1))
print(json.dumps({k: v for k, v in num.items() if not k.startswith("ds_")}, indent=1)[:3000])
