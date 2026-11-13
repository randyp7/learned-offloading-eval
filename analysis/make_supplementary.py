#!/usr/bin/env python3
"""
Build the tables of the supplementary analyses (docs/supplementary-*.md) from results/.

  python3 analysis/make_supplementary.py      # writes analysis/supplementary_tables.md

Every table in docs/supplementary-service-analysis.md, docs/supplementary-state-continuity.md
and docs/supplementary-energy-analysis.md is a copy of a section of the output file.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = Path(__file__).resolve().parent / "supplementary_tables.md"
sys.path.insert(0, str(ROOT))

LOADED = [("20-24", "Evening 20–24"), ("08-09", "Peak 08–09"), ("16-18", "Peak 16–18")]
WIN = [("00-04", "Night 00–04")] + LOADED
LABEL = {"RSU-V2V-Local": "DQN-Full", "RSU-Local": "DQN-NoV2V", "greedy": "Greedy", "local": "Local", "random": "Random"}
ORDER = list(LABEL)
sections = []


def load(f):
    d = pd.read_csv(f); d["seed"] = d["seed"].astype(str)
    return d[~d["seed"].str.startswith("n=")]


def seeds(f, m="RSU-V2V-Local", col="deadline_satisfaction"):
    d = load(f); return d[d.method == m].sort_values("seed")[col].to_numpy()


def table(head, rows):
    s = "| " + " | ".join(head) + " |\n|" + "|".join("---" if i == 0 else "---:" for i in range(len(head))) + "|\n"
    return s + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)


def add(name, body):
    sections.append(f"## {name}\n\n{body}")


def pc(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{nd}f}"


def welch_ci(x, y, level=0.95):
    """Difference of means x - y with a Welch confidence interval."""
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = np.sqrt(vx + vy); df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    h = stats.t.ppf(0.5 + level / 2, df) * se; d = x.mean() - y.mean()
    return d, d - h, d + h


# ============================ RQ1 ============================
PEER = [("main_00-04", "Night 00–04"), ("main_04-08", "Ramp 04–08"), ("main_20-24", "Evening 20–24"),
        ("main_08-09", "Peak 08–09"), ("main_16-18", "Peak 16–18"), ("general_24h", "Full day")]
pa = {t: pd.read_csv(R / f"peer_audit/eval_peer_availability_{t}.csv") for t, _ in PEER}
mixf = {t: (R / "deadline_only/eval_general_24h.csv" if t == "general_24h" else R / f"eval_{t}.csv") for t, _ in PEER}


def pcol(t, c, pol="DQN-Full"):
    d = pa[t]; return float(d[d.policy == pol][c].mean())


def mix(f, m):
    d = load(f); d = d[d.method == m]
    return " / ".join(f"{100 * float(d[c].astype(float).mean()):.1f}" for c in ("frac_local", "frac_v2v", "frac_rsu"))


rows = [
    ["Peer within 200 m (% of tasks)"] + [pc(pcol(t, "p_peer", "Greedy")) for t, _ in PEER],
    ["Candidate peer has a free core (% of tasks)"] + [pc(pcol(t, "p_peer_free")) for t, _ in PEER],
    ["Vehicles within 200 m, median (IQR)"] + [f"{pcol(t, 'nearby_median', 'Greedy'):.0f} ({pcol(t, 'nearby_q25', 'Greedy'):.0f}–{pcol(t, 'nearby_q75', 'Greedy'):.0f})" for t, _ in PEER],
    ["DQN-Full action mix, local / V2V / RSU (%)"] + [mix(mixf[t], "RSU-V2V-Local") for t, _ in PEER],
    ["Greedy action mix, local / V2V / RSU (%)"] + [mix(mixf[t], "greedy") for t, _ in PEER],
    ["DQN-Full V2V share given a peer (%)"] + [pc(pcol(t, "v2v_share_given_peer")) for t, _ in PEER],
    ["Greedy V2V share given a peer (%)"] + [pc(pcol(t, "v2v_share_given_peer", "Greedy")) for t, _ in PEER],
    ["V2V tasks within deadline, DQN-Full (%)"] + [pc(pcol(t, "met_v2v")) for t, _ in PEER],
    ["V2V tasks within deadline, Greedy (%)"] + [pc(pcol(t, "met_v2v", "Greedy")) for t, _ in PEER],
]
ctrl = max(float((d.published_control_ds - d.audit_ds).abs().max()) for d in pa.values())
add("RQ1-peer", table(["Quantity"] + [l for _, l in PEER], rows) + f"\nControl: largest difference between audited and published deadline satisfaction = {ctrl:.1e}.\n")

LA = {w: pd.read_csv(R / f"load_audit/eval_executor_load_main_{w}.csv") for w, _ in WIN}
rows = []
for w, lab in WIN:
    g = LA[w].groupby("policy").mean(numeric_only=True)
    for p in ("DQN-Full", "Greedy", "Local"):
        r = g.loc[p]; v2v = r.v2v_action_share > 0
        rows.append([lab if p == "DQN-Full" else "", p, pc(r.v2v_busy_core_share), f"{r.v2v_backlog_core_s:.2f}", f"{1000 * r.v2v_wait_mean_s:.0f}",
                     pc(r.met_local), pc(r.met_v2v) if v2v else "–", pc(r.v2v_action_share), pc(r.missed_share_v2v) if v2v else "–", pc(r.share_all_available_loaded, 2)])
ctrl = max(float((d.published_control_ds - d.audit_ds).abs().max()) for d in LA.values())
add("RQ1-load", table(["Period", "Policy", "Candidate-peer busy cores (%)", "Candidate-peer backlog (core-s)", "Predicted peer wait (ms)",
                       "Local within deadline (%)", "V2V within deadline (%)", "V2V share of tasks (%)", "V2V share of misses (%)", "All executors loaded (% of decisions)"], rows)
    + f"\nControl: largest difference between audited and published deadline satisfaction = {ctrl:.1e}.\n")

GN = [("main_00-04", "Night 00–04"), ("main_04-08", "Ramp 04–08"), ("main_20-24", "Evening 20–24"),
      ("main_08-09", "Peak 08–09"), ("main_16-18", "Peak 16–18"), ("general_24h", "Full day")]
rows = []
for t, lab in GN:
    g = pd.read_csv(R / f"greedy_nov2v/eval_greedy_nov2v_{t}.csv").set_index("method")
    f = mixf[t]; full, nov = seeds(f).mean(), seeds(f, "RSU-Local").mean()
    rows.append([lab, f"{g.loc['greedy', 'deadline_satisfaction']:.3f}", f"{g.loc['greedy_nov2v', 'deadline_satisfaction']:.3f}",
                 f"{100 * (g.loc['greedy_nov2v', 'deadline_satisfaction'] - g.loc['greedy', 'deadline_satisfaction']):+.1f}", f"{100 * (full - nov):+.1f}"])
add("RQ1-greedy-nov2v", table(["Period", "Greedy", "Greedy-NoV2V", "Greedy-NoV2V − Greedy (pp)", "DQN-Full − DQN-NoV2V (pp)"], rows))

f = R / "eval_main_04-08.csv"; pm = load(f).groupby("method").deadline_satisfaction.agg(["mean", "std"])
add("RQ1-ramp", table(["Policy", "Deadline satisfaction (mean ± SD over seeds)"],
                      [[LABEL[k], f"{pm.loc[k, 'mean']:.3f}" + (f" ± {pm.loc[k, 'std']:.3f}" if not np.isnan(pm.loc[k, "std"]) else "")] for k in ORDER])
    + f"\nDQN-Full − DQN-NoV2V = {100 * (seeds(f).mean() - seeds(f, 'RSU-Local').mean()):+.1f} pp; DQN-Full − Greedy = {100 * (seeds(f).mean() - pm.loc['greedy', 'mean']):+.1f} pp.\n")

# ============================ RQ2 ============================
rows = []
wp = pd.read_csv(R / "window-pairs.csv")
for w, lab in WIN:
    for reg in ("warm", "cold"):
        p = pd.read_csv(R / f"peer_audit/eval_peer_availability_paired_{reg}_{w}.csv").groupby("policy").mean(numeric_only=True)
        l = pd.read_csv(R / f"load_audit/eval_executor_load_paired_{reg}_{w}.csv").groupby("policy").mean(numeric_only=True)
        sub = wp[wp.period == w]; veh = float(np.average(sub[f"{reg}_veh_per_frame"], weights=sub["times_evaluated"]))
        for pol in ("DQN-Full", "Greedy"):
            a, b = p.loc[pol], l.loc[pol]
            rows.append([lab if (reg == "warm" and pol == "DQN-Full") else "", reg if pol == "DQN-Full" else "", pol, f"{veh:.0f}" if pol == "DQN-Full" else "",
                         f"{a.nearby_median:.0f}", pc(a.p_peer), pc(a.p_peer_free),
                         f"{100 * b.local_action_share:.0f} / {100 * b.v2v_action_share:.0f} / {100 * b.rsu_action_share:.0f}",
                         pc(a.met_local), pc(a.met_v2v), pc(a.met_rsu), pc(b.v2v_busy_core_share), pc(b.rsu_busy_core_share),
                         f"{b.v2v_backlog_core_s:.1f} / {b.rsu_backlog_core_s:.1f}", f"{1000 * b.v2v_wait_mean_s:.0f} / {1000 * b.rsu_wait_mean_s:.0f}"])
add("RQ2-paired-audit", table(["Period", "Regime", "Policy", "Vehicles per frame", "Nearby vehicles (median)", "Peer reachable (%)", "Candidate peer free core (%)",
                               "Local / V2V / RSU (%)", "Local met (%)", "V2V met (%)", "RSU met (%)", "Peer busy cores (%)", "RSU busy cores (%)",
                               "Backlog peer / RSU (core-s)", "Predicted wait peer / RSU (ms)"], rows))

rows = []
for w, lab in WIN:
    a = load(R / f"eval_paired_warm_{w}.csv").groupby("method").deadline_satisfaction.mean()
    b = load(R / f"eval_paired_cold_{w}.csv").groupby("method").deadline_satisfaction.mean()
    rows.append([lab] + [f"{a[k]:.3f} → {b[k]:.3f} ({100 * (b[k] - a[k]):+.1f})" for k in ORDER])
add("RQ2-warm-cold-all-policies", table(["Period"] + [LABEL[k] for k in ORDER], rows))

# cold-training arm
CT = lambda reg, w: R / f"eval_coldtrain_paired_{reg}_{w}.csv"
WT = lambda reg, w: R / f"eval_paired_{reg}_{w}.csv"
ALLW = WIN + [("24h", "Full day")]
rows = []
for w, lab in ALLW:
    for tr, F in (("warm", WT), ("cold", CT)):
        cells = []
        for reg in ("warm", "cold"):
            a, b = seeds(F(reg, w)), seeds(F(reg, w), "RSU-Local"); g = float(load(F(reg, w)).query("method=='greedy'").deadline_satisfaction.mean())
            cells.append((a.mean(), 100 * (a - b).mean(), 100 * (a.mean() - g)))
        rows.append([lab if tr == "warm" else "", f"{tr} episodes", f"{cells[0][0]:.3f}", f"{cells[1][0]:.3f}", f"{cells[0][1]:+.1f}", f"{cells[1][1]:+.1f}", f"{cells[0][2]:+.1f}", f"{cells[1][2]:+.1f}"])
add("RQ2-coldtrain", table(["Window", "Trained on", "DQN-Full, warm eval", "DQN-Full, cold eval", "Full − NoV2V (pp), warm eval",
                            "Full − NoV2V (pp), cold eval", "Full − Greedy (pp), warm eval", "Full − Greedy (pp), cold eval"], rows))

rows = []
for w, lab in ALLW:
    for reg in ("warm", "cold"):
        wt, ct = seeds(WT(reg, w)), seeds(CT(reg, w)); wn, cn = seeds(WT(reg, w), "RSU-Local"), seeds(CT(reg, w), "RSU-Local")
        d, lo, hi = welch_ci(wt - wn, ct - cn)
        rows.append([lab if reg == "warm" else "", reg, f"{wt.mean():.3f} ± {wt.std(ddof=1):.3f}", f"{ct.mean():.3f} ± {ct.std(ddof=1):.3f}",
                     f"{stats.ttest_ind(wt, ct, equal_var=False).pvalue:.2f}", f"{ct.std(ddof=1) / wt.std(ddof=1):.2f}",
                     f"{wn.mean():.3f} → {cn.mean():.3f}", f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"])
add("RQ2-coldtrain-tests", table(["Window", "Eval regime", "DQN-Full warm-trained (mean ± SD)", "DQN-Full cold-trained (mean ± SD)", "Welch p",
                                  "SD ratio cold/warm-trained", "DQN-NoV2V warm- → cold-trained", "Interaction: V2V benefit warm- minus cold-trained (pp) [95% CI]"], rows))

rows = []
for reg in ("warm", "cold"):
    wt = np.mean([seeds(WT(reg, w)) for w, _ in LOADED], axis=0); ct = np.mean([seeds(CT(reg, w)) for w, _ in LOADED], axis=0)
    bw = np.mean([seeds(WT(reg, w)) - seeds(WT(reg, w), "RSU-Local") for w, _ in LOADED], axis=0)
    bc = np.mean([seeds(CT(reg, w)) - seeds(CT(reg, w), "RSU-Local") for w, _ in LOADED], axis=0)
    d, lo, hi = welch_ci(bw, bc)
    rows.append([reg, f"{wt.mean():.3f} ± {wt.std(ddof=1):.3f}", f"{ct.mean():.3f} ± {ct.std(ddof=1):.3f}", f"{100 * (wt.mean() - ct.mean()):+.1f}",
                 f"{stats.ttest_ind(wt, ct, equal_var=False).pvalue:.2f}", f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"])
add("RQ2-coldtrain-loaded", table(["Eval regime", "Warm-trained DQN-Full", "Cold-trained DQN-Full", "Difference (pp)", "Welch p",
                                   "Interaction (pp) [95% CI]"], rows))

rows = []
for (w, lab) in ALLW:
    for reg in ("warm", "cold"):
        a, b = load(WT(reg, w)), load(CT(reg, w))
        a, b = a[a.method == "RSU-V2V-Local"].sort_values("seed"), b[b.method == "RSU-V2V-Local"].sort_values("seed")
        rows.append([lab if reg == "warm" else "", reg] + [f"{x:.3f}" for x in a.deadline_satisfaction] + [f"{x:.3f}" for x in b.deadline_satisfaction])
add("RQ2-coldtrain-seeds", table(["Window", "Eval regime"] + [f"warm-trained {s}" for s in range(100, 105)] + [f"cold-trained {s}" for s in range(100, 105)], rows))

dens = pd.read_csv(R / "coldtrain_density/coldtrain_density_frames.csv")
dens["block"] = pd.cut(dens.k, [-1, 4, 9, 14, 19], labels=["1–5", "6–10", "11–15", "16–20"])
hour = dens["T"] // 3600
groups = [("All day", dens), ("Loaded periods (08–09, 16–18, 20–24)", dens[hour.isin([8, 16, 17, 20, 21, 22, 23])]), ("Night 00–04", dens[hour < 4])]
rows = []
for name, d in groups:
    g = d.groupby("block", observed=True)[["cold", "warm"]]
    blk = zip(g.mean().cold, 100 * g.sum().cold / g.sum().warm)
    rows.append([f"{name} (n = {d['T'].nunique():,})"] + [f"{c:.1f} ({r:.1f}%)" for c, r in blk] + [f"{100 * d.cold.sum() / d.warm.sum():.1f}%", f"{d.warm.mean():.0f}"])
add("RQ2-coldtrain-density", table(["Cold training episodes", "Frames 1–5", "Frames 6–10", "Frames 11–15", "Frames 16–20", "All frames (cold/warm)", "Warm vehicles per frame"], rows))

rows = []
for w, lab in WIN:
    for reg in ("warm", "cold"):
        cells = []
        for tag in (f"paired_{reg}_{w}", f"coldtrain_paired_{reg}_{w}"):
            d = pd.read_csv(R / f"peer_audit/eval_peer_availability_{tag}.csv"); d = d[d.policy == "DQN-Full"]
            cells += [pc(d.v2v_share.mean()), pc(d.v2v_share_given_peer.mean()), pc(d.met_v2v.mean())]
        rows.append([lab if reg == "warm" else "", reg] + cells)
add("RQ2-coldtrain-actions", table(["Window", "Eval regime", "Warm-trained V2V share (%)", "Warm-trained V2V share given peer (%)", "Warm-trained V2V met (%)",
                                    "Cold-trained V2V share (%)", "Cold-trained V2V share given peer (%)", "Cold-trained V2V met (%)"], rows))

# ============================ RQ3 ============================
EW = WIN[:1] + [("04-08", "Ramp 04–08")] + LOADED + [("24h", "Full day")]
ef = lambda w: R / ("deadline_only/eval_general_24h.csv" if w == "24h" else f"eval_main_{w}.csv")
rows = []
for w, lab in EW:
    pm = load(ef(w)).groupby("method").mean(numeric_only=True)
    ea = R / f"eval_main_ea_{w}.csv"
    for k in ORDER:
        r = pm.loc[k]; rows.append([lab if k == ORDER[0] else "", LABEL[k], f"{r.deadline_satisfaction:.3f}", f"{r.energy_user_j_per_task:.2f}", f"{r.energy_system_j_per_task:.2f}",
                                     f"{r.energy_system_j_per_task / r.deadline_satisfaction:.2f}"])
    if ea.exists():
        r = load(ea).query("method=='RSU-V2V-Local'").mean(numeric_only=True)
        rows.append(["", "DQN-Energy", f"{r.deadline_satisfaction:.3f}", f"{r.energy_user_j_per_task:.2f}", f"{r.energy_system_j_per_task:.2f}",
                     f"{r.energy_system_j_per_task / r.deadline_satisfaction:.2f}"])
add("RQ3-ledgers", table(["Period", "Policy", "Deadline satisfaction", "Requester energy (J/task)", "System energy (J/task)", "System energy per deadline-satisfying task (J)"], rows))

rows = []; seedrows = []
for w, lab in WIN:
    a = load(ef(w)).query("method=='RSU-V2V-Local'").sort_values("seed").set_index("seed")
    e = load(R / f"eval_main_ea_{w}.csv").query("method=='RSU-V2V-Local'").sort_values("seed").set_index("seed")
    for col, name, unit in (("deadline_satisfaction", "Deadline satisfaction", "pp"), ("energy_user_j_per_task", "Requester energy (J)", "%"),
                            ("energy_system_j_per_task", "System energy (J)", "%")):
        x, y = a[col].to_numpy(), e[col].to_numpy(); dm = y.mean() - x.mean()
        ch = f"{100 * dm:+.1f} pp" if unit == "pp" else f"{dm:+.3f} J ({100 * dm / x.mean():+.0f}%)"
        rows.append([lab if col == "deadline_satisfaction" else "", name, f"{x.mean():.3f}", f"{y.mean():.3f}", ch,
                     f"{stats.ttest_rel(y, x).pvalue:.2f}", f"{stats.ttest_ind(y, x, equal_var=False).pvalue:.2f}"])
    for s in a.index:
        seedrows.append([lab if s == a.index[0] else "", s, f"{100 * (e.loc[s, 'deadline_satisfaction'] - a.loc[s, 'deadline_satisfaction']):+.1f}",
                         f"{100 * (e.loc[s, 'energy_user_j_per_task'] / a.loc[s, 'energy_user_j_per_task'] - 1):+.0f}",
                         f"{100 * (e.loc[s, 'energy_system_j_per_task'] / a.loc[s, 'energy_system_j_per_task'] - 1):+.0f}",
                         f"{100 * a.loc[s, 'frac_local']:.0f} / {100 * a.loc[s, 'frac_v2v']:.0f} / {100 * a.loc[s, 'frac_rsu']:.0f}",
                         f"{100 * e.loc[s, 'frac_local']:.0f} / {100 * e.loc[s, 'frac_v2v']:.0f} / {100 * e.loc[s, 'frac_rsu']:.0f}"])
add("RQ3-reward", table(["Period", "Metric", "DQN-Full", "DQN-Energy", "Change", "Paired t p", "Welch p"], rows))
add("RQ3-reward-seeds", table(["Period", "Seed", "Deadline change (pp)", "Requester energy change (%)", "System energy change (%)",
                               "DQN-Full local / V2V / RSU (%)", "DQN-Energy local / V2V / RSU (%)"], seedrows))

# break-even: RSU offloading versus local execution for the mean task of each class
from config import (NR_CARRIER_HZ, NR_P_TX_VEH_W, NR_P_RX_VEH_W, NR_P_TX_RSU_W, NR_P_RX_RSU_W, NR_NOISE_FIG_VEH_DB,
                    NR_NOISE_FIG_RSU_DB, ENERGY_PER_CYCLE_J_VEH as EV, ENERGY_PER_CYCLE_J_RSU as ER, RESULT_SIZE_RATIO as RHO)
from net.nrv2x_lite import one_nr_sidelink_hop
from envs.vec_env_offload_nrv2x_m_v01_seeded import NRRadio
veh = NRRadio(carrier_hz=NR_CARRIER_HZ, p_tx_w=NR_P_TX_VEH_W, p_rx_w=NR_P_RX_VEH_W, noise_fig_db=NR_NOISE_FIG_VEH_DB, bw_hz=20e6, ant_gain_db=3.0)
rsu = NRRadio(carrier_hz=NR_CARRIER_HZ, p_tx_w=NR_P_TX_RSU_W, p_rx_w=NR_P_RX_RSU_W, noise_fig_db=NR_NOISE_FIG_RSU_DB, bw_hz=80e6, ant_gain_db=8.0)
CLS = [("Safety", 6e8, 3e5), ("Automated driving", 1.2e9, 8e5), ("Infotainment", 3.6e9, 3e6)]
dist = np.arange(1, 2001)
assert all(one_nr_sidelink_hop(d, 1, rsu, veh)["rate_mbps"] >= one_nr_sidelink_hop(d, 1, veh, rsu)["rate_mbps"] for d in dist)
rmin = min(one_nr_sidelink_hop(d, 1, veh, rsu)["rate_mbps"] for d in dist)
rows = []
for name, C, S in CLS:
    bits = 8 * S
    sys_lo = bits * (NR_P_TX_VEH_W + NR_P_RX_RSU_W) / ((EV - ER) * C) / 1e6
    sys_hi = (bits * (NR_P_TX_VEH_W + NR_P_RX_RSU_W) + RHO * bits * (NR_P_RX_VEH_W + NR_P_TX_RSU_W)) / ((EV - ER) * C) / 1e6
    usr_lo = bits * NR_P_TX_VEH_W / (EV * C) / 1e6
    usr_hi = (bits * NR_P_TX_VEH_W + RHO * bits * NR_P_RX_VEH_W) / (EV * C) / 1e6

    def esys(d):
        u = one_nr_sidelink_hop(d, S, veh, rsu); r = one_nr_sidelink_hop(d, RHO * S, rsu, veh)
        return u["e_tx_j"] + r["e_rx_j"] + C * ER + u["e_rx_j"] + r["e_tx_j"]
    dmax = max(int(d) for d in dist if esys(d) < C * EV)
    rows.append([name, f"{C / 1e6:,.0f}", f"{S / 1e3:,.0f}", f"{C * EV:.2f}", f"{sys_lo:.1f}–{sys_hi:.1f}", f"{dmax}", f"{usr_lo:.1f}–{usr_hi:.1f}"])
add("RQ3-break-even", table(["Class (mean task)", "Cycles (M)", "Input (KB)", "E_local (J)", "System ledger: RSU uplink rate above (Mbit/s)",
                             "System ledger: break-even distance to RSU (m)", "Requester ledger: RSU uplink rate above (Mbit/s)"], rows)
    + f"\nLowest uplink rate the radio model produces at any distance: {rmin:.1f} Mbit/s. The RSU downlink rate is at least the uplink rate at every distance from 1 to 2,000 m.\n")

OUT.write_text("# Supplementary tables (generated by analysis/make_supplementary.py)\n\n" + "\n".join(sections))
print(f"wrote {OUT}")
