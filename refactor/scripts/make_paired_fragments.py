#!/usr/bin/env python3
"""
Paired warm/cold evaluation fragments for the cold-start arm (CAiSE 2027, RQ1).

For each evaluation window we draw N start times T_i (seeded) and build two
fragment directories with IDENTICAL file names:

  warm_<win>/frag_t<T>.csv : the first 100 non-empty frames at/after T cut from
                             the continuous 24 h corridor trace (warm road);
  cold_<win>/frag_t<T>.csv : the first 100 non-empty frames of a SUMO run that
                             STARTS at T on an empty network (--begin T), i.e.
                             the original per-episode cold-start protocol.

VECEnvOffloadCSV fragment mode lists *.csv sorted by name and picks the
fragment for each episode from the seeded episode RNG, so with equal file
names both arms replay the same sequence of start times; the only difference
between the arms is the state of the road at T.

Frames are counted as NON-EMPTY timesteps because the replay environment
skips timesteps without vehicles (its "100-step episode" is 100 such frames).

Sub-commands
  warm  --trace <24h csv> --out <dir> [--windows ...] [--n 24] [--seed 7000]
        also writes starts_<win>.json (the T_i) and, with --slices, the plain
        window slices corridor_<win>.csv used for the main-results re-runs.
  cold  --starts starts_<win>.json --cfg <sumocfg> --out <dir> [--sumo <bin>]
  stats --dir <fragment dir>   -> per-fragment mean vehicles per frame (CSV)
"""
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

WINDOWS = {  # seconds of day [t0, t1)
    "00-04": (0, 14400),
    "08-09": (28800, 32400),
    "16-18": (57600, 64800),
    "20-24": (72000, 86400),
    "24h":   (0, 86400),
}
N_FRAMES = 100          # episode length in non-empty frames
COLD_HORIZON_S = 1800   # max simulated seconds to collect N_FRAMES after a cold start
MARGIN_S = 2000         # keep T_i at least this far from the window end


def draw_starts(win, n, seed):
    t0, t1 = WINDOWS[win]
    rng = np.random.default_rng(seed + sum(map(ord, win)))
    return sorted(int(t) for t in rng.integers(t0, t1 - MARGIN_S, size=n))


def first_n_frames(df, t_start, n=N_FRAMES):
    """Rows of the first n distinct timesteps with time >= t_start."""
    d = df[df["time"] >= t_start]
    steps = np.sort(d["time"].unique())[:n]
    if len(steps) < n:
        return None
    return d[d["time"].isin(steps)].sort_values(["time", "vehicle_id"])


def cmd_warm(a):
    print(f"loading {a.trace} ...", flush=True)
    df = pd.read_csv(a.trace, dtype={"time": "float64", "vehicle_id": "str", "x": "float32", "y": "float32", "speed": "float32"})
    df["time"] = df["time"].round().astype("int64")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for win in a.windows:
        starts = draw_starts(win, a.n, a.seed)
        d = out / f"warm_{win}"; d.mkdir(exist_ok=True)
        kept = []
        for T in starts:
            fr = first_n_frames(df, T)
            if fr is None:
                print(f"  {win} T={T}: fewer than {N_FRAMES} frames before end of trace, skipped"); continue
            fr.to_csv(d / f"frag_t{T:05d}.csv", index=False); kept.append(T)
        (out / f"starts_{win}.json").write_text(json.dumps(kept))
        print(f"{win}: {len(kept)} warm fragments -> {d}")
        if a.slices and win != "24h":
            t0, t1 = WINDOWS[win]
            sl = df[(df["time"] >= t0) & (df["time"] < t1)]
            sl.to_csv(out / f"corridor_{win}_seed42.csv", index=False)
            print(f"{win}: slice {len(sl):,} rows -> corridor_{win}_seed42.csv")


def run_sumo_cold(sumo, cfg, T, seed, out_csv, n_frames=N_FRAMES, horizon=COLD_HORIZON_S):
    """SUMO from an empty network at T; keep the first n_frames non-empty frames."""
    conv = Path(__file__).with_name("fcd_to_csv.py")
    with tempfile.TemporaryDirectory() as td:
        fcd = os.path.join(td, "fcd.xml"); raw = os.path.join(td, "raw.csv")
        cmd = [sumo, "-c", cfg, "--seed", str(seed), "--step-length", "1",
               "--begin", str(T), "--end", str(T + horizon),
               "--output-prefix", "", "--fcd-output", fcd,
               "--no-step-log", "true", "--no-warnings", "true"]
        subprocess.run(cmd, check=True, cwd=str(Path(cfg).parent), capture_output=True)
        subprocess.run([sys.executable, str(conv), fcd, raw], check=True, capture_output=True)
        df = pd.read_csv(raw)
    df["time"] = df["time"].round().astype("int64")
    fr = first_n_frames(df, T, n_frames)
    if fr is None:
        raise RuntimeError(f"cold run at T={T}: only {df['time'].nunique()} non-empty frames in {horizon}s")
    fr.to_csv(out_csv, index=False)
    return fr


def cmd_coldtrain(a):
    """Cold-start TRAINING fragments (the original study's protocol): SUMO launched fresh at
    a random time of day, first `--frames` non-empty frames kept. `--n` start times are drawn
    with a seed; `--shard i/n` lets several jobs split the list. Files are named so that the
    replay env's fragment mode samples one at random per episode."""
    rng = np.random.default_rng(a.seed)
    starts = sorted(int(t) for t in rng.integers(0, 86400 - 600, size=a.n))
    i, n = map(int, a.shard.split("/")); mine = starts[i::n]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"starts_all_seed{a.seed}.json").write_text(json.dumps(starts))
    cfg = str(Path(a.cfg).resolve()); skipped = []
    for k, T in enumerate(mine):
        f = out / f"frag_t{T:05d}.csv"
        if f.exists() and f.stat().st_size > 0:
            continue
        fr = None
        for horizon in (a.horizon, a.horizon * 5, a.horizon * 15):   # night starts may see no vehicle for minutes
            try:
                fr = run_sumo_cold(a.sumo, cfg, T, a.sumo_seed + T, f, n_frames=a.frames, horizon=horizon); break
            except RuntimeError:
                continue
        if fr is None:   # empty road for the whole horizon: no episode is possible, as in the original protocol
            skipped.append(T); print(f"[shard {i}/{n} {k+1}/{len(mine)}] T={T}: no vehicle within {a.horizon*15}s, skipped", flush=True); continue
        print(f"[shard {i}/{n} {k+1}/{len(mine)}] T={T}: {a.frames} frames, mean veh/frame {len(fr)/a.frames:.1f}", flush=True)
    (out / f"skipped_shard{i}of{n}.json").write_text(json.dumps(skipped))
    print(f"shard {i}/{n}: {len(mine)-len(skipped)} fragments, {len(skipped)} start times skipped")


def cmd_cold(a):
    starts = json.loads(Path(a.starts).read_text())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = str(Path(a.cfg).resolve())
    for i, T in enumerate(starts):
        f = out / f"frag_t{T:05d}.csv"
        if f.exists() and f.stat().st_size > 0:
            print(f"[{i+1}/{len(starts)}] T={T} exists, skip"); continue
        fr = run_sumo_cold(a.sumo, cfg, T, a.seed, f)
        span = int(fr["time"].max() - fr["time"].min() + 1)
        print(f"[{i+1}/{len(starts)}] T={T}: {N_FRAMES} frames span {span}s, "
              f"mean veh/frame {len(fr)/N_FRAMES:.1f}", flush=True)


def cmd_stats(a):
    rows = []
    for f in sorted(Path(a.dir).glob("frag_t*.csv")):
        df = pd.read_csv(f, usecols=["time", "vehicle_id"])
        g = df.groupby("time").size()
        rows.append(dict(fragment=f.name, T=int(f.stem.split("_t")[1]), n_frames=len(g),
                         span_s=int(g.index.max() - g.index.min() + 1),
                         veh_per_frame_mean=round(g.mean(), 2),
                         veh_first10=round(g.iloc[:10].mean(), 2), veh_last10=round(g.iloc[-10:].mean(), 2)))
    out = pd.DataFrame(rows); out.to_csv(Path(a.dir) / "_stats.csv", index=False)
    print(out.to_string(index=False)); print("MEAN veh/frame:", round(out.veh_per_frame_mean.mean(), 2))


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    w = sp.add_parser("warm"); w.add_argument("--trace", required=True); w.add_argument("--out", required=True)
    w.add_argument("--windows", nargs="*", default=list(WINDOWS)); w.add_argument("--n", type=int, default=24)
    w.add_argument("--seed", type=int, default=7000); w.add_argument("--slices", action="store_true")
    c = sp.add_parser("cold"); c.add_argument("--starts", required=True); c.add_argument("--cfg", required=True)
    c.add_argument("--out", required=True); c.add_argument("--sumo", default="sumo"); c.add_argument("--seed", type=int, default=42)
    s = sp.add_parser("stats"); s.add_argument("--dir", required=True)
    ct = sp.add_parser("coldtrain"); ct.add_argument("--cfg", required=True); ct.add_argument("--out", required=True)
    ct.add_argument("--n", type=int, default=2000); ct.add_argument("--frames", type=int, default=20)
    ct.add_argument("--horizon", type=int, default=120); ct.add_argument("--seed", type=int, default=4242)
    ct.add_argument("--sumo-seed", type=int, default=42); ct.add_argument("--shard", default="0/1"); ct.add_argument("--sumo", default="sumo")
    a = ap.parse_args()
    {"warm": cmd_warm, "cold": cmd_cold, "stats": cmd_stats, "coldtrain": cmd_coldtrain}[a.cmd](a)


if __name__ == "__main__":
    main()
