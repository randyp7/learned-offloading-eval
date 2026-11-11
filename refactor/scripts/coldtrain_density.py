"""Frame-wise vehicle density of the cold-start training fragments.

For every cold-start training fragment (20 frames) counts the vehicles present
in each frame and the vehicles present at the same clock second in the
continuous 24-hour warm-start trace, and writes one row per fragment frame.
Read-only with respect to all inputs.

usage: python coldtrain_density.py <coldtrain_fragment_dir> <warm_fcd_csv> <out_csv>
  coldtrain_fragment_dir: folder holding the frag_t<seconds>*.csv fragments
  warm_fcd_csv:           the continuous trace (e.g. corridor_24h_seed42.csv)
"""
import glob, re, sys
from pathlib import Path
import pandas as pd

frag_dir, warm_csv, out_csv = sys.argv[1:4]
if Path(out_csv).exists():
    sys.exit(f"refusing to overwrite {out_csv}")
w = pd.read_csv(warm_csv, usecols=["time"])["time"].round().astype(int).value_counts()
rows = []
for f in glob.glob(str(Path(frag_dir) / "frag_t*.csv")):
    T = int(re.search(r"frag_t(\d+)", f).group(1))
    c = pd.read_csv(f, usecols=["time"])["time"].round().astype(int).value_counts().sort_index()
    for k, (t, n) in enumerate(c.items()):
        rows.append((T, k, t, n, int(w.get(t, 0))))
df = pd.DataFrame(rows, columns=["T", "k", "t", "cold", "warm"])
Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_csv, index=False)


def per(lo, hi, lab):
    s = df[(df["T"] >= lo * 3600) & (df["T"] < hi * 3600)]
    if s.empty:
        return
    print(f"{lab:12s} fragments={s['T'].nunique():5d}  cold/warm vehicles {s.cold.sum() / max(s.warm.sum(), 1) * 100:5.1f}%")


per(0, 24, "all day"); per(0, 4, "night 00-04"); per(8, 9, "peak 08-09"); per(16, 18, "peak 16-18"); per(20, 24, "eve 20-24")
print("by frame (mean vehicles, cold / warm):")
print(df.groupby("k")[["cold", "warm"]].mean().round(1).T.to_string())
