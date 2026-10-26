#!/usr/bin/env python3
"""
Convert a SUMO FCD (Floating Car Data) XML dump to the position CSV that
VECEnvOffloadCSV replays. This lets training/evaluation replay a pre-generated
trajectory instead of running SUMO live every episode (~100x faster: the 162 MB
Berlin net is parsed once by SUMO here, never again during RL).

FCD is produced by:  sumo -c <cfg> --fcd-output traj.xml ...
Output CSV columns:  time,vehicle_id,x,y,speed   (what VECEnvOffloadCSV expects)

Streaming parse (ElementTree.iterparse) so multi-GB FCD files use O(1) memory.

Usage:
    python fcd_to_csv.py <fcd.xml> <out.csv>
"""
import csv
import sys
import xml.etree.ElementTree as ET


def convert(src, dst):
    n_veh = 0
    n_steps = 0
    with open(dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "vehicle_id", "x", "y", "speed"])
        t = None
        for ev, el in ET.iterparse(src, events=("start", "end")):
            if ev == "start" and el.tag == "timestep":
                t = el.get("time")
            elif ev == "end" and el.tag == "vehicle":
                w.writerow([t, el.get("id"), el.get("x"), el.get("y"),
                            el.get("speed", "")])
                n_veh += 1
            elif ev == "end" and el.tag == "timestep":
                n_steps += 1
                el.clear()  # free the timestep subtree
    return n_steps, n_veh


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    src, dst = sys.argv[1], sys.argv[2]
    n_steps, n_veh = convert(src, dst)
    print(f"Wrote {dst}: {n_veh:,} vehicle-rows across {n_steps:,} timesteps")


if __name__ == "__main__":
    main()
