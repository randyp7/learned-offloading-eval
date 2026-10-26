# utils/sumo_utils.py
import os, time
import traci
from sumolib import checkBinary
import random
import numpy as np

def _has_remote_port_arg(cmd_list):
    # Always compare strings (WindowsPath, ints, etc. can be in the list)
    for tok in map(str, cmd_list):
        if tok == "-p" or tok == "--remote-port" or tok.startswith("--remote-port="):
            return True
    return False

def init_sumo(cfg_path, gui=False, seed=None, step_length=1.0, label=None, add_remote_port=True, begin=None, end=None, scale=None):
    """
    Start SUMO/TraCI with a unique connection label.
    Adds '--remote-port 0' only if it's not already present.

    Args:
        cfg_path: Path to SUMO config file
        gui: Whether to use SUMO GUI
        seed: Random seed
        step_length: Simulation step length in seconds
        label: TraCI connection label
        add_remote_port: Whether to add --remote-port 0
        begin: Simulation begin time in seconds (overrides config file)
        end: Simulation end time in seconds (overrides config file)

    Returns: traci.Connection for this label.
    """
    sumo_bin = checkBinary("sumo-gui" if gui else "sumo")

    # Ensure paths/values are strings in the cmd list
    cfg_path = str(cfg_path)
    step_length = str(step_length)

    cmd = [
        sumo_bin, "-c", cfg_path,
        "--step-length", step_length,
        "--no-warnings", "true",
    ]
    if seed is not None:
        cmd += ["--seed", str(int(seed))]
    if scale is not None:
        cmd += ["--scale", str(float(scale))]

    # Add begin/end time overrides if specified
    if begin is not None:
        cmd += ["--begin", str(int(begin))]
    if end is not None:
        cmd += ["--end", str(int(end))]

    # Add auto-port ONLY if missing
    if add_remote_port and not _has_remote_port_arg(cmd):
        cmd += ["--remote-port", "0"]

    # Unique label if none provided
    if not label:
        label = f"env-{os.getpid()}-{int(time.time()*1e6)%1_000_000}"

    # OPTIONAL: uncomment once to verify final args (helps if dup port errors reappear)
    # print("SUMO CMD:", " ".join(map(str, cmd)), "LABEL:", label)

    # Do NOT pass 'port=' here; only the labeled connection
    traci.start(list(map(str, cmd)), label=label)
    return traci.getConnection(label)

def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
