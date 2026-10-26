import pandas as pd
import numpy as np

# ---------- RSUs ----------
def load_rsus(path):
    import pandas as pd
    df = pd.read_csv(path)
    needed = {"rsu_id", "x", "y"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"rsu_data.csv missing columns: {missing}")
    # Optional specs (used if present)
    for col in ["cpu_hz", "uplink_mbps", "downlink_mbps", "range_m",
                "cores", "per_core_hz"]:
        if col not in df.columns:
            df[col] = None
    return df

# ---------- Tasks (task arrivals) ----------
def load_tasks(path, time_col="t_gen", vid_col="vehicle_id"):
    """
    Minimal expected columns:
      - vehicle_id
      - t_gen (seconds; integer or float aligned to your SUMO step length)
    Any additional task attributes are preserved and passed through.
    """
    df = pd.read_csv(path)
    # Normalize likely aliases
    col_map = {}
    if time_col not in df.columns:
        for cand in ["gen_time", "arrival_time", "time", "t", "timestamp_s"]:
            if cand in df.columns:
                col_map[cand] = time_col
                break
    if vid_col not in df.columns:
        for cand in ["veh_id", "vid", "vehicle", "node_id"]:
            if cand in df.columns:
                col_map[cand] = vid_col
                break
    if col_map:
        df = df.rename(columns=col_map)

    for must in [time_col, vid_col]:
        if must not in df.columns:
            raise ValueError(f"offload_dataset.csv must contain '{must}' column")

    # Ensure numeric time
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce").fillna(0).astype(float)
    # Index by step (rounded to nearest second if your step=1s)
    df["_step"] = df[time_col].round().astype(int)
    # Group for fast lookup
    grouped = df.groupby("_step")
    # Keep original (for logging) and a fast map: step -> DataFrame for that step
    step_map = {k: v.copy() for k, v in grouped}
    return df, step_map
