import math
import gymnasium as gym
import numpy as np

class OffloadDecisionEnv(gym.Env):
    """
    Wraps your time-stepped SUMO env to present ONE task at a time.
    Observation = features for the current task + context.
    Action = {0: local, 1: v2v, 2: rsu} with masking for invalid.
    Reward = your _compute_task_reward on the committed decision.
    """
    metadata = {"render_modes": []}

    def __init__(self, core_env, feature_fn, use_action_mask=True):
        super().__init__()
        self.e = core_env
        self.feature_fn = feature_fn
        self.use_action_mask = use_action_mask

        # ---- infer feature length once (no SUMO reset needed) ----
        dummy_task = {
            "vehicle_id": "_",
            "category": "autonomous",
            "cpu_cycles": 1e8,
            "data_bytes": 1e6,
            "deadline_s": 0.1,
        }
        def z():
            return {
                "delay_s": 0.0, "tx_up_s": 0.0, "tx_down_s": 0.0, "proc_s": 0.0,
                "ul_cqi": 0, "dl_cqi": 0, "distance_m": 0.0,
                "e_user_tx_up_j": 0.0, "e_user_rx_down_j": 0.0,
                "e_exec_comp_j": 0.0, "e_exec_tx_down_j": 0.0,
            }
        dummy_cands = {"local": z(), "v2v": None, "rsu": None}
        feats = feature_fn(dummy_task, dummy_cands, self.e).astype(np.float32)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=feats.shape, dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)

    # def _gather_tasks(self):
    #     """Advance time until we have at least one task, store tasks & candidates."""
    #     while not self._pending_tasks:
    #         # advance the core env by one time step
    #         obs, _, term, trunc, info = self.e.step(0)
    #         if term or trunc:
    #             return True  # episode ended
    #         tasks = info.get("tasks_generated", [])
    #         # precompute candidates for all tasks from this sim step
    #         for t in tasks:
    #             c = self.e._evaluate_candidates(t, self.e.traci.vehicle.getIDList())
    #             self._pending_tasks.append(t)
    #             self._pending_cands.append(c)
    #     return False

    def _gather_tasks(self):
        """
        Advance the core env until we have buffered tasks for the current core tick,
        or the episode ends. Also finalize the previous tick's decisions once we've
        consumed all tasks from that tick.
        Returns True iff the episode ended (terminated/truncated), else False.
        """
        # if we still have tasks from current tick, nothing to do
        if self._pending_tasks:
            return False

        # finalize logging for the tick we just finished (agent mode)
        if getattr(self, "_decisions_tick", None):
            try:
                self.e.agent_finalize_tick(self._decisions_tick)
            finally:
                self._decisions_tick = []

        # advance core until we actually get tasks or the episode ends
        while True:
            _, _, term, trunc, info = self.e.step(0)
            if term or trunc:
                return True
            tasks = info.get("tasks_generated", [])
            if tasks:
                self._tick_n_v2v_peers = int(info.get("n_v2v_peers", 0))
                live_ids = self.e.traci.vehicle.getIDList()
                for t in tasks:
                    c = self.e._evaluate_candidates(t, live_ids)
                    self._pending_tasks.append(t)
                    self._pending_cands.append(c)
                return False
    
    # def _next_item(self):
    #     """Pop the next task/candidates and build observation + mask."""
    #     task = self._pending_tasks.pop(0)
    #     cands = self._pending_cands.pop(0)
    #     feats = self.feature_fn(task, cands, self.e)  # np.float32 vector
    #     mask = np.array([
    #         1,                                 # local always valid
    #         1 if (cands["v2v"] is not None) else 0,
    #         1 if (cands["rsu"] is not None) else 0
    #     ], dtype=np.int8)
    #     self._current = (task, cands, feats, mask)

    # def reset(self, *, seed=None, options=None):
    #     obs, info = self.e.reset(seed=seed)
    #     self._pending_tasks = []; self._pending_cands = []; self._decisions_tick = []; self._current = None
    #     ended = self._gather_tasks()
    #     if ended:
    #         return np.zeros(self.observation_space.shape, np.float32), {}
    #     self._next_item()
    #     info = {"action_mask": self._current[3]} if self.use_action_mask else {}
    #     return self._current[2], info


    # def step(self, action):
    #     # apply mask (optional)
    #     task, cands, feats, mask = self._current
    #     if self.use_action_mask and mask[int(action)] == 0:
    #         # invalid action: heavy penalty, force local fallback
    #         action = 0

    #     cand = cands["local"] if action == 0 else cands["v2v"] if action == 1 else cands["rsu"]
    #     out = self.e._commit_candidate(cand, task)  # schedules + logs; returns the row dict
    #     self._decisions_tick.append(out)
    #     reward = self.e._compute_task_reward(out)

    #     # if more tasks in the current second, show them; else advance time
    #     if not self._pending_tasks:
    #         ended = self._gather_tasks()
    #         if ended:
    #             terminated = True
    #             # provide a final obs (zeros)
    #             obs = np.zeros(self.observation_space.shape, np.float32)
    #             info = {}
    #             return obs, reward, terminated, False, info

    #     # present next task
    #     self._next_item()
    #     obs = self._current[2]
    #     info = {"action_mask": self._current[3]} if self.use_action_mask else {}
    #     return obs, reward, False, False, info

    def _next_item(self):
        """
        Present the next task; assumes _gather_tasks has ensured _pending_* are non-empty.
        """
        # Defensive guard: if empty, try to gather (and bail cleanly if episode ended)
        if not self._pending_tasks:
            ended = self._gather_tasks()
            if ended:
                # caller should handle terminal return; here we just avoid popping
                self._current = (None, None, self._sanitize_features(np.zeros(self.observation_space.shape, np.float32)),
                                np.array([1, 0, 0], dtype=np.int8))
                return

        task = self._pending_tasks.pop(0)
        cands = self._pending_cands.pop(0)
        feats = self.feature_fn(task, cands, self.e).astype(np.float32)
        feats = self._sanitize_features(feats)
        mask = np.array([
            1,
            1 if (cands.get("v2v") is not None) else 0,
            1 if (cands.get("rsu") is not None) else 0
        ], dtype=np.int8)
        self._current = (task, cands, feats, mask)

    def reset(self, *, seed=None, options=None):
        obs, info = self.e.reset(seed=seed)
        self._pending_tasks = []
        self._pending_cands = []
        self._decisions_tick = []
        self._current = None

        ended = self._gather_tasks()  # blocks until we have tasks or episode ended
        if ended:
            return self._sanitize_features(np.zeros(self.observation_space.shape, np.float32)), {}

        self._next_item()
        info = {"action_mask": self._current[3]} if self.use_action_mask else {}
        return self._current[2], info

    def step(self, action):
        # Guard: _current is None when the episode had 0 tasks (e.g. sparse night fragment
        # with no vehicles near the sampled RSUs). Return done immediately.
        if self._current is None:
            obs = self._sanitize_features(np.zeros(self.observation_space.shape, np.float32))
            return obs, 0.0, True, False, {}
        # use current item
        task, cands, feats, mask = self._current
        invalid_action_penalty = 0.0
        if self.use_action_mask and mask[int(action)] == 0:
            invalid_action_penalty = -3.0  # Penalty for choosing unavailable action
            action = 0  # fallback to local

        cand = cands["local"] if action == 0 else cands["v2v"] if action == 1 else cands["rsu"]
        out = self.e._commit_candidate(cand, task)
        self._decisions_tick.append(out)
        reward, d = self.e._compute_task_reward(out, include_info=True)
        reward += invalid_action_penalty

        energy_info = {
            "energy_user_j":   float(out.get("energy_user_j",   0.0)),
            "energy_system_j": float(out.get("energy_system_j", 0.0)),
            "rsu_queue_s":   float(cands["rsu"].get("queue_s", 0.0)) if cands.get("rsu") is not None else 0.0,
            "local_queue_s": float(cands["local"].get("queue_s", 0.0)) if cands.get("local") is not None else 0.0,
        }

        # If we still have tasks buffered for this core tick, present next one.
        if self._pending_tasks:
            self._next_item()
            info = {"action_mask": self._current[3]} if self.use_action_mask else {}
            info["task_delay_s"] = d["task_delay_s"]
            info["n_v2v_peers"] = getattr(self, "_tick_n_v2v_peers", 0)
            info.update(energy_info)
            return self._current[2], reward, False, False, info

        # Otherwise, finalize this tick + advance core until next batch or episode end.
        ended = self._gather_tasks()
        if ended:
            # flush step summary for this last tick if any decisions remain (safety)
            if self._decisions_tick:
                self.e.agent_finalize_tick(self._decisions_tick)
                self._decisions_tick = []
            zero_obs = self._sanitize_features(np.zeros(self.observation_space.shape, np.float32))
            info = {"task_delay_s": d["task_delay_s"]}
            info["n_v2v_peers"] = getattr(self, "_tick_n_v2v_peers", 0)
            info.update(energy_info)
            return zero_obs, reward, True, False, info

        # Now we have fresh tasks buffered; present next item
        self._next_item()
        info = {"action_mask": self._current[3]} if self.use_action_mask else {}
        info["task_delay_s"] = d["task_delay_s"]
        info["n_v2v_peers"] = getattr(self, "_tick_n_v2v_peers", 0)
        info.update(energy_info)
        return self._current[2], reward, False, False, info
    
    # inside OffloadDecisionEnv
    def _sanitize_features(self, feats, clip=1e6):
        import numpy as np
        x = np.asarray(feats, dtype=np.float32)
        # replace NaN with 0, +inf with +clip, -inf with -clip, then clamp
        x = np.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
        np.clip(x, -clip, clip, out=x)
        return x

# in envs/offload_task_env.py (or wherever build_features lives)

# ============================================================================
# STATE SPACE V1 (Original - includes oracular features)
# ============================================================================
# Features per candidate (9): delay_s, tx_time, proc_s, ul_cqi, dl_cqi,
#                             distance_m, e_user, e_exec_comp, e_exec_tx
# Total: 7 base + 27 candidate = 34 features

def build_features(task, cands, env):
    """Original v1 feature function - includes oracular delay/energy predictions."""
    def nzlog(x):
        return np.log10(max(1.0, float(x)))

    def f32(v, cap=1e6):
        v = float(v)
        if not np.isfinite(v):
            return float(np.sign(v) * cap if np.isfinite(np.sign(v)) else 0.0)
        return v

    cat = task["category"]
    cat_onehot = {"safety":[1,0,0], "autonomous":[0,1,0], "infotainment":[0,0,1]}.get(cat,[0,1,0])

    base = [
        nzlog(task["cpu_cycles"]),
        nzlog(task["data_bytes"]),
        float(task["deadline_s"]),
        env.result_size_ratio
    ] + cat_onehot

    def opt_feats(c):
        if c is None:
            return [0,0,0, 0,0,0, 0,0,0]
        return [
            f32(c.get("delay_s", 0.0)),
            f32(c.get("tx_up_s", 0.0) + c.get("tx_down_s", 0.0)),
            f32(c.get("proc_s", 0.0)),
            f32(c.get("ul_cqi", 0.0), cap=100.0),
            f32(c.get("dl_cqi", 0.0), cap=100.0),
            f32(c.get("distance_m", 0.0)),
            f32(c.get("e_user_tx_up_j", 0.0) + c.get("e_user_rx_down_j", 0.0)),
            f32(c.get("e_exec_comp_j", 0.0)),
            f32(c.get("e_exec_tx_down_j", 0.0)),
        ]

    feats = np.array(base + opt_feats(cands["local"]) + opt_feats(cands["v2v"]) + opt_feats(cands["rsu"]), dtype=np.float32)
    # coarse scaling
    feats[0:2] = (feats[0:2] - 8.0) / 2.0     # log10 cpu/data ~ [8..10]
    feats[2]   = feats[2] * 10.0              # deadlines ~0..1 -> 0..10
    # final sanitize (belt-and-suspenders)
    feats = np.nan_to_num(feats, nan=0.0, posinf=1e6, neginf=-1e6)
    return feats

# Alias for backwards compatibility
build_features_v1 = build_features


# ============================================================================
# STATE SPACE V2 (Realistic - only currently measurable features)
# ============================================================================
# Removed oracular features: delay_s, dl_cqi, tx_down_s, proc_s, e_exec_*
# Features per candidate (5): valid, distance_m, ul_cqi, tx_up_s, e_user_tx_up_j
# Total: 7 base + 15 candidate = 22 features

def build_features_v2(task, cands, env):
    """
    V2 feature function - only uses currently measurable/realistic features.

    Removed (oracular/unrealistic):
        - delay_s: Perfect prediction of total completion time
        - dl_cqi: Based on predicted future position
        - tx_down_s: Based on future channel conditions
        - proc_s: Exact processing time from queue simulation
        - e_exec_comp_j: Exact energy from perfect proc_s
        - e_exec_tx_down_j: Based on future DL conditions

    Kept (measurable now):
        - distance_m: Current distance is measurable
        - ul_cqi: Current uplink channel quality
        - tx_up_s: Estimate from current UL conditions
        - e_user_tx_up_j: Based on current UL estimate
        - valid: Whether candidate exists (action mask info)
    """
    def nzlog(x):
        return np.log10(max(1.0, float(x)))

    def f32(v, cap=1e6):
        v = float(v)
        if not np.isfinite(v):
            return float(np.sign(v) * cap if np.isfinite(np.sign(v)) else 0.0)
        return v

    cat = task["category"]
    cat_onehot = {"safety":[1,0,0], "autonomous":[0,1,0], "infotainment":[0,0,1]}.get(cat,[0,1,0])

    # Base features (7): same as v1
    base = [
        nzlog(task["cpu_cycles"]),
        nzlog(task["data_bytes"]),
        float(task["deadline_s"]),
        env.result_size_ratio
    ] + cat_onehot

    def opt_feats_v2(c):
        """Per-candidate features (5): valid, distance, ul_cqi, tx_up_s, e_user_tx_up"""
        if c is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        return [
            1.0,  # valid indicator (candidate exists)
            f32(c.get("distance_m", 0.0)),           # Current distance (measurable)
            f32(c.get("ul_cqi", 0.0), cap=15.0),     # Current UL channel quality
            f32(c.get("tx_up_s", 0.0)),              # UL time from current conditions
            f32(c.get("e_user_tx_up_j", 0.0)),       # User TX energy (UL only)
        ]

    feats = np.array(
        base + opt_feats_v2(cands["local"]) + opt_feats_v2(cands["v2v"]) + opt_feats_v2(cands["rsu"]),
        dtype=np.float32
    )

    # Scaling (same base scaling as v1)
    feats[0:2] = (feats[0:2] - 8.0) / 2.0     # log10 cpu/data ~ [8..10]
    feats[2]   = feats[2] * 10.0              # deadlines ~0..1 -> 0..10

    # Scale distance (typically 0-500m for V2V, 0-1000m for RSU)
    # Indices: 7+1=8 (local), 7+5+1=13 (v2v), 7+10+1=18 (rsu)
    feats[8] = feats[8] / 100.0   # local distance (always 0)
    feats[13] = feats[13] / 100.0  # v2v distance
    feats[18] = feats[18] / 100.0  # rsu distance

    # Scale CQI (0-15)
    # Indices: 7+2=9 (local), 7+5+2=14 (v2v), 7+10+2=19 (rsu)
    feats[9] = feats[9] / 15.0    # local cqi (always 0)
    feats[14] = feats[14] / 15.0   # v2v cqi
    feats[19] = feats[19] / 15.0   # rsu cqi

    # final sanitize
    feats = np.nan_to_num(feats, nan=0.0, posinf=1e6, neginf=-1e6)
    return feats


# Feature index documentation for v2
# Base (indices 0-6):
#   0: log(cpu_cycles) scaled
#   1: log(data_bytes) scaled
#   2: deadline_s scaled
#   3: result_size_ratio
#   4-6: category one-hot [safety, autonomous, infotainment]
#
# Local candidate (indices 7-11):
#   7: valid (always 1.0 for local)
#   8: distance_m (always 0 for local)
#   9: ul_cqi (always 0 for local)
#   10: tx_up_s (always 0 for local)
#   11: e_user_tx_up_j (always 0 for local)
#
# V2V candidate (indices 12-16):
#   12: valid (1.0 if peer exists, 0.0 otherwise)
#   13: distance_m (current distance to peer)
#   14: ul_cqi (current UL channel quality)
#   15: tx_up_s (UL time estimate)
#   16: e_user_tx_up_j (user TX energy)
#
# RSU candidate (indices 17-21):
#   17: valid (1.0 if RSU in range, 0.0 otherwise)
#   18: distance_m (current distance to RSU)
#   19: ul_cqi (current UL channel quality)
#   20: tx_up_s (UL time estimate)
#   21: e_user_tx_up_j (user TX energy)


# ============================================================================
# STATE SPACE V3 (v2 + traffic context)
# ============================================================================
# Added: nearby_vehicle_count, vehicle_speed
# Features per candidate (5): same as v2
# Total: 7 base + 15 candidate + 2 context = 24 features

def build_features_v3(task, cands, env):
    """
    V3 feature function - v2 + traffic context.

    Added over v2:
        - nearby_vehicle_count: vehicles within V2V range (traffic density proxy)
        - vehicle_speed: current speed in m/s (mobility context)
    """
    def nzlog(x):
        return np.log10(max(1.0, float(x)))

    def f32(v, cap=1e6):
        v = float(v)
        if not np.isfinite(v):
            return float(np.sign(v) * cap if np.isfinite(np.sign(v)) else 0.0)
        return v

    cat = task["category"]
    cat_onehot = {"safety":[1,0,0], "autonomous":[0,1,0], "infotainment":[0,0,1]}.get(cat,[0,1,0])

    base = [
        nzlog(task["cpu_cycles"]),
        nzlog(task["data_bytes"]),
        float(task["deadline_s"]),
        env.result_size_ratio
    ] + cat_onehot

    def opt_feats(c):
        if c is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        return [
            1.0,
            f32(c.get("distance_m", 0.0)),
            f32(c.get("ul_cqi", 0.0), cap=15.0),
            f32(c.get("tx_up_s", 0.0)),
            f32(c.get("e_user_tx_up_j", 0.0)),
        ]

    # Traffic context features
    vid = task["vehicle_id"]
    nearby_count = 0.0
    speed_ms = 0.0
    try:
        src_x, src_y = env.traci.vehicle.getPosition(vid)
        range_sq = env.v2v_range_m * env.v2v_range_m
        for v in env.traci.vehicle.getIDList():
            if v == vid:
                continue
            vx, vy = env.traci.vehicle.getPosition(v)
            dx, dy = src_x - vx, src_y - vy
            if dx * dx + dy * dy <= range_sq:
                nearby_count += 1
        speed_ms = float(env.traci.vehicle.getSpeed(vid))
    except Exception:
        pass

    feats = np.array(
        base
        + opt_feats(cands["local"])
        + opt_feats(cands["v2v"])
        + opt_feats(cands["rsu"])
        + [nearby_count, speed_ms],
        dtype=np.float32
    )

    # Scaling (same base scaling as v2)
    feats[0:2] = (feats[0:2] - 8.0) / 2.0     # log10 cpu/data ~ [8..10]
    feats[2]   = feats[2] * 10.0              # deadlines ~0..1 -> 0..10

    # Scale distance
    feats[8] = feats[8] / 100.0   # local distance (always 0)
    feats[13] = feats[13] / 100.0  # v2v distance
    feats[18] = feats[18] / 100.0  # rsu distance

    # Scale CQI (0-15)
    feats[9] = feats[9] / 15.0    # local cqi (always 0)
    feats[14] = feats[14] / 15.0   # v2v cqi
    feats[19] = feats[19] / 15.0   # rsu cqi

    # Scale context features
    feats[22] = feats[22] / 10.0   # nearby count (~0-20 -> ~0-2)
    feats[23] = feats[23] / 30.0   # speed m/s (~0-40 -> ~0-1.3)

    # final sanitize
    feats = np.nan_to_num(feats, nan=0.0, posinf=1e6, neginf=-1e6)
    return feats


# Feature index documentation for v3
# Base (indices 0-6): same as v2
# Local candidate (indices 7-11): same as v2
# V2V candidate (indices 12-16): same as v2
# RSU candidate (indices 17-21): same as v2
# Context (indices 22-23):
#   22: nearby_vehicle_count (vehicles within V2V range, scaled /10)
#   23: vehicle_speed (m/s, scaled /30)


# ============================================================================
# STATE SPACE V4 (v3 + estimated processing time per candidate)
# ============================================================================
# Added: est_proc_s for local, v2v, rsu (cpu_cycles / node_cpu_hz, idle estimate)
# Assumption: node compute capacity is broadcast via V2X service advertisements
# (realistic for C-V2X / IEEE 802.11p infrastructure).
# Features per candidate (5): same as v3 (valid, distance, ul_cqi, tx_up_s, e_tx)
# Total: 7 base + 15 candidate + 2 context + 3 est_proc = 27 features
# All v2/v3 indices preserved — new features appended after context.

def build_features_v4(task, cands, env):
    """
    V4 feature function - v3 + estimated (idle) processing time per candidate.

    Added over v3:
        - est_proc_s (local/v2v/rsu): cpu_cycles / node_cpu_hz
          This is the processing time assuming the executor is idle.
          Queue wait (proc_s - est_proc_s) is still unknown to the agent.

    Justification: RSU/vehicle compute capacity is a static advertised property
    in C-V2X and IEEE 802.11p service announcements. Unlike queue_s/proc_s, it
    does not require oracle access to the simulator's internal state.

    Feature layout (27 total):
        0-6:   base (same as v2/v3)
        7-11:  local candidate (same as v2/v3)
        12-16: v2v candidate (same as v2/v3)
        17-21: rsu candidate (same as v2/v3)
        22-23: traffic context (same as v3)
        24:    local est_proc_s
        25:    v2v est_proc_s  (0.0 if no peer)
        26:    rsu est_proc_s  (0.0 if no RSU)
    """
    def nzlog(x):
        return np.log10(max(1.0, float(x)))

    def f32(v, cap=1e6):
        v = float(v)
        if not np.isfinite(v):
            return float(np.sign(v) * cap if np.isfinite(np.sign(v)) else 0.0)
        return v

    cat = task["category"]
    cat_onehot = {"safety":[1,0,0], "autonomous":[0,1,0], "infotainment":[0,0,1]}.get(cat,[0,1,0])

    base = [
        nzlog(task["cpu_cycles"]),
        nzlog(task["data_bytes"]),
        float(task["deadline_s"]),
        env.result_size_ratio
    ] + cat_onehot

    def opt_feats(c):
        if c is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        return [
            1.0,
            f32(c.get("distance_m", 0.0)),
            f32(c.get("ul_cqi", 0.0), cap=15.0),
            f32(c.get("tx_up_s", 0.0)),
            f32(c.get("e_user_tx_up_j", 0.0)),
        ]

    # Traffic context (same as v3)
    vid = task["vehicle_id"]
    nearby_count = 0.0
    speed_ms = 0.0
    try:
        src_x, src_y = env.traci.vehicle.getPosition(vid)
        range_sq = env.v2v_range_m * env.v2v_range_m
        for v in env.traci.vehicle.getIDList():
            if v == vid:
                continue
            vx, vy = env.traci.vehicle.getPosition(v)
            dx, dy = src_x - vx, src_y - vy
            if dx * dx + dy * dy <= range_sq:
                nearby_count += 1
        speed_ms = float(env.traci.vehicle.getSpeed(vid))
    except Exception:
        pass

    # Estimated processing time per candidate (idle executor, no queue wait)
    cpu_cycles = float(task["cpu_cycles"])

    def get_est_proc_s(c, mode):
        if c is None:
            return 0.0
        node_id = c.get("node_id")
        if mode == "rsu":
            spec = getattr(env, 'rsu_spec', {}).get(node_id, {})
            cpu_hz = spec.get("cpu_hz", getattr(env, 'rsu_default_cpu_hz', 16e9))
        else:  # local or v2v vehicle
            spec = getattr(env, 'veh_spec', {}).get(node_id, {})
            cpu_hz = spec.get("per_core_hz", getattr(env, 'veh_cpu_hz_mean', 3e9))
        return cpu_cycles / max(float(cpu_hz), 1e6)

    feats = np.array(
        base
        + opt_feats(cands["local"])
        + opt_feats(cands["v2v"])
        + opt_feats(cands["rsu"])
        + [nearby_count, speed_ms]
        + [get_est_proc_s(cands["local"], "local"),
           get_est_proc_s(cands["v2v"],   "v2v"),
           get_est_proc_s(cands["rsu"],   "rsu")],
        dtype=np.float32
    )

    # Scaling (same base scaling as v3)
    feats[0:2] = (feats[0:2] - 8.0) / 2.0
    feats[2]   = feats[2] * 10.0

    # Scale distance
    feats[8]  = feats[8]  / 100.0
    feats[13] = feats[13] / 100.0
    feats[18] = feats[18] / 100.0

    # Scale CQI (0-15)
    feats[9]  = feats[9]  / 15.0
    feats[14] = feats[14] / 15.0
    feats[19] = feats[19] / 15.0

    # Scale context
    feats[22] = feats[22] / 10.0
    feats[23] = feats[23] / 30.0

    # Scale est_proc_s (raw seconds, typical range 0.003-5s → multiply by 2)
    feats[24] = feats[24] * 2.0
    feats[25] = feats[25] * 2.0
    feats[26] = feats[26] * 2.0

    feats = np.nan_to_num(feats, nan=0.0, posinf=1e6, neginf=-1e6)
    return feats


# Feature index documentation for v4
# Base (indices 0-6): same as v2/v3
# Local candidate (indices 7-11): same as v2/v3
# V2V candidate (indices 12-16): same as v2/v3
# RSU candidate (indices 17-21): same as v2/v3
# Context (indices 22-23): same as v3
# Est proc (indices 24-26):
#   24: local est_proc_s (cpu_cycles / veh_cpu_hz, scaled *2)
#   25: v2v est_proc_s   (cpu_cycles / peer_cpu_hz, scaled *2; 0 if no peer)
#   26: rsu est_proc_s   (cpu_cycles / rsu_cpu_hz, scaled *2; 0 if no RSU)


# ============================================================================
# STATE SPACE V5 (v4 + queue wait times)
# ============================================================================
# Added: current queue wait at V2V peer and RSU (from heap simulation).
# This is the key missing signal — the agent can now detect when peers/RSU
# are congested and route away from them.
# Total: 27 (v4) + 2 (queue waits) = 29 features
# All v4 indices preserved — new features appended after est_proc.

def build_features_v5(task, cands, env):
    """
    V5 feature function - v4 + queue wait times for V2V peer and RSU.

    Added over v4:
        - v2v_queue_wait_s: current queue wait at the selected peer vehicle.
          Derived from candidate['queue_s'] — the time the task would wait
          before the peer's CPU is free. Reflects actual contention.
        - rsu_queue_wait_s: current queue wait at the RSU (same source).

    Both are already computed by _evaluate_candidates() via _schedule_multi_on_*
    and stored in candidate['queue_s']. No oracle access required: the queue
    state is observable at decision time.

    Feature layout (29 total):
        0-26:  same as v4
        27:    v2v_queue_wait_s (seconds; 0 if V2V unavailable; scaled *2)
        28:    rsu_queue_wait_s (seconds; 0 if RSU unavailable; scaled *2)
    """
    # Build v4 features as base
    feats_v4 = build_features_v4(task, cands, env)

    v2v_queue_wait = float(cands["v2v"].get("queue_s", 0.0)) if cands.get("v2v") is not None else 0.0
    rsu_queue_wait = float(cands["rsu"].get("queue_s", 0.0)) if cands.get("rsu") is not None else 0.0

    extra = np.array([v2v_queue_wait * 2.0, rsu_queue_wait * 2.0], dtype=np.float32)
    feats = np.concatenate([feats_v4, extra])
    feats = np.nan_to_num(feats, nan=0.0, posinf=1e6, neginf=-1e6)
    return feats


# Feature index documentation for v5
# Indices 0-26: identical to v4
# Queue waits (indices 27-28):
#   27: v2v_queue_wait_s (seconds agent would wait at peer CPU; scaled *2; 0 if unavailable)
#   28: rsu_queue_wait_s (seconds agent would wait at RSU; scaled *2; 0 if unavailable)


def build_features_v6(task, cands, env):
    """
    V6 feature function - v5 + local queue wait time.

    Added over v5:
        - local_queue_wait_s: current queue wait on the source vehicle's own CPU.
          Derived from cands['local']['queue_s'] — how long the task would wait
          before the local CPU is free. Closes the asymmetry where v5 exposed
          V2V and RSU queue depths but not the local backlog.

    Feature layout (30 total):
        0-28:  same as v5
        29:    local_queue_wait_s (seconds; scaled *2)
    """
    feats_v5 = build_features_v5(task, cands, env)
    local_queue_wait = float(cands["local"].get("queue_s", 0.0)) if cands.get("local") is not None else 0.0
    extra = np.array([local_queue_wait * 2.0], dtype=np.float32)
    feats = np.concatenate([feats_v5, extra])
    feats = np.nan_to_num(feats, nan=0.0, posinf=1e6, neginf=-1e6)
    return feats


# Feature index documentation for v6
# Indices 0-28: identical to v5
# Local queue wait (index 29):
#   29: local_queue_wait_s (seconds agent would wait at own CPU; scaled *2)