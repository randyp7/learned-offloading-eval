import gymnasium as gym
import numpy as np, random, hashlib
from gymnasium import spaces
from pathlib import Path
from typing import Dict, Tuple, Optional
import math
import heapq
from math import isfinite
import os, time
from collections import deque
from numpy.random import SeedSequence, default_rng

from utils.sumo_utils import init_sumo, set_global_seeds
from utils.data_utils import load_rsus


from net.nrv2x_lite import NRRadio, NRParams, one_nr_sidelink_hop
from config import (SEED,NR_CARRIER_HZ,
    NR_P_TX_VEH_W, NR_P_RX_VEH_W, NR_P_TX_RSU_W,
    NR_P_RX_RSU_W, NR_NOISE_FIG_VEH_DB, NR_NOISE_FIG_RSU_DB,
    NR_SCS_KHZ, NR_SLOT_SYMBOLS, NR_SLOT_DURATION_S, NR_BANDWIDTH_PRB_VEH, NR_BANDWIDTH_PRB_RSU, NR_OVERHEAD_DATA, NR_HARQ_REPETITIONS,
    NR_SCENARIO, NR_LOS_MODEL, NR_SHADOW_SIGMA_DB_LOS, NR_SHADOW_SIGMA_DB_NLOS, NR_CQI_THRESH_DB,
    VEH_CPU_CLASSES, VEH_CPU_CLASS_WEIGHTS,
    RSU_CORES_DEFAULT, RSU_PER_CORE_HZ_DEFAULT, PARALLEL, PARALLEL_ETA_VEH,
    PARALLEL_ETA_RSU, THREAD_OVERHEAD_S, REWARD_CFG
)


def _trunc_pos(rng, mean, std, min_val=1e-9):
    x = rng.normal(mean, std)
    while x <= min_val:
        x = rng.normal(mean, std)
    return x

def _bytes_per_sec(mbps: float) -> float:
    return max(1e-9, float(mbps)) * 1e6 / 8.0

def _trunc_norm_pos(rng, mean, std, min_val):
    x = rng.normal(mean, std)
    while x < min_val:
        x = rng.normal(mean, std)
    return float(x)

def _sample_veh_cpu(rng):
    """Sample vehicle cores and per-core frequency from the class distribution."""
    idx = rng.choice(len(VEH_CPU_CLASSES), p=VEH_CPU_CLASS_WEIGHTS)
    _, cores, hz_min, hz_max, _ = VEH_CPU_CLASSES[idx]
    per_core_hz = float(rng.uniform(hz_min, hz_max))
    return cores, per_core_hz

def _u32_stable(x: str) -> np.uint32:
    # 4-byte digest → stable uint32 across processes/platforms
    d = hashlib.blake2s(x.encode("utf-8"), digest_size=4).digest()
    return np.frombuffer(d, dtype=np.uint32)[0]

def _rng_for(self, *keys, base=None):
    """
    Stateless RNG keyed by (base seed + keys) with *stable* hashing.
    Guaranteed identical across processes (CSV vs live SUMO).
    """
    if base is None:
        base = int(getattr(self, "task_seed_base", 12345))
    # optional salt (set to 0 for exact reproducibility across episodes/workers)
    salt = int(getattr(self, "_episode_salt", 0)) ^ int(getattr(self, "worker_id", 0))
    spawn_key = [_u32_stable(str(k)) for k in (*keys, salt)]
    ss = SeedSequence(base, spawn_key=spawn_key)
    return default_rng(ss)



class VECEnvOffload(gym.Env):
    """
    Minimal offloading env:
      - Generates tasks on live vehicles (Poisson)
      - Chooses Local / V2V / RSU greedily by total delay (tx + queue + proc)
      - Tracks simple queues via 'available_at' per compute node
      - Logs breakdown to task_offloading_log.csv
      - Reward: 0 (logging-first baseline). Easy to switch later.
    """
    metadata = {"render_modes": ["none"], "render_fps": 1}

    def __init__(self,
                 sumo_cfg: Path,
                 rsu_csv: Path,
                 log_dir: Path,
                 # Observation
                 obs_max_veh: int = 50,
                 obs_per_veh: int = 3,
                 # Sim
                 max_steps: int = 3600,
                 step_length_s: float = 1.0,
                 seed: int = 42,
                 gui: bool = False,
                 begin_time: Optional[int] = None,
                 end_time: Optional[int] = None,
                 # Task gen
                 lambda_per_veh_per_sec: float = 0.02,
                 category_mix: Dict[str, float] = None,
                 category_profiles: Dict[str, Dict[str, Tuple[float, float]]] = None,
                 # Offloading knobs
                 veh_cpu_hz_mean: float = 3.0e9,   # legacy, unused
                 veh_cpu_hz_std: float  = 0.5e9,   # legacy, unused
                 veh_cpu_hz_min: float  = 1.5e9,   # legacy, unused
                 veh_per_core_hz_mean: float = None,  # None → VEH_PER_CORE_HZ_MEAN from config.py
                 veh_per_core_hz_std:  float = None,  # None → VEH_PER_CORE_HZ_STD
                 veh_per_core_hz_min:  float = None,  # None → VEH_PER_CORE_HZ_MIN
                 v2v_range_m: float = 80.0,
                 v2v_rate_mbps: float = 50.0,
                 rsu_default_cpu_hz: float = 20e9,
                 rsu_default_uplink_mbps: float = 30.0,
                 rsu_default_downlink_mbps: float = 60.0,
                 rsu_default_range_m: float = 250.0,
                 energy_per_cycle_j_veh: float = 1e-9,
                 energy_per_cycle_j_rsu: float = 1.5e-10,
                 p_tx_v2v_w: float = 1.0,
                 p_tx_rsu_w: float = 1.2,
                 offload_policy: str = "greedy",
                 v2v_rate_mbps_ul: float = 50.0,
                 v2v_rate_mbps_dl: float = 50.0,
                 result_size_ratio: float = 0.20,
                 p_rx_veh_w: float = 0.8,
                 p_rx_rsu_w: float = 1.0,
                 traci_label=None,
                 task_seed_base: int = 12345,
                 task_rng_mode: str = "stateless",  # or "episode"
                 demand_scale: float = None,  # SUMO --scale factor (None = default 1.0)
                #  worker_id=rank,
                # evaluation
                include_drl_overhead=False,
                 # RSU profile overrides (takes precedence over config.py globals when set)
                 rsu_default_cores: int = None,
                 rsu_default_per_core_hz: float = None,
                 # Reward type override
                 reward_type: str = None,  # None=current; 'deadline_only'; 'latency_only'
                 # Peer selection strategy for V2V
                 peer_selection: str = "best",  # "best" (min finish time) or "nearest" (geographic)
                 ):
        super().__init__()
        # Paths
        self.sumo_cfg = Path(sumo_cfg)
        self.rsu_csv = Path(rsu_csv)
        self.log_dir = Path(log_dir); self.log_dir.mkdir(parents=True, exist_ok=True)

        # unique label for this env instance; can pass from make_env(rank)
        self.traci_label = traci_label or f"env-{os.getpid()}-{int(time.time()*1e6)%1_000_000}"
        self.demand_scale = demand_scale

        self._a = self._c = 0

        # Spaces
        self.obs_max_veh = obs_max_veh
        self.obs_per_veh = obs_per_veh
        self.action_space = spaces.Discrete(1)  # no-op for now (auto decisions)
        self.obs_task_features = 5
        self.observation_space = spaces.Box(
            low=-1e6, high=1e6,
            shape=(self.obs_max_veh * self.obs_per_veh + self.obs_task_features,),
            dtype=np.float32
        )

        self.include_drl_overhead = include_drl_overhead

        # Sim
        self.max_steps = max_steps
        self.step_length_s = step_length_s
        self.seed_val = seed
        self.gui = gui

        # Store timeframe range for random sampling
        # If begin_time and end_time span longer than max_steps, we'll sample random windows
        self.timeframe_begin = begin_time
        self.timeframe_end = end_time
        self.begin_time = begin_time  # Will be set per-episode in reset()
        self.end_time = end_time  # Will be set per-episode in reset()

        # Task gen
        self.lambda_per_veh_per_sec = float(lambda_per_veh_per_sec)
        self.category_mix = dict(category_mix or {"autonomous": 0.3, "safety": 0.25, "infotainment": 0.45})
        self.category_profiles = category_profiles or {
            "autonomous":   {"cpu_cycles": (2.5e9, 0.5e9), "data_bytes": (1.5e6, 5e5), "deadline_s": (0.10, 0.04)},
            "safety":       {"cpu_cycles": (1.6e9, 3.5e8), "data_bytes": (8e5,  2.5e5), "deadline_s": (0.06, 0.03)},
            "infotainment": {"cpu_cycles": (9e8,  2.5e8),  "data_bytes": (6e6,  2e6),   "deadline_s": (0.50, 0.25)},
        }

        # Offload knobs
        self.veh_cpu_hz_mean = veh_cpu_hz_mean
        self.veh_cpu_hz_std  = veh_cpu_hz_std
        self.veh_cpu_hz_min  = veh_cpu_hz_min
        # veh_per_core_hz_mean/std/min args are unused: CPU is now sampled
        # from the class distribution (VEH_CPU_CLASSES) in _ensure_vehicle_spec.
        self.v2v_range_m = v2v_range_m
        self.v2v_rate_bps = _bytes_per_sec(v2v_rate_mbps)
        self.rsu_default_cpu_hz = rsu_default_cpu_hz
        self.rsu_default_range_m = rsu_default_range_m
        # Profile-level overrides (set by env_factory from config_loader profiles)
        self._rsu_default_cores = rsu_default_cores
        self._rsu_default_per_core_hz = rsu_default_per_core_hz
        self.energy_per_cycle_j_veh = energy_per_cycle_j_veh
        self.energy_per_cycle_j_rsu = energy_per_cycle_j_rsu
        self.p_tx_v2v_w = p_tx_v2v_w
        self.p_tx_rsu_w = p_tx_rsu_w
        self.offload_policy = offload_policy
        self.peer_selection = peer_selection
        self.v2v_ul_bps = _bytes_per_sec(v2v_rate_mbps_ul)
        self.v2v_dl_bps = _bytes_per_sec(v2v_rate_mbps_dl)
        self.rsu_default_uplink_bps   = _bytes_per_sec(rsu_default_uplink_mbps)
        self.rsu_default_downlink_bps = _bytes_per_sec(rsu_default_downlink_mbps)
        self.result_size_ratio = float(result_size_ratio)
        self.p_rx_veh_w = float(p_rx_veh_w)
        self.p_rx_rsu_w = float(p_rx_rsu_w)

        # Data
        self.df_rsu = load_rsus(self.rsu_csv)

        # Runtime state
        self.traci = None
        self.step_count = 0
        self.rng = np.random.default_rng(self.seed_val)

        # Per-node compute queues (available_at time in sim seconds)
        self.veh_spec = {}           # vid -> {"cpu_hz": float, "available_at": float}
        self.rsu_spec = {}           # rsu_id -> {"x","y","cpu_hz","uplink_bps","range_m","available_at": float}

        # Logs
        self._pos_rows = []
        self._task_rows = []         # generated tasks
        self._offload_rows = []      # offloading outcomes

        # Track last written step to avoid duplicates
        self.last_written_step = -1

        # Vehicle radio and params (V2V hops)
        self.rng_radio = np.random.default_rng(self.seed_val + 777)
        self.nr_radio_veh = NRRadio(
            carrier_hz=NR_CARRIER_HZ, p_tx_w=NR_P_TX_VEH_W, p_rx_w=NR_P_RX_VEH_W,
            noise_fig_db=NR_NOISE_FIG_VEH_DB, bw_hz=20e6, ant_gain_db=3.0
        )

        self.nr_params_veh = NRParams(
            scs_khz=NR_SCS_KHZ, slot_symbols=NR_SLOT_SYMBOLS, slot_duration_s=NR_SLOT_DURATION_S,
            prbs=NR_BANDWIDTH_PRB_VEH, overhead_data=NR_OVERHEAD_DATA, harq_reps=NR_HARQ_REPETITIONS,
            scenario=NR_SCENARIO, los_model=NR_LOS_MODEL,
            shadow_sigma_db_los=NR_SHADOW_SIGMA_DB_LOS, shadow_sigma_db_nlos=NR_SHADOW_SIGMA_DB_NLOS,
            cqi_thresh_db=NR_CQI_THRESH_DB
        )

        # RSU-side radio/params (veh<->RSU hops). You can share PRBs or set a different slice.
        self.nr_radio_rsu = NRRadio(
            carrier_hz=NR_CARRIER_HZ, p_tx_w=NR_P_TX_RSU_W, p_rx_w=NR_P_RX_RSU_W,
            noise_fig_db=NR_NOISE_FIG_RSU_DB, bw_hz=80e6, ant_gain_db=8.0
        )
        
        self.nr_params_rsu = NRParams(
            scs_khz=NR_SCS_KHZ, slot_symbols=NR_SLOT_SYMBOLS, slot_duration_s=NR_SLOT_DURATION_S,
            prbs=NR_BANDWIDTH_PRB_RSU, overhead_data=NR_OVERHEAD_DATA, harq_reps=NR_HARQ_REPETITIONS,
            scenario=NR_SCENARIO, los_model=NR_LOS_MODEL,
            shadow_sigma_db_los=NR_SHADOW_SIGMA_DB_LOS, shadow_sigma_db_nlos=NR_SHADOW_SIGMA_DB_NLOS,
            cqi_thresh_db=NR_CQI_THRESH_DB
        )

        # RNG already exists; reuse self.rng for cores too
        self.veh_spec = {}   # vid -> {"cores": int, "per_core_hz": float, "heap": [(available_at, core_id), ...]}
        self.rsu_spec = {}   # rid -> {"x","y",..., "cores","per_core_hz","heap":[(t,core)], ...}

        self.parallel = dict(PARALLEL)
        self.parallel_eta_veh = float(PARALLEL_ETA_VEH)
        self.parallel_eta_rsu = float(PARALLEL_ETA_RSU)
        self.thread_overhead_s = float(THREAD_OVERHEAD_S)

        self.reward_cfg = dict(REWARD_CFG)  # copy to avoid mutating global
        self.reward_cfg["weights"] = dict(self.reward_cfg["weights"])
        if reward_type == "deadline_only":
            self.reward_cfg["weights"] = {"sla": 1.0, "delay": 0.0, "energy_user": 0.0, "energy_system": 0.0}
            self.reward_cfg["miss_penalty"] = 1.0
            self.reward_cfg["ppo_mode"] = False  # pure binary SLA signal; no base_reward or efficiency bonus
        elif reward_type == "latency_only":
            self.reward_cfg["weights"] = {"sla": 0.0, "delay": 1.0, "energy_user": 0.0, "energy_system": 0.0}
            self.reward_cfg["miss_penalty"] = 0.0
            self.reward_cfg["ppo_mode"] = False  # delay-only signal; no base_reward or efficiency bonus

        self._step_rows = []   # one row per environment step

        # __init__
        self._task_uid = 0
        self._committed_task_ids = set()

        # in __init__
        self.pos_predict_mode = getattr(self, "pos_predict_mode", "const_vel")  # "const_vel" or "linear_csv"
        self._motion_cache = {}  # vid -> deque([(t,x,y), ...], maxlen=2)


    # -------- SUMO lifecycle --------
    def _start_sumo(self):
        self._close_sumo()
        self.traci = init_sumo(
            self.sumo_cfg,
            gui=self.gui,
            seed=self.seed_val,
            step_length=self.step_length_s,
            label=self.traci_label,
            add_remote_port=False,
            begin=self.begin_time,
            end=self.end_time,
            scale=self.demand_scale,
        )
        self.simulation = self.traci.simulation
        self.veh_api = self.traci.vehicle

        # Build RSU specs
        self.rsu_spec.clear()

        for _, row in self.df_rsu.iterrows():
            rid = str(row["rsu_id"])
            x, y = float(row["x"]), float(row["y"])

            # Link params you already compute here (uplink/downlink_bps, range_m, etc.)
            cpu_hz_total = None if row["cpu_hz"] is None or (isinstance(row["cpu_hz"], float) and not isfinite(row["cpu_hz"])) \
                        else float(row["cpu_hz"])

            # --- cores & per-core Hz ---
            if row["cores"] is not None and str(row["cores"]).strip() != "":
                cores = int(row["cores"])
            elif self._rsu_default_cores is not None:
                cores = int(self._rsu_default_cores)   # from config_loader profile
            else:
                cores = int(RSU_CORES_DEFAULT)

            if row["per_core_hz"] is not None and str(row["per_core_hz"]).strip() != "":
                per_core_hz = float(row["per_core_hz"])
            elif self._rsu_default_per_core_hz is not None:
                per_core_hz = float(self._rsu_default_per_core_hz)  # from config_loader profile
            else:
                # If a total cpu_hz was provided, derive per-core from it; else use default
                if cpu_hz_total is not None and cpu_hz_total > 0 and cores > 0:
                    per_core_hz = cpu_hz_total / cores
                else:
                    per_core_hz = float(RSU_PER_CORE_HZ_DEFAULT)

            # Build core heap: (available_at_time, core_id)
            heap = [(0.0, i) for i in range(cores)]
            heapq.heapify(heap)

            # Keep your existing RSU link/spec fields; add cores/per_core_hz/heap
            self.rsu_spec[rid] = {
                "x": x, "y": y,
                # keep your link bps/range/cpu_hz *if you still use them elsewhere*
                "cpu_hz": cores * per_core_hz,
                "uplink_bps": self.rsu_default_uplink_bps,
                "downlink_bps": self.rsu_default_downlink_bps,
                "range_m": self.rsu_default_range_m,
                # multi-core:
                "cores": cores,
                "per_core_hz": per_core_hz,
                "heap": heap
            }

    def reset(self, *, seed=None, options=None):
        set_global_seeds(self.seed_val if seed is None else seed)
        if self.traci is not None:
            try: self.traci.close()
            except Exception: pass
            self.traci = None

        self._close_sumo()

        # Sample random episode window from timeframe if applicable
        if self.timeframe_begin is not None and self.timeframe_end is not None:
            episode_duration_s = int(self.max_steps * self.step_length_s)
            timeframe_duration_s = self.timeframe_end - self.timeframe_begin

            if episode_duration_s < timeframe_duration_s:
                # Sample random start within timeframe using numpy random (not affected by global seed)
                max_start_offset = timeframe_duration_s - episode_duration_s
                # Use instance RNG with seed derived from base seed for reproducible but varied episodes
                if not hasattr(self, '_episode_rng'):
                    self._episode_rng = np.random.RandomState(self.seed_val + 999)
                random_offset = self._episode_rng.randint(0, max_start_offset + 1)
                self.begin_time = self.timeframe_begin + random_offset
                self.end_time = self.begin_time + episode_duration_s
            else:
                # Episode covers full timeframe or longer
                self.begin_time = self.timeframe_begin
                self.end_time = self.timeframe_begin + episode_duration_s

        self._start_sumo()
        self._reset_compute_heaps(now=0.0)
        self.step_count = 0  # Episode step count
        
        # RESET on new training: Clear log files and reset counters
        if not hasattr(self, 'global_step_count'):
            self.global_step_count = 0
            self.episode_number = 1  # Track episode numbers
            # Clear log files on fresh start
            self._clear_log_files()
            # Reset counters
            self._vehicle_task_counts = {}
            self._episode_stats = {
                'total_tasks': 0,
                'max_tasks_per_step': 0,
                'step_count': 0
            }
        else:
            # New episode - increment episode number and reset episode stats
            self.episode_number += 1
            self._episode_stats = {
                'total_tasks': 0,
                'max_tasks_per_step': 0,
                'step_count': 0
            }
            # Advance salt so _rng_for produces fresh task draws each episode,
            # preventing repeated (sim_step, vid) pairs from yielding identical tasks
            self._episode_salt = getattr(self, "_episode_salt", 0) + 1
        
        self.veh_spec.clear()
        self._pos_rows.clear()
        # DON'T clear task and offload rows - keep accumulating across episodes
        # self._task_rows.clear()
        # self._offload_rows.clear()
        # self._step_rows.clear()

        # Reset duplicate tracking
        self.last_written_step = -1

        self._task_uid = 0
        self._committed_task_ids.clear()

        obs = self._make_observation([])
        return obs, {"tasks_generated": [], "offloads": []}
        # return obs, info
    
    # -------- Step --------
    def step(self, action):
        self.traci.simulationStep()
        self.step_count += 1
        self.global_step_count += 1  # Track global steps across episodes

        # Update vehicle positions & ensure specs exist
        live_ids = list(self.traci.vehicle.getIDList())
        self._touch_motion_cache(live_ids)
        self._log_positions()
        
        for v in live_ids:
            self._ensure_vehicle_spec(v)

        # for vid in live_ids:
        #     if vid not in self.veh_spec:
        #         cpu = max(self.veh_cpu_hz_min, _trunc_pos(self.rng, self.veh_cpu_hz_mean, self.veh_cpu_hz_std, self.veh_cpu_hz_min))
        #         self.veh_spec[vid] = {"cpu_hz": float(cpu), "available_at": 0.0}

        # Generate tasks
        # generated = self._generate_tasks(live_ids)

        # # Offload each generated task
        # decisions = []
        # for rec in generated:
        #     drec = self._decide_and_schedule(rec, live_ids)
        #     decisions.append(drec)

        # Generate tasks only once
        # generated = self._generate_tasks(live_ids) # not seeded
        generated = self._generate_tasks_seeded(live_ids) # seeded

        decisions = []
        if getattr(self, "offload_policy", "greedy") != "agent":
            # baseline/greedy path (keeps old behavior)
            for rec in generated:
                drec = self._decide_and_schedule(rec, live_ids)   # this logs once
                decisions.append(drec)
        # else: when "agent", DO NOT offload here. The wrapper will call _commit_candidate()


        # reward = 0.0  # logging-first baseline

        task_rewards = [self._compute_task_reward(dec) for dec in decisions]
        if not task_rewards:
            step_reward = 0.0
        else:
            agg = self.reward_cfg["aggregate"]
            if agg == "sum":
                step_reward = float(sum(task_rewards))
            elif agg == "median":
                step_reward = float(np.median(task_rewards))
            else:  # "mean"
                step_reward = float(np.mean(task_rewards))

        terminated = False
        truncated = self.step_count >= self.max_steps
        obs = self._make_observation(generated)
        info = {"tasks_generated": generated, "offloads": decisions}

        # decisions: the list returned from self._decide_and_schedule(...) for all tasks this step
        n = len(decisions)

        if n:
            delays  = np.array([float(d.get("total_delay_s", 0.0)) for d in decisions], dtype=float)
            e_user  = np.array([float(d.get("energy_user_j", 0.0)) for d in decisions], dtype=float)
            e_sys   = np.array([float(d.get("energy_system_j", 0.0)) for d in decisions], dtype=float)
            sla_hit = np.array([bool(d.get("deadline_met", False)) for d in decisions], dtype=bool)
            decs    = [str(d.get("decision", "")) for d in decisions]
        else:
            delays = e_user = e_sys = sla_hit = np.array([])
            decs   = []

        row = {
            "step": int(self.global_step_count),  # Use global step count for continuous numbering
            "step_reward": float(step_reward),
            "tasks": int(n),
            "deadline_met": int(sla_hit.sum()) if n else 0,
            "deadline_miss": int(n - sla_hit.sum()) if n else 0,
            "deadline_met_rate": float(sla_hit.mean()) if n else 0.0,
            "mean_delay_s": float(delays.mean()) if n else 0.0,
            "p95_delay_s": float(np.quantile(delays, 0.95)) if n else 0.0,
            "mean_user_energy_j": float(e_user.mean()) if n else 0.0,
            "mean_system_energy_j": float(e_sys.mean()) if n else 0.0,
            "dec_local": int(sum(1 for x in decs if x == "local")),
            "dec_v2v":   int(sum(1 for x in decs if x == "v2v")),
            "dec_rsu":   int(sum(1 for x in decs if x == "rsu")),
        }

        self._step_rows.append(row)
        
        # ONLY write to CSV in non-agent mode (agent mode writes in agent_finalize_tick)
        if getattr(self, "offload_policy", "greedy") != "agent":
            # CONTINUOUS LOGGING: Write step data immediately to CSV
            self._write_step_to_csv(row)
            
            # MEMORY OPTIMIZATION: Clear old data periodically
            self._clear_memory_periodically()

        # (optional) also expose in info for live debugging
        info["step_reward"] = row["step_reward"]
        info["step_summary"] = {k: row[k] for k in [
            "tasks","deadline_met_rate","mean_delay_s","mean_user_energy_j"
        ]}

        if terminated or truncated:
            self._flush_logs()
            self._close_sumo()

        return obs, step_reward, terminated, truncated, info

    # -------- Helpers --------
    def _make_observation(self, current_tasks):
        feats = []
        for vid in self.traci.vehicle.getIDList()[: self.obs_max_veh]:
            x, y = self.traci.vehicle.getPosition(vid)
            speed = self.traci.vehicle.getSpeed(vid)
            feats.extend([x, y, speed])
        
        if current_tasks and len(current_tasks) > 0:
            task = current_tasks[0]
            # TODO
            # category, cpu cycles, data bytes
            # I'd also like to add the time to deadline (current time - arrival + deadline)
            task_features = [
                task['cpu_cycles'] / 5e9,           # Normalize to [0, 1]
                task['data_bytes'] / 5e6,           # Normalize to [0, 1]
                1.0 if task['category'] == "autonomous" else 0.0,
                1.0 if task['category'] == "safety" else 0.0,
                1.0 if task['category'] == "infotainment" else 0.0,
            ]

            feats.extend(task_features)

        need = self.obs_max_veh * self.obs_per_veh - len(feats)
        if need > 0:
            feats.extend([0.0] * need)
        return np.array(feats, dtype=np.float32)

    def _log_positions(self):
        sim_time = int(self.step_count)
        for vid in self.traci.vehicle.getIDList():
            x, y = self.traci.vehicle.getPosition(vid)
            speed = self.traci.vehicle.getSpeed(vid)
            self._pos_rows.append({"step": sim_time, "vehicle_id": vid, "x": x, "y": y, "speed": speed})

    def _generate_tasks(self, live_ids):
        
        lam = self.lambda_per_veh_per_sec * self.step_length_s
        out = []
        for vid in live_ids:
            k = self.rng.poisson(lam)
            if k <= 0: continue
            for _ in range(int(k)):
                pos_step = int(getattr(self, "_pos_step_current", self.step_count))
                self._task_uid += 1
                tid = f"{pos_step}:{vid}:{self._task_uid}"  # stable within episode
                cat = self._cat_sample()
                prof = self.category_profiles[cat]
                cpu = _trunc_pos(self.rng, *prof["cpu_cycles"])
                dat = _trunc_pos(self.rng, *prof["data_bytes"])
                ddl = _trunc_pos(self.rng, *prof["deadline_s"])
                rec = {"step": int(self.global_step_count), "tid": tid, "vehicle_id": vid, "category": cat,
                       "cpu_cycles": float(cpu), "data_bytes": float(dat), "deadline_s": float(ddl), "arrival_s": float(pos_step)}
                self._task_rows.append(rec)
                self._write_task_to_csv(rec)  # Write immediately to CSV
                out.append(rec)
        return out
    
    
    # TODO impement dynamic rates per vehicle dependent on nearby density
    def _generate_tasks_seeded(self, live_ids):
        lam = float(self.lambda_per_veh_per_sec) * float(self.step_length_s)
        sim_step = int(getattr(self, "_pos_step_current", self.step_count))
        out = []

        # fixed order so the sequence of draws is identical
        for vid in sorted(map(str, live_ids)):
            g = _rng_for("tasks", sim_step, vid)

            rate = lam
            # TODO
            # get vehicle nearby density
            # change rate accordingly
            # use config to inform how rate is

            k = int(g.poisson(rate))
            if k <= 0: 
                continue

            for i in range(k):
                # category using the same RNG
                
                cats = list(self.category_profiles.keys())
                p = np.array([self.category_mix[c] for c in cats], float)
                p = p / p.sum()
                cat = str(g.choice(cats, p=p))

                prof = self.category_profiles[cat]

                # choose the correct draw per field (normal vs lognormal) — keep SAME in both envs
                cpu = _trunc_pos(g,*prof["cpu_cycles"])        # or _trunc_lognorm(...)
                dat = _trunc_pos(g,*prof["data_bytes"])        # or _trunc_lognorm(...)
                ddl = _trunc_pos(g,*prof["deadline_s"])        # usually normal

                # clamps (prevent NaN/inf drift)
                cpu = float(min(max(cpu, 1.0), 1e13))
                dat = float(min(max(dat, 1.0), int(getattr(self, "max_ul_bytes", 32*1024*1024))))
                ddl = float(max(ddl, 1e-6))

                tid = f"{sim_step}:{vid}:{i+1}"
                rec = {"step": int(self.global_step_count), "tid": tid, "vehicle_id": vid, "category": cat,
                    "cpu_cycles": cpu, "data_bytes": dat, "deadline_s": ddl, "arrival_s": float(sim_step)}
                self._task_rows.append(rec)
                self._write_task_to_csv(rec)  # Write immediately to CSV
                out.append(rec)
        return out

    def _cat_sample(self):
        cats = list(self.category_mix.keys())
        p = np.array([self.category_mix[c] for c in cats], float)
        p = p / p.sum()
        return cats[self.rng.choice(len(cats), p=p)]

    def _get_pos(self, vid):
        try:
            x, y = self.traci.vehicle.getPosition(vid)
        except Exception:
            x, y = self._get_xy(vid)
        return float(x), float(y)

    def _nearest_peer(self, src_vid, live_ids, max_range_m) -> Optional[str]:
        if len(live_ids) <= 1: return None
        sx, sy = self._get_pos(src_vid)
        best, bestd = None, float("inf")
        for vid in live_ids:
            if vid == src_vid: continue
            x, y = self.traci.vehicle.getPosition(vid)
            d = math.hypot(x - sx, y - sy)
            if d <= max_range_m and d < bestd:
                bestd, best = d, vid
        return best

    def _best_peer(self, src_vid, live_ids, max_range_m,
                   now: float, cpu_cycles: float, data_bytes: int,
                   category: str) -> Optional[str]:
        """
        Return the peer within V2V range that minimises estimated task finish time.

        Uses simplified UL estimate (data_bytes * 8 / v2v_rate_bps) instead of the
        full stochastic NR model so we avoid N noisy radio calls just for ranking.
        The actual UL/DL NR computation still happens after the peer is selected.

        All heap probes use commit=False — no side effects during the search.
        Callers must have already called _ensure_vehicle_spec for all live_ids.
        """
        if len(live_ids) <= 1:
            return None

        sx, sy = self._get_pos(src_vid)
        P, kmax, eta = self._parallel_params(category, executor="veh")
        ul_s = data_bytes * 8.0 / max(1.0, self.v2v_rate_bps)  # simplified UL

        best_peer, best_finish = None, float("inf")

        for vid in live_ids:
            if vid == src_vid:
                continue
            x, y = self.traci.vehicle.getPosition(vid)
            d = math.hypot(x - sx, y - sy)
            if d > max_range_m:
                continue

            arrival = now + ul_s
            _, _, finish_s, _, _, _ = self._schedule_multi_on_vehicle(
                vid, arrival, cpu_cycles, P, kmax, eta, self.thread_overhead_s, commit=False
            )
            if finish_s < best_finish:
                best_finish = finish_s
                best_peer = vid

        return best_peer

    def _select_peer(self, src_vid, live_ids, max_range_m,
                     now: float, cpu_cycles: float, data_bytes: int,
                     category: str) -> Optional[str]:
        """Dispatch to _best_peer or _nearest_peer based on self.peer_selection."""
        if self.peer_selection == "best":
            return self._best_peer(src_vid, live_ids, max_range_m,
                                   now, cpu_cycles, data_bytes, category)
        return self._nearest_peer(src_vid, live_ids, max_range_m)

    def _nearest_rsu(self, src_vid) -> Optional[Tuple[str, float]]:
        sx, sy = self._get_pos(src_vid)
        best_id, best_d, best_rng = None, float("inf"), 0.0
        for rid, spec in self.rsu_spec.items():
            d = math.hypot(spec["x"] - sx, spec["y"] - sy)
            if d <= spec["range_m"] and d < best_d:
                best_id, best_d, best_rng = rid, d, spec["range_m"]

        if best_id is None:
            return None
        self._a+=1

        if best_d > 500:
            self._c+=1
            print(self._c/self._a)
        return best_id, best_d

    def _decide_and_schedule(self, task, live_ids):
        """
        Greedy by total delay with two-way NR-V2X links and multi-core compute.
        LOCAL: requester cores; V2V: peer cores; RSU: RSU cores.
        """

        # now = float(self.step_count)
        now = self._get_sim_step()
        vid = task["vehicle_id"]
        cpu_cycles  = float(task["cpu_cycles"])
        data_bytes  = int(max(1.0, float(task["data_bytes"])))
        result_bytes = int(max(1.0, self.result_size_ratio * data_bytes))
        deadline_s  = float(task["deadline_s"])

        # Ensure specs exist for requester + all live vehicles
        for v in live_ids:
            self._ensure_vehicle_spec(v)

        # -------- LOCAL (on requester, multi-threaded) --------
        self._ensure_vehicle_spec(vid)  # from your previous patch
        P_loc, kmax_loc, eta_loc = self._parallel_params(task["category"], executor="veh")
        l_start, l_proc, l_finish, l_queue, l_core_ids, l_k = self._schedule_multi_on_vehicle(
            vid, now, cpu_cycles, P_loc, kmax_loc, eta_loc, self.thread_overhead_s, commit=False
        )
        cand_local = {
            "mode": "local", "node_id": vid,
            "tx_up_s": 0.0, "rx_up_s_exec": 0.0,
            "queue_s": l_queue, "proc_s": l_proc,
            "tx_down_s": 0.0, "rx_down_s_user": 0.0,
            "finish_s": l_finish, "delay_s": l_finish - now,
            "tx_up_rate_mbps": 0.0, "tx_down_rate_mbps": 0.0,
            "distance_m": 0.0,"ul_distance_m": 0.0,"dl_distance_m": 0.0,
            "ul_snr_db": 0.0, "ul_cqi": 0, "dl_snr_db": 0.0, "dl_cqi": 0,
            "e_user_comp_j": cpu_cycles * self.energy_per_cycle_j_veh,
            "e_user_tx_up_j": 0.0, "e_user_rx_down_j": 0.0,
            "e_exec_rx_up_j": 0.0, "e_exec_comp_j": 0.0, "e_exec_tx_down_j": 0.0,
            "executor_core_id": l_core_ids[0] if l_k == 1 else -1,
            "executor_core_ids": l_core_ids,
            "executor_k_used": l_k,
        }
        decisions = [cand_local]

        # -------- V2V (peer vehicle compute, multi-threaded) --------
        peer_id = self._select_peer(vid, live_ids, self.v2v_range_m,
                                    now, cpu_cycles, data_bytes, task["category"])
        if peer_id is not None:
            self._ensure_vehicle_spec(peer_id)
            d = self._dist_veh(vid, peer_id)

            # uplink (keep your existing NR hop)
            ul = one_nr_sidelink_hop(d_m=d, payload_bytes=data_bytes,
                                    tx=self.nr_radio_veh, rx=self.nr_radio_veh,
                                    nr=self.nr_params_veh, rng=self.rng_radio)

            # compute on peer (simulate)
            P_v2v, kmax_v2v, eta_v2v = self._parallel_params(task["category"], executor="veh")
            arrival_peer = now + ul["tx_time_s"]
            p_start, p_proc, p_finish, p_queue, p_core_ids, p_k = self._schedule_multi_on_vehicle(
                peer_id, arrival_peer, cpu_cycles, P_v2v, kmax_v2v, eta_v2v, self.thread_overhead_s, commit=False
            )

            t_dl_start = float(p_finish)

            xy_user = self._predict_xy_new(vid, t_dl_start)
            xy_peer = self._predict_xy_new(peer_id, t_dl_start)

            if xy_user is not None and xy_peer is not None:
                dx = xy_user[0] - xy_peer[0]
                dy = xy_user[1] - xy_peer[1]
                d_dl = math.hypot(dx, dy)
            else:
                d_dl = d  # fallback

            # downlink (keep your existing NR hop)
            dl = one_nr_sidelink_hop(d_m=d_dl, payload_bytes=int(max(1, self.result_size_ratio * data_bytes)),
                                    tx=self.nr_radio_veh, rx=self.nr_radio_veh,
                                    nr=self.nr_params_veh, rng=self.rng_radio)
            finish_total = p_finish + dl["tx_time_s"]

            decisions.append({
                "mode": "v2v", "node_id": peer_id,
                "tx_up_s": ul["tx_time_s"], "rx_up_s_exec": ul["tx_time_s"],
                "queue_s": p_queue, "proc_s": p_proc,
                "tx_down_s": dl["tx_time_s"], "rx_down_s_user": dl["tx_time_s"],
                "finish_s": finish_total, "delay_s": finish_total - now,
                "tx_up_rate_mbps": ul["rate_mbps"], "tx_down_rate_mbps": dl["rate_mbps"],
                "distance_m": d,
                "ul_snr_db": ul["snr_db"], "ul_cqi": ul["cqi"],
                "dl_snr_db": dl["snr_db"], "dl_cqi": dl["cqi"],
                "e_user_comp_j": 0.0,
                "e_user_tx_up_j": ul["e_tx_j"], "e_user_rx_down_j": dl["e_rx_j"],
                "e_exec_rx_up_j": ul["e_rx_j"],
                "e_exec_comp_j": cpu_cycles * self.energy_per_cycle_j_veh,
                "e_exec_tx_down_j": dl["e_tx_j"],
                "executor_core_id": p_core_ids[0] if p_k == 1 else -1,
                "executor_core_ids": p_core_ids,
                "executor_k_used": p_k,
                "_sched_sim": ("veh", peer_id, arrival_peer, cpu_cycles, P_v2v, kmax_v2v, eta_v2v),
                "ul_distance_m": float(d),       # distance at UL time
                "dl_distance_m": float(d_dl)    # distance at DL start time
            })


        # -------- RSU (multi-threaded) --------
        rsu_pick = self._nearest_rsu(vid)
        if rsu_pick is not None:
            rid, d = rsu_pick

            # uplink
            ul = one_nr_sidelink_hop(d_m=d, payload_bytes=data_bytes,
                                    tx=self.nr_radio_veh, rx=self.nr_radio_rsu,
                                    nr=self.nr_params_rsu, rng=self.rng_radio)

            # compute on RSU (simulate)
            P_rsu, kmax_rsu, eta_rsu = self._parallel_params(task["category"], executor="rsu")
            arrival_rsu = now + ul["tx_time_s"]
            r_start, r_proc, r_finish, r_queue, r_core_ids, r_k = self._schedule_multi_on_rsu(
                rid, arrival_rsu, cpu_cycles, P_rsu, kmax_rsu, eta_rsu, self.thread_overhead_s, commit=False
            )

            t_dl_start = float(r_finish)

            # distance at download start: vehicle moved; RSU is fixed
            veh_xy_at_dl = self._predict_xy(vid, t_dl_start)
            if veh_xy_at_dl is not None:
                vx, vy = veh_xy_at_dl
                rsu_spec = self.rsu_spec[rid]
                dx = vx - float(rsu_spec["x"])
                dy = vy - float(rsu_spec["y"])
                d_dl = math.hypot(dx, dy)
            else:
                d_dl = d  # fallback to distance measured at decision time
            
            # Vehicle moved; RSU fixed → recalc DL distance at t_dl_start
            # vx, vy = self._predict_xy(vid, t_dl_start)  # CSV: interpolates (x,y)
            # rsu_x, rsu_y = float(self.rsu_spec[rid]["x"]), float(self.rsu_spec[rid]["y"])
            # d_dl = math.hypot(vx - rsu_x, vy - rsu_y)

            d_dl = self._dl_distance_rsu(vid, rid, t_dl_start)

            # downlink
            dl = one_nr_sidelink_hop(d_m=d_dl, payload_bytes=int(max(1, self.result_size_ratio * data_bytes)),
                                    tx=self.nr_radio_rsu, rx=self.nr_radio_veh,
                                    nr=self.nr_params_rsu, rng=self.rng_radio)
            finish_total = r_finish + dl["tx_time_s"]

            decisions.append({
                "mode": "rsu", "node_id": rid,
                "tx_up_s": ul["tx_time_s"], "rx_up_s_exec": ul["tx_time_s"],
                "queue_s": r_queue, "proc_s": r_proc,
                "tx_down_s": dl["tx_time_s"], "rx_down_s_user": dl["tx_time_s"],
                "finish_s": finish_total, "delay_s": finish_total - now,
                "tx_up_rate_mbps": ul["rate_mbps"], "tx_down_rate_mbps": dl["rate_mbps"],
                "distance_m": d,
                "ul_snr_db": ul["snr_db"], "ul_cqi": ul["cqi"],
                "dl_snr_db": dl["snr_db"], "dl_cqi": dl["cqi"],
                "e_user_comp_j": 0.0,
                "e_user_tx_up_j": ul["e_tx_j"], "e_user_rx_down_j": dl["e_rx_j"],
                "e_exec_rx_up_j": ul["e_rx_j"],
                "e_exec_comp_j": cpu_cycles * self.energy_per_cycle_j_rsu,
                "e_exec_tx_down_j": dl["e_tx_j"],
                "executor_core_id": r_core_ids[0] if r_k == 1 else -1,
                "executor_core_ids": r_core_ids,
                "executor_k_used": r_k,
                "_sched_sim": ("rsu", rid, arrival_rsu, cpu_cycles, P_rsu, kmax_rsu, eta_rsu),
                "ul_distance_m": float(d),       # distance at UL time
                "dl_distance_m": float(d_dl)    # distance at DL start time
            })

        # -------- choose best (delay) --------
        chosen = min(decisions, key=lambda c: c["delay_s"]) if self.offload_policy == "greedy" \
                else decisions[self.rng.integers(0, len(decisions))]

        # ---- commit compute reservation on chosen executor ----
        if chosen["mode"] == "local":
            self._schedule_multi_on_vehicle(
                vid, now, cpu_cycles,
                *self._parallel_params(task["category"], "veh"),
                self.thread_overhead_s, commit=True
            )
        else:
            kind, node_id, arr_t, cyc, P, kmax, eta = chosen.get("_sched_sim", (None, None, None, None, 0.0, 1, self.parallel_eta_rsu))
            if kind == "veh":
                self._schedule_multi_on_vehicle(node_id, arr_t, cyc, P, kmax, eta, self.thread_overhead_s, commit=True)
            elif kind == "rsu":
                self._schedule_multi_on_rsu(node_id, arr_t, cyc, P, kmax, eta, self.thread_overhead_s, commit=True)


        # -------- totals & log row --------
        meets_deadline = chosen["delay_s"] <= deadline_s
        e_user = chosen["e_user_comp_j"] + chosen["e_user_tx_up_j"] + chosen["e_user_rx_down_j"]
        e_exec = chosen["e_exec_comp_j"] + chosen["e_exec_rx_up_j"] + chosen["e_exec_tx_down_j"]

        out = {
            "step": int(self.global_step_count), "vehicle_id": vid, "category": task["category"],
            "cpu_cycles": cpu_cycles, "data_bytes": data_bytes, "result_bytes": result_bytes,
            "deadline_s": deadline_s, "decision": chosen["mode"], "node_id": chosen["node_id"],
            "executor_core_id": chosen.get("executor_core_id", -1),
            "executor_core_ids": chosen.get("executor_core_ids", -1),
            "executor_k_used": chosen.get("executor_k_used", -1),

            "tx_up_time_s": chosen["tx_up_s"], "rx_up_time_s_executor": chosen["rx_up_s_exec"],
            "queue_delay_s": chosen["queue_s"], "proc_time_s": chosen["proc_s"],
            "tx_down_time_s": chosen["tx_down_s"], "rx_down_time_s_user": chosen["rx_down_s_user"],
            "finish_time_s": chosen["finish_s"], "total_delay_s": chosen["delay_s"],
            "deadline_met": bool(meets_deadline),

            "tx_up_rate_mbps": chosen["tx_up_rate_mbps"], "tx_down_rate_mbps": chosen["tx_down_rate_mbps"],
            "distance_m": chosen["distance_m"], "ul_distance_m": chosen["ul_distance_m"], "dl_distance_m": chosen["dl_distance_m"],
            "ul_snr_db": chosen["ul_snr_db"], "ul_cqi": chosen.get("ul_cqi", 0),
            "dl_snr_db": chosen["dl_snr_db"], "dl_cqi": chosen.get("dl_cqi", 0),

            "e_user_comp_j": chosen["e_user_comp_j"], "e_user_tx_up_j": chosen["e_user_tx_up_j"],
            "e_user_rx_down_j": chosen["e_user_rx_down_j"], "e_exec_rx_up_j": chosen["e_exec_rx_up_j"],
            "e_exec_comp_j": chosen["e_exec_comp_j"], "e_exec_tx_down_j": chosen["e_exec_tx_down_j"],
            "energy_user_j": e_user, "energy_executor_j": e_exec, "energy_system_j": e_user + e_exec,
            "env_step": int(self.step_count),
            "sim_step": int(self._get_sim_step())
        }
        self._offload_rows.append(out)
        # self._write_offload_to_csv(out)  # Disabled to save disk space during training

        return out

    
    def _ensure_vehicle_spec(self, vid: str):
        if vid in self.veh_spec:
            return
        # Sample cores and per-core Hz from the vehicle class distribution
        cores, per_core_hz = _sample_veh_cpu(self.rng)
        heap = [(0.0, i) for i in range(cores)]
        heapq.heapify(heap)
        self.veh_spec[vid] = {"cores": cores, "per_core_hz": per_core_hz, "heap": heap}

    def _schedule_on_heap(self, heap, per_core_hz, arrival_time, cpu_cycles, commit=False):
        """
        Multi-core queue: take earliest-free core from heap, start=max(arrival, available_at),
        proc = cycles / per_core_hz, finish = start + proc.
        Returns (start, proc, finish, queue_delay, core_id). If commit=True, mutates heap.
        """
        # Copy for simulation
        if commit:
            h = heap
        else:
            h = list(heap)
            heapq.heapify(h)

        available_at, core_id = heapq.heappop(h)
        start = max(arrival_time, available_at)
        proc  = cpu_cycles / max(1.0, per_core_hz)
        # print(f"Proc_s:{cpu_cycles} + {max(1.0, per_core_hz)} = {proc}")
        finish = start + proc
        heapq.heappush(h, (finish, core_id))

        if commit:
            # already mutated the real heap
            pass
        return start, proc, finish, (start - arrival_time), core_id

    def _schedule_on_vehicle(self, vid, arrival_time, cpu_cycles, commit=False):
        spec = self.veh_spec[vid]
        return self._schedule_on_heap(spec["heap"], spec["per_core_hz"], arrival_time, cpu_cycles, commit=commit)

    def _schedule_on_rsu(self, rid, arrival_time, cpu_cycles, commit=False):
        spec = self.rsu_spec[rid]
        return self._schedule_on_heap(spec["heap"], spec["per_core_hz"], arrival_time, cpu_cycles, commit=commit)
    
    def _parallel_params(self, category: str, executor: str):
        """
        executor: 'veh' or 'rsu'
        Returns (P, kmax, eta). Falls back to single-thread if category/executor missing.
        """
        cfg = self.parallel.get(category, None)
        if not cfg:
            return 0.0, 1, (self.parallel_eta_veh if executor == "veh" else self.parallel_eta_rsu)
        kmax = int(cfg.get("kmax_veh" if executor == "veh" else "kmax_rsu", 1))
        kmax = max(1, kmax)
        P = float(cfg.get("P", 0.0))
        eta = self.parallel_eta_veh if executor == "veh" else self.parallel_eta_rsu
        return P, kmax, eta

    def _schedule_multi_on_heap(self, heap, per_core_hz, arrival_time, cpu_cycles,
                                P, kmax, eta, delta_s, commit=False):
        """
        Reserve k earliest-free cores (k<=kmax) from heap. Compute time:
        t1 = cpu_cycles / per_core_hz
        t_k = t1 * ((1 - P) + P / (eta * k)) + (k - 1) * delta_s
        Returns: (start, proc, finish, queue_delay, used_core_ids(list), k_used)
        If commit=True, mutates the real heap; otherwise simulates on a copy.
        """
        # Work on real heap (commit) or a copy (simulate)
        h = heap if commit else list(heap)
        if not commit:
            heapq.heapify(h)

        # Number of cores we can take now
        k = max(1, min(kmax, len(h)))

        # Pop k earliest-free cores
        popped = [heapq.heappop(h) for _ in range(k)]
        avail_times = [t for (t, _) in popped]
        used_core_ids = [c for (_, c) in popped]

        start = max(arrival_time, max(avail_times) if popped else arrival_time)
        t1 = cpu_cycles / max(1.0, per_core_hz)
        proc = t1 * ((1.0 - P) + (P / max(1e-6, eta * k))) + (k - 1) * max(0.0, delta_s)
        finish = start + proc
        queue_delay = start - arrival_time

        # Push the k cores back with the new finish time
        for _, core_id in popped:
            heapq.heappush(h, (finish, core_id))

        # If we simulated on a copy, do nothing else; if commit=True we already mutated
        return start, proc, finish, queue_delay, used_core_ids, k

    def _schedule_multi_on_vehicle(self, vid, arrival_time, cpu_cycles, P, kmax, eta, delta_s, commit=False):
        spec = self.veh_spec[vid]
        if "heap" not in spec:
            # migrate legacy single-queue spec if needed (from previous patch)
            self._migrate_spec_to_heap(spec, spec.get("per_core_hz", spec.get("cpu_hz", self.veh_cpu_hz_mean)))
        return self._schedule_multi_on_heap(spec["heap"], spec["per_core_hz"], arrival_time, cpu_cycles,
                                            P, kmax, eta, delta_s, commit=commit)

    def _schedule_multi_on_rsu(self, rid, arrival_time, cpu_cycles, P, kmax, eta, delta_s, commit=False):
        spec = self.rsu_spec[rid]
        if "heap" not in spec:
            # minimal safety (should already be built with heap)
            cores = int(spec.get("cores", 1)); per_core_hz = float(spec.get("per_core_hz", 3.0e9))
            spec["heap"] = [(0.0, i) for i in range(cores)]; heapq.heapify(spec["heap"])
        return self._schedule_multi_on_heap(spec["heap"], spec["per_core_hz"], arrival_time, cpu_cycles,
                                            P, kmax, eta, delta_s, commit=commit)

    def _dist_veh(self, a, b):
        ax, ay = self.traci.vehicle.getPosition(a)
        bx, by = self.traci.vehicle.getPosition(b)
        return float(math.hypot(ax - bx, ay - by))

    # -------- I/O --------
    def _append_step_log(self):
        """Fast continuous logging - append only the latest step"""
        if self._step_rows:
            latest_row = self._step_rows[-1]  # Get the most recent step
            # Keep only first 3 values: step, step_reward, tasks
            row_data = [latest_row.get('step', ''), latest_row.get('step_reward', ''), latest_row.get('tasks', '')]
            
            # Fast append to file
            step_rewards_path = self.log_dir / "step_rewards.csv"
            with open(step_rewards_path, 'a') as f:
                f.write(f"{row_data[0]},{row_data[1]},{row_data[2]}\n")

    def _write_step_to_csv(self, row):
        """Write step data immediately to CSV for continuous logging"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Write step data immediately 
        step_file = self.log_dir / "step_rewards.csv"
        if not step_file.exists():
            with open(step_file, 'w') as f:
                f.write("step,step_reward,tasks\n")
        
        with open(step_file, 'a') as f:
            f.write(f"{row['step']},{row['step_reward']},{row['tasks']}\n")

        # Full per-tick summary (incl. deadline_met/miss + action counts) for eval.
        self._write_step_summary_full(row)

        # Update episode task statistics
        self._update_episode_task_count(row['tasks'])

    # Columns for the full per-tick evaluation summary.
    _STEP_SUMMARY_COLS = [
        "step", "tasks", "deadline_met", "deadline_miss", "deadline_met_rate",
        "mean_delay_s", "p95_delay_s", "mean_user_energy_j", "mean_system_energy_j",
        "dec_local", "dec_v2v", "dec_rsu", "step_reward",
    ]

    def _write_step_summary_full(self, row):
        """Append the FULL per-tick summary to step_summary.csv.

        step_rewards.csv keeps only 3 columns and self._step_rows is trimmed
        every 1000 steps for memory, so neither is a reliable source for the
        deadline-satisfaction metric. This file (one row per sim-tick, ~100
        rows/episode) is what the evaluation harness reads. Cheap on disk.
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)
        f_path = self.log_dir / "step_summary.csv"
        if not f_path.exists():
            with open(f_path, "w") as f:
                f.write(",".join(self._STEP_SUMMARY_COLS) + "\n")
        with open(f_path, "a") as f:
            f.write(",".join(str(row.get(c, "")) for c in self._STEP_SUMMARY_COLS) + "\n")

    def _write_task_to_csv(self, task_data):
        """Write task data immediately to CSV"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        task_file = self.log_dir / "task_generation_log.csv"
        if not task_file.exists():
            # Write header
            with open(task_file, 'w') as f:
                keys = list(task_data.keys())
                f.write(",".join(keys) + "\n")
        
        with open(task_file, 'a') as f:
            values = [str(task_data.get(k, '')) for k in task_data.keys()]
            f.write(",".join(values) + "\n")
        
        # Update tasks_per_vehicle.csv incrementally
        self._update_vehicle_task_count(task_data['vehicle_id'])

    def _update_vehicle_task_count(self, vehicle_id):
        """Update the task count for a specific vehicle in tasks_per_vehicle.csv"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        vehicle_file = self.log_dir / "tasks_per_vehicle.csv"
        
        # Initialize vehicle task counter if not exists
        if not hasattr(self, '_vehicle_task_counts'):
            self._vehicle_task_counts = {}
        
        # Increment count for this vehicle
        self._vehicle_task_counts[vehicle_id] = self._vehicle_task_counts.get(vehicle_id, 0) + 1
        
        # Only update CSV file periodically to avoid fluctuation (every 100 tasks)
        if not hasattr(self, '_vehicle_update_counter'):
            self._vehicle_update_counter = 0
        
        self._vehicle_update_counter += 1
        
        # Write/update the CSV file only periodically or at episode end
        if self._vehicle_update_counter % 100 == 0 or getattr(self, '_force_vehicle_update', False):
            try:
                with open(vehicle_file, 'w') as f:
                    f.write("vehicle_id,task_count\n")
                    for vid, count in sorted(self._vehicle_task_counts.items()):
                        f.write(f"{vid},{count}\n")
                self._force_vehicle_update = False
            except IOError:
                # File might be locked, skip this update
                pass

    def _update_episode_task_count(self, task_count):
        """Update episode-level task statistics in tasks_per_episode.csv"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        episode_file = self.log_dir / "tasks_per_episode.csv"
        
        # Initialize episode counters if not exists
        if not hasattr(self, '_episode_stats'):
            self._episode_stats = {
                'total_tasks': 0,
                'max_tasks_per_step': 0,
                'step_count': 0
            }
        
        # Update statistics
        self._episode_stats['total_tasks'] += task_count
        self._episode_stats['max_tasks_per_step'] = max(self._episode_stats['max_tasks_per_step'], task_count)
        self._episode_stats['step_count'] += 1
        
        # Calculate average
        avg_tasks = self._episode_stats['total_tasks'] / self._episode_stats['step_count'] if self._episode_stats['step_count'] > 0 else 0
        
        # Initialize episode tracking if not exists
        if not hasattr(self, 'episode_number'):
            self.episode_number = 1
        
        # Write/update the CSV file - append mode for multiple episodes
        if not episode_file.exists():
            with open(episode_file, 'w') as f:
                f.write("episode,total_tasks,max_tasks_per_step,avg_tasks_per_step\n")
        
        # Read existing episodes to avoid duplicates
        existing_episodes = set()
        if episode_file.exists():
            try:
                with open(episode_file, 'r') as f:
                    lines = f.readlines()[1:]  # Skip header
                    for line in lines:
                        if line.strip():
                            ep_num = int(line.split(',')[0])
                            existing_episodes.add(ep_num)
            except:
                pass
        
        # Update current episode data
        all_episodes = {}
        if episode_file.exists():
            try:
                with open(episode_file, 'r') as f:
                    lines = f.readlines()[1:]  # Skip header
                    for line in lines:
                        if line.strip():
                            parts = line.strip().split(',')
                            ep_num = int(parts[0])
                            if ep_num != self.episode_number:  # Keep other episodes
                                all_episodes[ep_num] = line.strip()
            except:
                pass
        
        # Add current episode
        all_episodes[self.episode_number] = f"{self.episode_number},{self._episode_stats['total_tasks']},{self._episode_stats['max_tasks_per_step']},{avg_tasks:.2f}"
        
        # Write all episodes
        with open(episode_file, 'w') as f:
            f.write("episode,total_tasks,max_tasks_per_step,avg_tasks_per_step\n")
            for ep_num in sorted(all_episodes.keys()):
                f.write(all_episodes[ep_num] + "\n")

    def _write_offload_to_csv(self, offload_data):
        """Write offload data immediately to CSV"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        offload_file = self.log_dir / "task_offloading_log.csv"
        if not offload_file.exists():
            # Write header
            with open(offload_file, 'w') as f:
                keys = list(offload_data.keys())
                f.write(",".join(keys) + "\n")
        
        with open(offload_file, 'a') as f:
            values = [str(offload_data.get(k, '')) for k in offload_data.keys()]
            f.write(",".join(values) + "\n")

    def _clear_memory_periodically(self):
        """Clear memory lists every 1000 steps to prevent memory bloat"""
        if hasattr(self, 'global_step_count') and self.global_step_count % 1000 == 0:
            # Keep only summary data we need, clear the rest
            self._task_rows = self._task_rows[-100:]  # Keep last 100 for summary
            self._offload_rows = self._offload_rows[-100:]
            self._step_rows = self._step_rows[-100:]

    def _clear_log_files(self):
        """Clear all log files on fresh training start"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear main log files
        log_files = [
            "step_rewards.csv",
            "task_generation_log.csv", 
            "task_offloading_log.csv",
            "tasks_per_vehicle.csv",
            "tasks_per_episode.csv"
        ]
        
        for filename in log_files:
            file_path = self.log_dir / filename
            file_path.unlink(missing_ok=True)  # Delete if exists, ignore if not

        # Also clear the rewards progression file
        rewards_progression_dir = self.log_dir.parent / "Rewards_progression"
        if rewards_progression_dir.exists():
            rewards_file = rewards_progression_dir / "dqn_rewards.csv"
            rewards_file.unlink(missing_ok=True)

    def _flush_logs(self):
        """Force final updates to summary files at episode end"""
        # Force update vehicle counts CSV to ensure latest data is written
        if hasattr(self, '_vehicle_task_counts') and self._vehicle_task_counts:
            self._force_vehicle_update = True
            # Trigger one final vehicle count update
            vehicle_ids = list(self._vehicle_task_counts.keys())
            if vehicle_ids:
                self._update_vehicle_task_count(vehicle_ids[0])  # Force update without incrementing
                self._vehicle_task_counts[vehicle_ids[0]] -= 1  # Undo the increment

    def _compute_task_reward(self, d: dict, include_info=False) -> float:
        cfg = self.reward_cfg
        w   = cfg["weights"]
        eps = cfg["eps"]

        # Robust guards for outage/infinite delays
        total_delay = float(d.get("total_delay_s", math.inf))
        deadline    = max(float(d.get("deadline_s", 0.0)), eps)
        energy_user = float(d.get("energy_user_j", 0.0))
        energy_sys  = float(d.get("energy_system_j", energy_user))
        cpu_cycles  = float(d.get("cpu_cycles", 0.0))

        # DRL overhead for making decision
        if self.include_drl_overhead:
            try:
                rsu_id, rsu_dist = self._nearest_rsu(d.get("vehicle_id"))
                overhead = self._calculate_drl_overhead(d.get("vehicle_id"), rsu_id, rsu_dist)
                total_delay += overhead.get("drl_overhead_s")
                energy_user += overhead.get("drl_energy_vehicle_j")
                energy_sys += overhead.get("drl_energy_rsu_j")

                #print(f"Base RSU delay={float(d.get("total_delay_s", math.inf))}s, "
                 #   f"Overhead={overhead.get("drl_overhead_s")}s, "
                  #  f"Total with overhead={total_delay}s, ")
                

            except:
                # TODO

                pass

        # Baseline energy if done locally by user (single-core)
        local_energy_ref = max(cpu_cycles * self.energy_per_cycle_j_veh, eps)

        # Normalizations
        delay_norm = 2.0 if not math.isfinite(total_delay) else min(2.0, total_delay / deadline)
        eu_ratio   = min(3.0, energy_user / local_energy_ref)
        es_ratio   = min(3.0, energy_sys  / local_energy_ref)

        # Core reward terms
        sla_term   = (1.0 if bool(d.get("deadline_met", False)) else -cfg["miss_penalty"])
        r = (w["sla"] * sla_term
            - w["delay"]        * delay_norm
            - w["energy_user"]  * eu_ratio
            - w["energy_system"]* es_ratio)

        # PPO-specific reward shaping for better learning
        if cfg.get("ppo_mode", False):
            # Base positive reward for any valid decision
            r += cfg.get("base_reward", 2.0)
            
            # Efficiency bonus: reward energy savings compared to local execution
            if energy_user < local_energy_ref:
                efficiency_ratio = 1.0 - (energy_user / local_energy_ref)
                r += cfg.get("efficiency_bonus", 1.0) * efficiency_ratio

        # Category scaling
        cat = str(d.get("category", "autonomous"))
        r *= float(cfg["category_weight"].get(cat, 1.0))

        # Clip
        r = float(np.clip(r, cfg["clip_min"], cfg["clip_max"]))
        if not np.isfinite(r):
            r = 0.0
        if include_info:
            return r, {'task_delay_s': total_delay}
        return r
    
    def _evaluate_candidates(self, task, live_ids):
        """
        Build candidate dicts for local, v2v, rsu WITHOUT committing.
        Mirrors your _decide_and_schedule() computations, but returns:
        {"local": cand or None, "v2v": cand or None, "rsu": cand or None}
        Each cand must include the same fields used for reward + a '_sched_sim' tuple
        so we can commit later (veh/rsu id, arrival_time, cpu_cycles, parallel params).
        """

        # now = float(self.step_count)
        now = self._get_sim_step()
        vid = task["vehicle_id"]
        cpu_cycles  = float(task["cpu_cycles"])
        data_bytes  = int(max(1.0, float(task["data_bytes"])))
        result_bytes = int(max(1.0, self.result_size_ratio * data_bytes))
        cat = task["category"]

        self._ensure_vehicle_spec(vid)

        # ---- LOCAL (simulate multi-threaded) ----
        P_loc, kmax_loc, eta_loc = self._parallel_params(cat, "veh")
        l_start, l_proc, l_finish, l_queue, l_core_ids, l_k = self._schedule_multi_on_vehicle(
            vid, now, cpu_cycles, P_loc, kmax_loc, eta_loc, self.thread_overhead_s, commit=False
        )
        cand_local = {
            "mode": "local", "node_id": vid,
            "tx_up_s": 0.0, "rx_up_s_exec": 0.0,
            "queue_s": l_queue, "proc_s": l_proc,
            "tx_down_s": 0.0, "rx_down_s_user": 0.0,
            "finish_s": l_finish, "delay_s": l_finish - now,
            "tx_up_rate_mbps": 0.0, "tx_down_rate_mbps": 0.0,
            "distance_m": 0.0,"ul_distance_m": 0.0,"dl_distance_m": 0.0,
            "ul_snr_db": 0.0, "ul_cqi": 0, "dl_snr_db": 0.0, "dl_cqi": 0,
            "e_user_comp_j": cpu_cycles * self.energy_per_cycle_j_veh,
            "e_user_tx_up_j": 0.0, "e_user_rx_down_j": 0.0,
            "e_exec_rx_up_j": 0.0, "e_exec_comp_j": 0.0, "e_exec_tx_down_j": 0.0,
            "executor_core_id": (l_core_ids[0] if l_k == 1 else -1),
            "executor_core_ids": l_core_ids, "executor_k_used": l_k,
            "_sched_sim": ("veh_local", vid, now, cpu_cycles, P_loc, kmax_loc, eta_loc),
            "ul_distance_m": 0.0,       # distance at UL time
            "dl_distance_m": 0.0    # distance at DL start time
        }

        # ---- V2V (if peer exists) ----
        cand_v2v = None
        peer_id = self._select_peer(vid, self.traci.vehicle.getIDList(), self.v2v_range_m,
                                    now, cpu_cycles, data_bytes, cat)
        if peer_id is not None:
            self._ensure_vehicle_spec(peer_id)
            d = self._dist_veh(vid, peer_id)
            ul = one_nr_sidelink_hop(d, data_bytes, self.nr_radio_veh, self.nr_radio_veh, self.nr_params_veh, self.rng_radio)
            P_v2v, kmax_v2v, eta_v2v = self._parallel_params(cat, "veh")
            arrival_peer = now + ul["tx_time_s"]
            p_start, p_proc, p_finish, p_queue, p_core_ids, p_k = self._schedule_multi_on_vehicle(
                peer_id, arrival_peer, cpu_cycles, P_v2v, kmax_v2v, eta_v2v, self.thread_overhead_s, commit=False
            )

            t_dl_start = float(p_finish)

            xy_user = self._predict_xy_new(vid, t_dl_start)
            xy_peer = self._predict_xy_new(peer_id, t_dl_start)

            if xy_user is not None and xy_peer is not None:
                dx = xy_user[0] - xy_peer[0]
                dy = xy_user[1] - xy_peer[1]
                d_dl = math.hypot(dx, dy)
            else:
                d_dl = d  # fallback

            
            dl = one_nr_sidelink_hop(d_dl, result_bytes, self.nr_radio_veh, self.nr_radio_veh, self.nr_params_veh, self.rng_radio)
            cand_v2v = {
                "mode":"v2v","node_id":peer_id,
                "tx_up_s":ul["tx_time_s"], "rx_up_s_exec":ul["tx_time_s"],
                "queue_s":p_queue, "proc_s":p_proc,
                "tx_down_s":dl["tx_time_s"], "rx_down_s_user":dl["tx_time_s"],
                "finish_s":p_finish + dl["tx_time_s"],
                "delay_s":(p_finish + dl["tx_time_s"]) - now,
                "tx_up_rate_mbps":ul["rate_mbps"], "tx_down_rate_mbps":dl["rate_mbps"],
                "distance_m":d, "ul_snr_db":ul["snr_db"], "ul_cqi":ul["cqi"], "dl_snr_db":dl["snr_db"], "dl_cqi":dl["cqi"],
                "e_user_comp_j":0.0,
                "e_user_tx_up_j":ul["e_tx_j"], "e_user_rx_down_j":dl["e_rx_j"],
                "e_exec_rx_up_j":ul["e_rx_j"], "e_exec_comp_j":cpu_cycles * self.energy_per_cycle_j_veh, "e_exec_tx_down_j":dl["e_tx_j"],
                "executor_core_id": (p_core_ids[0] if p_k == 1 else -1),
                "executor_core_ids": p_core_ids, "executor_k_used": p_k,
                "_sched_sim": ("veh", peer_id, arrival_peer, cpu_cycles, P_v2v, kmax_v2v, eta_v2v),
                "ul_distance_m": float(d),       # distance at UL time
                "dl_distance_m": float(d_dl)    # distance at DL start time
            }

        # ---- RSU (if coverage exists) ----
        cand_rsu = None
        rsu_pick = self._nearest_rsu(vid)
        if rsu_pick is not None:
            rid, d = rsu_pick
            ul = one_nr_sidelink_hop(d, data_bytes, self.nr_radio_veh, self.nr_radio_rsu, self.nr_params_rsu, self.rng_radio)
            P_rsu, kmax_rsu, eta_rsu = self._parallel_params(cat, "rsu")

            arrival_rsu = now + ul["tx_time_s"]
            r_start, r_proc, r_finish, r_queue, r_core_ids, r_k = self._schedule_multi_on_rsu(
                rid, arrival_rsu, cpu_cycles, P_rsu, kmax_rsu, eta_rsu, self.thread_overhead_s, commit=False
            )
            
            # after you computed UL and scheduled compute at RSU:
            t_now     = float(self._get_sim_step())  # or self.step_count if you prefer
            t_ul_end  = t_now + float(ul["tx_time_s"])
            # r_start, r_proc, r_finish, r_queue, ... already computed by your RSU scheduler
            t_dl_start = float(r_finish)

            # distance at download start: vehicle moved; RSU is fixed
            veh_xy_at_dl = self._predict_xy(vid, t_dl_start)
            if veh_xy_at_dl is not None:
                vx, vy = veh_xy_at_dl
                rsu_spec = self.rsu_spec[rid]
                dx = vx - float(rsu_spec["x"])
                dy = vy - float(rsu_spec["y"])
                d_dl = math.hypot(dx, dy)
            else:
                d_dl = d  # fallback to distance measured at decision time
            
            # Vehicle moved; RSU fixed → recalc DL distance at t_dl_start
            # vx, vy = self._predict_xy(vid, t_dl_start)  # CSV: interpolates (x,y)
            # rsu_x, rsu_y = float(self.rsu_spec[rid]["x"]), float(self.rsu_spec[rid]["y"])
            # d_dl = math.hypot(vx - rsu_x, vy - rsu_y)

            d_dl = self._dl_distance_rsu(vid, rid, t_dl_start)

            dl = one_nr_sidelink_hop(d_dl, result_bytes, self.nr_radio_rsu, self.nr_radio_veh, self.nr_params_rsu, self.rng_radio)
            cand_rsu = {
                "mode":"rsu","node_id":rid,
                "tx_up_s":ul["tx_time_s"], "rx_up_s_exec":ul["tx_time_s"],
                "queue_s":r_queue, "proc_s":r_proc,
                "tx_down_s":dl["tx_time_s"], "rx_down_s_user":dl["tx_time_s"],
                "finish_s":r_finish + dl["tx_time_s"],
                "delay_s":(r_finish + dl["tx_time_s"]) - now,
                "tx_up_rate_mbps":ul["rate_mbps"], "tx_down_rate_mbps":dl["rate_mbps"],
                "distance_m":d, "ul_snr_db":ul["snr_db"], "ul_cqi":ul["cqi"], "dl_snr_db":dl["snr_db"], "dl_cqi":dl["cqi"],
                "e_user_comp_j":0.0,
                "e_user_tx_up_j":ul["e_tx_j"], "e_user_rx_down_j":dl["e_rx_j"],
                "e_exec_rx_up_j":ul["e_rx_j"], "e_exec_comp_j":cpu_cycles * self.energy_per_cycle_j_rsu, "e_exec_tx_down_j":dl["e_tx_j"],
                "executor_core_id": (r_core_ids[0] if r_k == 1 else -1),
                "executor_core_ids": r_core_ids, "executor_k_used": r_k,
                "_sched_sim": ("rsu", rid, arrival_rsu, cpu_cycles, P_rsu, kmax_rsu, eta_rsu),
                "ul_distance_m": float(d),       # distance at UL time
                "dl_distance_m": float(d_dl)    # distance at DL start time
            }

        return {"local": cand_local, "v2v": cand_v2v, "rsu": cand_rsu}
    
    def _commit_candidate(self, cand: dict, task: dict) -> dict:
        """
        Commit a previously evaluated candidate to real queues and return the log row.
        - `cand` is one of the dicts returned by _evaluate_candidates(...).
        - `task` is the original task record (must include vehicle_id, category, cpu_cycles, data_bytes, deadline_s).
        """

        tid = task.get("task_id")
        if tid is None:
            # fallback: derive one if old tasks lacked ID
            pos_step = int(getattr(self, "_pos_step_current", self.step_count))
            self._task_uid += 1
            tid = f"{pos_step}:{task['vehicle_id']}:{self._task_uid}"
            task["task_id"] = tid

        # prevent double-commit
        if tid in self._committed_task_ids:
            # already committed earlier this step by core or wrapper, skip
            return {
                "step": int(self.global_step_count),
                "vehicle_id": task["vehicle_id"],
                "category": task["category"],
                "task_id": tid,
                "decision": "duplicate_skip"
            }
        self._committed_task_ids.add(tid)




        # Defensive fallback: if cand is None, do local
        if cand is None:
            # Minimal local fallback (single-threaded compute if needed)
            vid = task["vehicle_id"]
            cat = task["category"]
            # now = float(self.step_count)
            now = self._get_sim_step
            cpu_cycles = float(task["cpu_cycles"])
            self._ensure_vehicle_spec(vid)
            P, kmax, eta = self._parallel_params(cat, "veh")
            l_start, l_proc, l_finish, l_queue, l_core_ids, l_k = self._schedule_multi_on_vehicle(
                vid, now, cpu_cycles, P, kmax, eta, self.thread_overhead_s, commit=True
            )
            out = {
                "step": int(self.global_step_count),
                "vehicle_id": vid,
                "category": cat,
                "cpu_cycles": cpu_cycles,
                "data_bytes": int(task["data_bytes"]),
                "result_bytes": int(max(1, self.result_size_ratio * int(task["data_bytes"]))),
                "deadline_s": float(task["deadline_s"]),
                "decision": "local",
                "node_id": vid,

                "tx_up_time_s": 0.0,
                "rx_up_time_s_executor": 0.0,
                "queue_delay_s": l_queue,
                "proc_time_s": l_proc,
                "tx_down_time_s": 0.0,
                "rx_down_time_s_user": 0.0,
                "finish_time_s": l_finish,
                "total_delay_s": l_finish - now,
                "deadline_met": bool((l_finish - now) <= float(task["deadline_s"])),

                "tx_up_rate_mbps": 0.0,
                "tx_down_rate_mbps": 0.0,
                "distance_m": 0.0,
                "ul_distance_m": 0.0,       # distance at UL time
                "dl_distance_m": 0.0,    # distance at DL start time
                "ul_snr_db": 0.0, "ul_cqi": 0,
                "dl_snr_db": 0.0, "dl_cqi": 0,

                "e_user_comp_j": cpu_cycles * self.energy_per_cycle_j_veh,
                "e_user_tx_up_j": 0.0,
                "e_user_rx_down_j": 0.0,
                "e_exec_rx_up_j": 0.0,
                "e_exec_comp_j": 0.0,
                "e_exec_tx_down_j": 0.0,
                "energy_user_j": cpu_cycles * self.energy_per_cycle_j_veh,
                "energy_executor_j": 0.0,
                "energy_system_j": cpu_cycles * self.energy_per_cycle_j_veh,

                "executor_core_id": (l_core_ids[0] if l_k == 1 else -1),
                "executor_core_ids": l_core_ids,
                "executor_k_used": l_k,
                "env_step": int(self.step_count),
                "sim_step": int(self._get_sim_step()),

            }
            self._offload_rows.append(out)
            # self._write_offload_to_csv(out)  # Disabled to save disk space during training
            return out

        # Normal path
        # now = float(self.step_count)
        now = self._get_sim_step()
        vid = task["vehicle_id"]
        cat = task["category"]
        cpu_cycles = float(task["cpu_cycles"])
        data_bytes = int(task["data_bytes"])
        result_bytes = int(max(1, self.result_size_ratio * data_bytes))
        deadline_s = float(task["deadline_s"])

        mode = cand["mode"]
        node_id = cand["node_id"]

        # Unpack scheduling tuple from candidate and COMMIT on the chosen executor
        # _sched_sim format: (kind, node_id, arrival_time, cpu_cycles, P, kmax, eta)
        kind, sim_node, arr_t, cyc, P, kmax, eta = cand.get("_sched_sim", (None, None, None, None, 0.0, 1, self.parallel_eta_rsu))

        if mode == "local":
            # Commit on requester vehicle
            self._ensure_vehicle_spec(vid)
            start, proc, finish, queue, core_ids, k_used = self._schedule_multi_on_vehicle(
                vid, arr_t, cyc, P, kmax, eta, self.thread_overhead_s, commit=True
            )
        elif kind == "veh":
            # Commit on peer vehicle
            self._ensure_vehicle_spec(sim_node)
            start, proc, finish, queue, core_ids, k_used = self._schedule_multi_on_vehicle(
                sim_node, arr_t, cyc, P, kmax, eta, self.thread_overhead_s, commit=True
            )
        elif kind == "rsu":
            # Commit on RSU
            start, proc, finish, queue, core_ids, k_used = self._schedule_multi_on_rsu(
                sim_node, arr_t, cyc, P, kmax, eta, self.thread_overhead_s, commit=True
            )
        else:
            # If somehow missing, fall back to local commit
            self._ensure_vehicle_spec(vid)
            P2, kmax2, eta2 = self._parallel_params(cat, "veh")
            start, proc, finish, queue, core_ids, k_used = self._schedule_multi_on_vehicle(
                vid, now, cpu_cycles, P2, kmax2, eta2, self.thread_overhead_s, commit=True
            )
            mode = "local"
            node_id = vid

        # Rebuild timing with committed compute; radio pieces come from cand
        tx_up_s   = float(cand.get("tx_up_s", 0.0))
        rx_up_s_e = float(cand.get("rx_up_s_exec", tx_up_s))
        tx_dn_s   = float(cand.get("tx_down_s", 0.0))
        rx_dn_s_u = float(cand.get("rx_down_s_user", tx_dn_s))

        # For total finish, keep: finish_exec (finish) + downlink TX
        finish_total = float(finish + tx_dn_s)
        total_delay  = float(finish_total - now)

        # Energies from cand (radio) + compute energy depending on executor kind
        e_user_tx_up   = float(cand.get("e_user_tx_up_j", 0.0))
        e_user_rx_down = float(cand.get("e_user_rx_down_j", 0.0))
        e_exec_rx_up   = float(cand.get("e_exec_rx_up_j", 0.0))
        e_exec_tx_down = float(cand.get("e_exec_tx_down_j", 0.0))

        if mode == "local":
            e_user_comp = cpu_cycles * self.energy_per_cycle_j_veh
            e_exec_comp = 0.0
        elif mode == "v2v":
            e_user_comp = 0.0
            e_exec_comp = cpu_cycles * self.energy_per_cycle_j_veh
        else:  # rsu
            e_user_comp = 0.0
            e_exec_comp = cpu_cycles * self.energy_per_cycle_j_rsu

        energy_user   = e_user_comp + e_user_tx_up + e_user_rx_down
        energy_exec   = e_exec_comp + e_exec_rx_up + e_exec_tx_down
        energy_system = energy_user + energy_exec

        out = {
            "step": int(self.global_step_count),
            "task_id": tid,
            "pos_step": int(getattr(self, "_pos_step_current", self.step_count)),
            "vehicle_id": vid,
            "category": cat,
            "cpu_cycles": cpu_cycles,
            "data_bytes": data_bytes,
            "result_bytes": result_bytes,
            "deadline_s": deadline_s,
            "decision": mode,
            "node_id": node_id,

            # timing breakdown (with committed compute)
            "tx_up_time_s": tx_up_s,
            "rx_up_time_s_executor": rx_up_s_e,
            "queue_delay_s": float(queue),
            "proc_time_s": float(proc),
            "tx_down_time_s": tx_dn_s,
            "rx_down_time_s_user": rx_dn_s_u,
            "finish_time_s": finish_total,
            "total_delay_s": total_delay,
            "deadline_met": bool(total_delay <= deadline_s),

            # link info (copied from candidate)
            "tx_up_rate_mbps": float(cand.get("tx_up_rate_mbps", 0.0)),
            "tx_down_rate_mbps": float(cand.get("tx_down_rate_mbps", 0.0)),
            "distance_m": float(cand.get("distance_m", 0.0)),
            "ul_distance_m": float(cand.get("ul_distance_m",0.0)),       # distance at UL time
            "dl_distance_m": float(cand.get("dl_distance_m",0.0)),    # distance at DL start time
            "ul_snr_db": float(cand.get("ul_snr_db", 0.0)),
            "ul_cqi": int(cand.get("ul_cqi", 0)),
            "dl_snr_db": float(cand.get("dl_snr_db", 0.0)),
            "dl_cqi": int(cand.get("dl_cqi", 0)),

            # energy breakdown
            "e_user_comp_j": float(e_user_comp),
            "e_user_tx_up_j": e_user_tx_up,
            "e_user_rx_down_j": e_user_rx_down,
            "e_exec_rx_up_j": e_exec_rx_up,
            "e_exec_comp_j": float(e_exec_comp),
            "e_exec_tx_down_j": e_exec_tx_down,

            "energy_user_j": float(energy_user),
            "energy_executor_j": float(energy_exec),
            "energy_system_j": float(energy_system),

            # core usage (scalar + list, consistent with your earlier convention)
            "executor_core_id": (core_ids[0] if k_used == 1 else -1),
            "executor_core_ids": core_ids,
            "executor_k_used": int(k_used),
            "env_step": int(self.step_count),
            "sim_step": int(self._get_sim_step()),
        }

        self._offload_rows.append(out)
        # self._write_offload_to_csv(out)  # Disabled to save disk space during training
        return out
    
    # def agent_finalize_tick(self, decisions: list):
    #     """
    #     Called by the DRL wrapper after it has committed all tasks for the current core tick.
    #     Computes the same step summary as the greedy path and appends to self._step_rows.
    #     """

    #     # resolve step and (if available) the true positions step
    #     step = int(self.step_count)
    #     pos_step = int(getattr(self, "_pos_step_current", step))

    #     n = len(decisions)
    #     if n == 0:
    #         # If you prefer to log idle ticks, uncomment below:
    #         # self._step_rows.append({"step": step, "pos_step": pos_step, "step_reward": 0.0, "tasks": 0,
    #         #                         "deadline_met": 0, "deadline_miss": 0, "deadline_met_rate": 0.0,
    #         #                         "mean_delay_s": 0.0, "p95_delay_s": 0.0,
    #         #                         "mean_user_energy_j": 0.0, "mean_system_energy_j": 0.0,
    #         #                         "dec_local": 0, "dec_v2v": 0, "dec_rsu": 0})
    #         return  # skip idle rows by default

    #     delays  = np.array([float(d.get("total_delay_s", 0.0)) for d in decisions], dtype=float)
    #     e_user  = np.array([float(d.get("energy_user_j", 0.0)) for d in decisions], dtype=float)
    #     e_sys   = np.array([float(d.get("energy_system_j", 0.0)) for d in decisions], dtype=float)
    #     sla_hit = np.array([bool(d.get("deadline_met", False)) for d in decisions], dtype=bool)
    #     decs    = [str(d.get("decision", "")) for d in decisions]

    #     # robust stats (drop NaN/Inf)
    #     finite = np.isfinite(delays)
    #     mean_delay = float(delays[finite].mean()) if finite.any() else 0.0
    #     p95_delay  = float(np.quantile(delays[finite], 0.95)) if finite.any() else 0.0
    #     finite_u   = np.isfinite(e_user)
    #     finite_s   = np.isfinite(e_sys)
    #     mean_user_e = float(e_user[finite_u].mean()) if finite_u.any() else 0.0
    #     mean_sys_e  = float(e_sys[finite_s].mean()) if finite_s.any() else 0.0

    #     # step reward = aggregate of per-task rewards (same logic you use elsewhere)
    #     task_rewards = [self._compute_task_reward(d) for d in decisions]
    #     if not task_rewards:
    #         step_reward = 0.0
    #     else:
    #         agg = self.reward_cfg.get("aggregate", "mean")
    #         if agg == "sum":
    #             step_reward = float(sum(task_rewards))
    #         elif agg == "median":
    #             import numpy as _np
    #             step_reward = float(_np.median(task_rewards))
    #         else:
    #             step_reward = float(np.mean(task_rewards))

    #     row = {
    #         "step": step,
    #         "pos_step": pos_step,  # helpful for joining with positions
    #         "step_reward": step_reward,
    #         "tasks": int(n),
    #         "deadline_met": int(sla_hit.sum()),
    #         "deadline_miss": int(n - sla_hit.sum()),
    #         "deadline_met_rate": float(sla_hit.mean()),
    #         "mean_delay_s": mean_delay,
    #         "p95_delay_s": p95_delay,
    #         "mean_user_energy_j": mean_user_e,
    #         "mean_system_energy_j": mean_sys_e,
    #         "dec_local": int(sum(1 for x in decs if x == "local")),
    #         "dec_v2v":   int(sum(1 for x in decs if x == "v2v")),
    #         "dec_rsu":   int(sum(1 for x in decs if x == "rsu")),
    #     }
    #     self._step_rows.append(row)

    def agent_finalize_tick(self, decisions: list):
        import numpy as np

        env_step = int(self.step_count)
        sim_step = int(self._get_sim_step())

        n = len(decisions)
        if n == 0:
            return  # skip idle ticks; or log zeros if you prefer

        delays  = np.array([float(d.get("total_delay_s", 0.0)) for d in decisions], dtype=float)
        e_user  = np.array([float(d.get("energy_user_j", 0.0)) for d in decisions], dtype=float)
        e_sys   = np.array([float(d.get("energy_system_j", 0.0)) for d in decisions], dtype=float)
        sla_hit = np.array([bool(d.get("deadline_met", False)) for d in decisions], dtype=bool)
        decs    = [str(d.get("decision", "")) for d in decisions]

        finite = np.isfinite(delays)
        mean_delay = float(delays[finite].mean()) if finite.any() else 0.0
        p95_delay  = float(np.quantile(delays[finite], 0.95)) if finite.any() else 0.0
        finite_u   = np.isfinite(e_user)
        finite_s   = np.isfinite(e_sys)
        mean_user_e = float(e_user[finite_u].mean()) if finite_u.any() else 0.0
        mean_sys_e  = float(e_sys[finite_s].mean()) if finite_s.any() else 0.0

        task_rewards = [self._compute_task_reward(d) for d in decisions]
        if not task_rewards:
            step_reward = 0.0
        else:
            agg = self.reward_cfg.get("aggregate", "mean")
            if agg == "sum":
                step_reward = float(sum(task_rewards))
            elif agg == "median":
                step_reward = float(np.median(task_rewards))
            else:
                step_reward = float(np.mean(task_rewards))

        row = {
            # --- NEW: explicit step fields ---
            "env_step": env_step,
            "sim_step": sim_step,

            # keep your legacy 'step' too if you like
            "step": int(self.global_step_count),  # Use global step count for consistency

            "step_reward": step_reward,
            "tasks": int(n),
            "deadline_met": int(sla_hit.sum()),
            "deadline_miss": int(n - sla_hit.sum()),
            "deadline_met_rate": float(sla_hit.mean()),
            "mean_delay_s": mean_delay,
            "p95_delay_s": p95_delay,
            "mean_user_energy_j": mean_user_e,
            "mean_system_energy_j": mean_sys_e,
            "dec_local": int(sum(1 for x in decs if x == "local")),
            "dec_v2v":   int(sum(1 for x in decs if x == "v2v")),
            "dec_rsu":   int(sum(1 for x in decs if x == "rsu")),
        }
        self._step_rows.append(row)

        # WRITE TO CSV IMMEDIATELY (for agent mode)
        self._write_step_to_csv(row)  # also writes the full step_summary.csv
        
        # Update episode task statistics
        self._update_episode_task_count(row['tasks'])
        
        # MEMORY OPTIMIZATION: Clear old data periodically
        self._clear_memory_periodically()

    def _sanitize_features(self, feats, clip=1e6):
        x = np.asarray(feats, dtype=np.float32)
        # replace NaN with 0, +inf with +clip, -inf with -clip, then clamp
        x = np.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
        np.clip(x, -clip, clip, out=x)
        return x

    def _close_sumo(self):
        try:
            if self.traci is not None:
                self.traci.close(False)
        except Exception:
            pass
        self.traci = None


    def close(self):
        self._close_sumo()
        super().close() if hasattr(super(), "close") else None


    def close(self):
        self._flush_logs()
        if self.traci is not None:
            try: self.traci.close()
            except Exception: pass
            self.traci = None
        super().close()

    # add anywhere inside the class (e.g., near other small helpers)
    def _get_sim_step(self) -> int:
        """
        Return the simulation step (SUMO time in seconds when using TraCI,
        or replayed 'pos_step' when using the CSV core).
        Falls back to env tick if neither is available.
        """
        # CSV replay provides _pos_step_current
        s = getattr(self, "_pos_step_current", None)
        if s is not None:
            try:
                return int(s)
            except Exception:
                pass
        # Live SUMO provides traci.simulation.getTime()
        try:
            if getattr(self, "traci", None) is not None and getattr(self.traci, "simulation", None) is not None:
                return int(self.traci.simulation.getTime())
        except Exception:
            pass
        # Fallback: env tick
        return int(self.step_count)
    
    def _reset_compute_heaps(self, now: float | None = None):
        if now is None:
            now = 0.0  # absolute SIM seconds baseline for new episode
        # RSU heaps
        for rid, spec in self.rsu_spec.items():
            cores = int(spec.get("cores", 1))
            spec["heap"] = [(now, i) for i in range(cores)]
            heapq.heapify(spec["heap"])
        # Vehicle availability / heaps (if you model per-vehicle cores)
        for v in self.veh_spec.values():
            v["available_at"] = float(now)
            if "cores" in v and "per_core_hz" in v:
                cores = int(v["cores"])
                v["heap"] = [(now, i) for i in range(cores)]
                heapq.heapify(v["heap"])

    def _predict_xy(self, vid: str, sim_step_future: float):
        """
        Return (x,y) for vehicle vid at a future simulation step.
        - CSV replay: uses recorded positions at the nearest integer step.
        - Live SUMO: kinematic extrapolation from current (x,y, speed, angle).
        Fallback: current position.
        """
        # Prefer CSV replay if arrays are present
        if hasattr(self, "_steps_unique") and getattr(self, "_frame_start", None) is not None:
            xy = self._xy_from_csv_linear(vid, sim_step_future)
            if xy is not None:
                return xy

        # Live SUMO / fallback: straight-line extrapolation
        try:
            now = float(self.traci.simulation.getTime())
            dt = max(0.0, float(sim_step_future) - now)
            x0, y0 = self.veh_api.getPosition(vid)
            v = float(self.veh_api.getSpeed(vid))  # m/s
            ang_deg = float(self.veh_api.getAngle(vid))  # SUMO angle: 0° = East, CCW positive
            th = math.radians(ang_deg)
            return (float(x0 + v * dt * math.cos(th)), float(y0 + v * dt * math.sin(th)))
        except Exception:
            # last known (current) position
            return self._get_xy(vid)
        
    def _xy_from_csv_linear(self, vid: str, t: float):
        traj = self._traj.get(str(vid))
        if not traj: return None
        T, X, Y = traj
        idx = np.searchsorted(T, t)
        if idx <= 0:    return float(X[0]),  float(Y[0])
        if idx >= T.size: return float(X[-1]), float(Y[-1])
        t0, t1 = float(T[idx-1]), float(T[idx])
        a = 0.0 if t1 == t0 else (float(t)-t0)/(t1-t0)
        x = X[idx-1]*(1-a) + X[idx]*a
        y = Y[idx-1]*(1-a) + Y[idx]*a
        return float(x), float(y)
    
    # called once per tick in BOTH cores right after you refresh positions
    def _touch_motion_cache(self, live_ids):
        t = self._get_sim_step()
        for vid in live_ids:
            x, y = self._get_pos(vid)  # your current position map
            if x is None: 
                continue
            dq = self._motion_cache.get(vid)
            if dq is None:
                dq = deque(maxlen=2)
                self._motion_cache[vid] = dq
            dq.append( (t, float(x), float(y)) )

    def _predict_xy_new(self, vid: str, t: float):
        """
        Unified predictor for both live SUMO and CSV-replay.
        - If we have the last two samples: constant-velocity extrapolation for t >= last_t,
        and linear interpolation for t in [t0, t1].
        - Else, CSV replay can fall back to trajectory interpolation if available.
        - Else, return current (x,y).
        """
        dq = self._motion_cache.get(vid)
        if dq and len(dq) == 2:
            (t0, x0, y0), (t1, x1, y1) = dq[0], dq[1]
            if t1 == t0:
                return (x1, y1)
            if t <= t1 and t >= t0:
                a = (t - t0) / (t1 - t0)
                return (x0*(1-a) + x1*a, y0*(1-a) + y1*a)
            # constant-velocity extrapolation from last sample
            vx = (x1 - x0) / (t1 - t0)
            vy = (y1 - y0) / (t1 - t0)
            dt = (t - t1)
            return (x1 + vx*dt, y1 + vy*dt)

        # CSV replay fallback (if you built self._traj)
        if hasattr(self, "_traj") and str(vid) in self._traj:
            T, X, Y = self._traj[str(vid)]
            if T.size:
                idx = np.searchsorted(T, t)
                if idx <= 0:  return (float(X[0]), float(Y[0]))
                if idx >= T.size: return (float(X[-1]), float(Y[-1]))
                t0, t1 = float(T[idx-1]), float(T[idx])
                a = 0.0 if t1 == t0 else (t - t0)/(t1 - t0)
                return (float(X[idx-1]*(1-a) + X[idx]*a), float(Y[idx-1]*(1-a) + Y[idx]*a))

        # last resort
        x, y = self._get_pos(vid)
        return (x, y)

    def _dl_distance_rsu(self, vid: str, rid: str, t_dl_start: float) -> float:
        vx, vy = self._predict_xy_new(vid, t_dl_start)
        rx, ry = float(self.rsu_spec[rid]["x"]), float(self.rsu_spec[rid]["y"])
        if vx is None: return float("inf")
        return math.hypot(vx - rx, vy - ry)

    def _calculate_drl_overhead(self, vehicle_id: str, rsu_id: str, distance_m: float) -> Dict[str, float]:
        """
        Returns {
            'drl_overhead_s': float
            'drl_energy_vehicle_j': float,
            'drl_energy_rsu_j': float
        }
        """
        
        # these overheads need to be added to reward function

        METADATA_BYTES = 200 # TODO move this to config file, also come up with better estimate for size
        #   -Task category
        #   -Task cpu cycles
        #   -Task data byte size
        #   -Task delay requirement 
        # DRL Inference Model Parameters
        # TODO move these to config.py
        DRL_BASE_CYCLES = 10e6          # Base cycles for inference
        DRL_CYCLES_PER_OBS = 50e3       # Additional cycles per observation feature
        DRL_METADATA_BYTES = 200         # Task metadata size
        DRL_RESPONSE_BYTES = 50          # Decision response size

        tx_radio = self.nr_radio_veh  # vehicle radio
        rx_radio = self.nr_radio_rsu   # RSU radio

        hop_result = one_nr_sidelink_hop(
            d_m=distance_m,
            payload_bytes=METADATA_BYTES,
            tx=tx_radio,
            rx=rx_radio
        )
      #  hop_result["tx_time_s"] += 0.01
        overhead_dict = {
            "drl_overhead_s": hop_result["tx_time_s"],
            "drl_energy_vehicle_j": hop_result["e_tx_j"],
            "drl_energy_rsu_j": hop_result["e_rx_j"]
        }

        # TODO calculate overhead from running model on RSU - clean this up
        obs_size = self.obs_max_veh * self.obs_per_veh
    
        # Calibrate these based on your actual DRL model architecture
        inference_cycles = DRL_BASE_CYCLES + (obs_size * DRL_CYCLES_PER_OBS)
        
        rsu_freq_hz = self.rsu_spec[rsu_id]['cpu_hz']
        rsu_cores = self.rsu_spec[rsu_id]['cores']
        rsu_freq_per_core = rsu_freq_hz / rsu_cores
        
        inference_time_s = inference_cycles / rsu_freq_per_core
        inference_energy_j = inference_cycles * self.energy_per_cycle_j_rsu


        overhead_dict["drl_overhead_s"] += inference_time_s
        overhead_dict["drl_energy_rsu_j"] += inference_energy_j

        
        print(f"DEBUG Overhead Calculation:")
        print(f"  inference_time_s: {inference_time_s:.6f}s ({inference_time_s*1000:.2f}ms)")
        print(f"  transmission_time_s: {hop_result['tx_time_s']:.6f}s ({hop_result['tx_time_s']*1000:.2f}ms)")
        print(f"  TOTAL overhead: {overhead_dict['drl_overhead_s']:.6f}s ({overhead_dict['drl_overhead_s']*1000:.2f}ms)")
        
        
        return overhead_dict
        # calculate latency from transmitting task metadata and receiving offload decision response
            # this should use the current communication model
            # just need to estimate the data size

        # calculate energy consumption from transmitting 
            # should use current communication model

        # calculate time to make decision on RSU
            # need to model RSU model performance
    

# Small util to handle pandas NA without importing pandas globally at top
def pd_isna(x):
    try:
        import pandas as pd
        return pd.isna(x)
    except Exception:
        return x is None
