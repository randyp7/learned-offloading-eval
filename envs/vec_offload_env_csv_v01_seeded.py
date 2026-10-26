# envs/vec_offload_env_csv_v01_seeded.py
"""
CSV-replay core environment.

Replays a pre-recorded CSV of vehicle positions (output of extract_positions.py)
instead of driving SUMO via TraCI.  Inherits all task-generation, scheduling,
reward, and feature-building logic from the SUMO core (VECEnvOffload).

CSV format (columns, case-insensitive):
    time (or step), vehicle_id, x, y [, speed]

Usage
-----
Create via env_factory by passing pos_csv=<path>.
"""
import math
import heapq
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from envs.vec_env_offload_nrv2x_m_v01_seeded import VECEnvOffload as CoreBase
from config import RSU_CORES_DEFAULT, RSU_PER_CORE_HZ_DEFAULT


# ── Fake TraCI shim ───────────────────────────────────────────────────────────

class _FakeVehicleAPI:
    def __init__(self, outer): self.outer = outer
    def getIDList(self) -> List[str]:   return list(self.outer._live_ids)
    def getPosition(self, vid: str):    return self.outer._pos_current.get(vid, (0.0, 0.0))
    def getSpeed(self, vid: str) -> float: return self.outer._spd_current.get(vid, 0.0)

class _FakeSimAPI:
    """Minimal simulation object — only used for getTime() in some paths."""
    def __init__(self, outer): self.outer = outer
    def getTime(self) -> float: return float(self.outer._pos_step_current)

class _FakeTraci:
    def __init__(self, outer):
        self.vehicle    = _FakeVehicleAPI(outer)
        self.simulation = _FakeSimAPI(outer)


# ── Main class ────────────────────────────────────────────────────────────────

class VECEnvOffloadCSV(CoreBase):
    """
    Drop-in replacement for VECEnvOffload that replays a position CSV instead
    of running SUMO.  The CSV must cover a continuous time range; each episode
    samples a random window of `max_steps` consecutive frames.

    Fragment mode: if `pos_csv` points to a directory, the env expects it to
    contain multiple fragment CSVs (e.g. from extract_positions_sumo.py
    --fragments N).  A random fragment file is chosen at every episode reset,
    giving temporal diversity across the day without loading all files at once.
    """

    def __init__(self, pos_csv: str, *args, rsu_sample_k: int = 0,
                 fragment_hour_range: Optional[tuple] = None, **kwargs):
        kwargs = dict(kwargs)
        kwargs.setdefault("gui", False)
        # sumo_cfg is required by parent but unused in CSV mode — pass a dummy
        if not args and 'sumo_cfg' not in kwargs:
            kwargs['sumo_cfg'] = '/dev/null'
        super().__init__(*args, **kwargs)

        self.rsu_sample_k = int(rsu_sample_k)  # 0 = use all RSUs (original behaviour)
        self._episode_rsu_indices: Optional[np.ndarray] = None  # set in reset()
        self.fragment_hour_range = fragment_hour_range  # (h0, h1) or None

        self.pos_csv = str(pos_csv)

        # Fragment-directory support
        _p = Path(pos_csv)
        if _p.is_dir():
            import re as _re
            all_frags = [f for f in sorted(_p.glob("*.csv")) if f.stat().st_size > 0 and not f.name.startswith("_")]  # skip _stats.csv etc.
            if fragment_hour_range is not None:
                # Filter to fragments whose start time (in seconds, from _tXXXXX in filename)
                # falls within [start_hour, end_hour) × 3600.
                h0, h1 = fragment_hour_range
                t_min, t_max = h0 * 3600, h1 * 3600
                def _frag_start(f):
                    m = _re.search(r'_t(\d+)', f.stem)
                    return int(m.group(1)) if m else 0
                self._frag_files = [f for f in all_frags if t_min <= _frag_start(f) < t_max]
                if not self._frag_files:
                    raise ValueError(
                        f"No fragments in hour range {h0}-{h1} found in {pos_csv}")
                print(f"[VECEnvOffloadCSV] Fragment mode: {len(self._frag_files)}/{len(all_frags)} "
                      f"files in {pos_csv} (hours {h0}–{h1})")
            else:
                self._frag_files = all_frags
                if not self._frag_files:
                    raise ValueError(f"No CSV files found in fragment directory: {pos_csv}")
                print(f"[VECEnvOffloadCSV] Fragment mode: {len(self._frag_files)} files in {pos_csv}")
        else:
            self._frag_files = None  # single-file mode

        # Track which file is currently loaded
        self._current_csv_path: Optional[str] = None

        # In-memory frame data — reloaded when fragment changes
        self._csv_loaded   = False
        self._steps_unique: np.ndarray = np.array([], dtype=np.int64)
        self._frame_start:  np.ndarray = np.array([], dtype=np.int64)
        self._frame_end:    np.ndarray = np.array([], dtype=np.int64)
        self._col_step: np.ndarray = np.array([], dtype=np.int64)
        self._col_vid:  np.ndarray = np.array([], dtype=object)
        self._col_x:    np.ndarray = np.array([], dtype=np.float32)
        self._col_y:    np.ndarray = np.array([], dtype=np.float32)
        self._col_spd:  np.ndarray = np.array([], dtype=np.float32)

        # Per-step live state
        self._live_ids:    List[str] = []
        self._pos_current: Dict[str, Tuple[float, float]] = {}
        self._spd_current: Dict[str, float] = {}
        self._frame_idx  = 0  # index into _steps_unique for current episode window
        self._window_start_idx = 0  # index in _steps_unique where this episode starts

        # Spatial index (rebuilt each frame in _load_frame)
        self._kdtree:      Optional[cKDTree] = None
        self._kdtree_vids: List[str] = []

        # Replace TraCI with shim
        self.traci   = _FakeTraci(self)
        self.veh_api = self.traci.vehicle

        # Episode RNG (separate from task RNG — for window sampling)
        if not hasattr(self, '_episode_rng'):
            self._episode_rng = np.random.RandomState(self.seed_val + 999)
        self._episode_salt = 0

        # Ensure global counters exist
        if not hasattr(self, 'global_step_count'):
            self.global_step_count = 0
        self.step_count = 0

        # Build RSU specs from the RSU CSV (no SUMO needed)
        self._build_rsu_specs()

        # KD-tree of fixed RSU positions (built once; used to filter task generation)
        self._rsu_kdtree: Optional[cKDTree] = None
        self._build_rsu_kdtree()

        self._obs_stub = np.zeros((self.observation_space.shape[0],), dtype=np.float32)

    # ── CSV loading (once) ────────────────────────────────────────────────────

    def _load_csv_once(self):
        if self._csv_loaded:
            return
        path = self._current_csv_path or self.pos_csv
        print(f"[VECEnvOffloadCSV] Loading positions CSV: {path} ...", flush=True)
        df = pd.read_csv(path)
        # normalise column names
        df.columns = [c.lower().strip() for c in df.columns]
        # accept 'time' as alias for 'step'
        if 'time' in df.columns and 'step' not in df.columns:
            df = df.rename(columns={'time': 'step'})
        # accept 'speed_mps' as alias for 'speed' (earlier pre-generated corridor CSVs)
        if 'speed_mps' in df.columns and 'speed' not in df.columns:
            df = df.rename(columns={'speed_mps': 'speed'})
        for req in ('step', 'vehicle_id', 'x', 'y'):
            if req not in df.columns:
                raise ValueError(f"CSV missing required column '{req}'. Found: {list(df.columns)}")

        # Pull raw numpy arrays once ('time' may be float like "1.0" -> via float).
        step = pd.to_numeric(df['step']).to_numpy(np.float64).astype(np.int64)
        # vehicle_id as categorical: integer codes + a small unique-label table.
        # Avoids materialising millions of Python str objects (the memory hog on
        # full-Berlin fragments: ~19M rows would cost ~2GB as strings).
        cat = df['vehicle_id'].astype('category')
        codes = cat.cat.codes.to_numpy()                    # int32 per row (cheap)
        labels = cat.cat.categories.astype(str).to_numpy()  # small: one per vehicle
        x = df['x'].to_numpy(np.float32)
        y = df['y'].to_numpy(np.float32)
        spd = (df['speed'].to_numpy(np.float32) if 'speed' in df.columns
               else np.zeros(len(df), np.float32))
        del df  # release the DataFrame early

        # --- Frame arrays: sort by step only (stable). Integer sort is fast. ---
        order = np.argsort(step, kind='stable')
        self._col_step = step[order]
        # _col_vid holds the string id per row (object array of references into
        # the small label table — not millions of distinct str allocations).
        self._col_vid  = labels[codes[order]]
        self._col_x    = x[order]
        self._col_y    = y[order]
        self._col_spd  = spd[order]

        change = np.r_[True, self._col_step[1:] != self._col_step[:-1]]
        starts = np.flatnonzero(change)
        ends   = np.r_[starts[1:], self._col_step.size]
        self._steps_unique = self._col_step[starts]
        self._frame_start  = starts
        self._frame_end    = ends

        # --- Per-vehicle trajectories for _predict_xy ---
        # Vectorised: lexsort by (code, step), slice contiguous per-vehicle blocks.
        # Avoids the slow string sort + Python-level pandas groupby (was O(minutes)
        # on full-Berlin CSVs).
        vorder = np.lexsort((step, codes))                   # primary=codes, secondary=step
        c_s = codes[vorder]
        T_all = step[vorder]
        X_all = x[vorder]
        Y_all = y[vorder]
        vchange = np.r_[True, c_s[1:] != c_s[:-1]]
        vstarts = np.flatnonzero(vchange)
        vends   = np.r_[vstarts[1:], c_s.size]
        self._traj = {}
        for s, e in zip(vstarts.tolist(), vends.tolist()):
            self._traj[str(labels[c_s[s]])] = (T_all[s:e], X_all[s:e], Y_all[s:e])

        self._csv_loaded = True
        n_frames = len(self._steps_unique)
        n_rows = len(self._col_step)
        avg_veh  = n_rows / max(n_frames, 1)
        print(f"  Loaded {n_rows:,} rows, {n_frames} frames, "
              f"t={self._steps_unique[0]}–{self._steps_unique[-1]}s, "
              f"avg {avg_veh:.0f} vehicles/frame", flush=True)

    # ── RSU specs ─────────────────────────────────────────────────────────────

    def _build_rsu_specs(self):
        self.rsu_spec.clear()
        if not hasattr(self, 'df_rsu') or self.df_rsu is None:
            raise RuntimeError("df_rsu not loaded — pass rsu_csv to constructor.")
        for _, row in self.df_rsu.iterrows():
            rid = str(row['rsu_id'])
            x, y = float(row['x']), float(row['y'])
            try:
                cpu_total = float(row['cpu_hz']) if pd.notna(row.get('cpu_hz')) else None
            except Exception:
                cpu_total = None
            cores = int(row['cores']) if (pd.notna(row.get('cores')) and str(row.get('cores', '')).strip()) else int(RSU_CORES_DEFAULT)
            if pd.notna(row.get('per_core_hz')) and str(row.get('per_core_hz', '')).strip():
                per_core = float(row['per_core_hz'])
            else:
                per_core = (cpu_total / cores) if (cpu_total and cores > 0) else float(RSU_PER_CORE_HZ_DEFAULT)
            heap = [(0.0, i) for i in range(cores)]
            heapq.heapify(heap)
            self.rsu_spec[rid] = {
                'x': x, 'y': y,
                'cpu_hz': cores * per_core,
                'uplink_bps':   self.rsu_default_uplink_bps,
                'downlink_bps': self.rsu_default_downlink_bps,
                'range_m': self.rsu_default_range_m,
                'cores': cores, 'per_core_hz': per_core, 'heap': heap,
                'available_at': 0.0,
            }

    def _build_rsu_kdtree(self):
        """Build a KD-tree of all RSU positions (called once after _build_rsu_specs)."""
        if not self.rsu_spec:
            self._rsu_kdtree = None
            self._rsu_pts: np.ndarray = np.empty((0, 2), dtype=np.float32)
            self._rsu_ids_list: List[str] = []
            return
        self._rsu_ids_list = list(self.rsu_spec.keys())
        self._rsu_pts = np.array([[self.rsu_spec[rid]['x'], self.rsu_spec[rid]['y']]
                                   for rid in self._rsu_ids_list], dtype=np.float32)
        self._rsu_kdtree = cKDTree(self._rsu_pts)
        print(f"[VECEnvOffloadCSV] RSU spatial index: {len(self.rsu_spec)} RSUs, "
              f"range={self.rsu_default_range_m:.0f}m", flush=True)

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def _get_xy(self, vid: str):
        return self._pos_current.get(vid, (None, None))

    def _get_speed(self, vid: str) -> float:
        return self._spd_current.get(vid, 0.0)

    def _dist_veh(self, a: str, b: str) -> float:
        xa, ya = self._pos_current.get(a, (None, None))
        xb, yb = self._pos_current.get(b, (None, None))
        if xa is None or xb is None:
            return float('inf')
        return math.hypot(xa - xb, ya - yb)

    def _candidates_in_range(self, vid: str, max_range_m: float) -> List[str]:
        """Return vehicle IDs within max_range_m of vid, using the frame's KD-tree."""
        if self._kdtree is None:
            return []
        xy = self._pos_current.get(vid)
        if xy is None:
            return []
        idxs = self._kdtree.query_ball_point(xy, max_range_m)
        return [self._kdtree_vids[i] for i in idxs if self._kdtree_vids[i] != vid]

    def _nearest_peer(self, vid: str, live_ids, max_range_m: float):
        candidates = self._candidates_in_range(vid, max_range_m)
        if not candidates:
            return None
        xa, ya = self._pos_current[vid]
        best_id, best_d = None, float('inf')
        for pid in candidates:
            xb, yb = self._pos_current[pid]
            d = math.hypot(xa - xb, ya - yb)
            if d < best_d:
                best_id, best_d = pid, d
        return best_id

    def _best_peer(self, src_vid, live_ids, max_range_m,
                   now: float, cpu_cycles: float, data_bytes: int,
                   category: str):
        """KD-tree accelerated override of parent's _best_peer."""
        candidates = self._candidates_in_range(src_vid, max_range_m)
        if not candidates:
            return None
        P, kmax, eta = self._parallel_params(category, executor="veh")
        ul_s = data_bytes * 8.0 / max(1.0, self.v2v_rate_bps)
        best_peer, best_finish = None, float("inf")
        for vid in candidates:
            self._ensure_vehicle_spec(vid)  # required before _schedule_multi_on_vehicle
            arrival = now + ul_s
            _, _, finish_s, _, _, _ = self._schedule_multi_on_vehicle(
                vid, arrival, cpu_cycles, P, kmax, eta, self.thread_overhead_s, commit=False
            )
            if finish_s < best_finish:
                best_finish = finish_s
                best_peer = vid
        return best_peer

    def _nearest_rsu(self, vid: str):
        x, y = self._pos_current.get(vid, (None, None))
        if x is None:
            return None
        best, best_d = None, float('inf')
        for rid, spec in self.rsu_spec.items():
            d = math.hypot(x - spec['x'], y - spec['y'])
            if d < best_d:
                best, best_d = (rid, d), d
        return best

    # ── Frame helpers ─────────────────────────────────────────────────────────

    def _load_frame(self, idx: int):
        """Load vehicles for frame index `idx` into live state."""
        abs_idx = self._window_start_idx + idx
        s, e = int(self._frame_start[abs_idx]), int(self._frame_end[abs_idx])
        vids = self._col_vid[s:e]
        xs   = self._col_x[s:e]
        ys   = self._col_y[s:e]
        spds = self._col_spd[s:e]
        self._pos_current = {str(v): (float(x), float(y)) for v, x, y in zip(vids, xs, ys)}
        self._spd_current = {str(v): float(sp) for v, sp in zip(vids, spds)}
        self._live_ids    = list(self._pos_current.keys())
        self._pos_step_current = int(self._steps_unique[abs_idx])

        # Build spatial index for fast radius queries (avoids O(N) scan in peer search)
        if self._live_ids:
            _pts = np.array(list(self._pos_current.values()), dtype=np.float32)
            self._kdtree      = cKDTree(_pts)
            self._kdtree_vids = self._live_ids  # same order as _pos_current.values()
        else:
            self._kdtree      = None
            self._kdtree_vids = []

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        # Fragment mode: pick a random CSV file for this episode
        if self._frag_files is not None:
            idx = int(self._episode_rng.randint(0, len(self._frag_files)))
            chosen = str(self._frag_files[idx])
            if chosen != self._current_csv_path:
                self._csv_loaded = False
                self._current_csv_path = chosen
        else:
            self._current_csv_path = self.pos_csv

        self._load_csv_once()

        # Advance episode salt for fresh task draws
        self._episode_salt = getattr(self, '_episode_salt', 0) + 1

        # Sample RSUs for this episode (if rsu_sample_k > 0)
        if self.rsu_sample_k > 0 and len(self._rsu_ids_list) > 0:
            k = min(self.rsu_sample_k, len(self._rsu_ids_list))
            self._episode_rsu_indices = self._episode_rng.choice(
                len(self._rsu_ids_list), size=k, replace=False)
        else:
            self._episode_rsu_indices = None

        n_frames = len(self._steps_unique)
        ep_len   = int(self.max_steps)

        if n_frames < ep_len:
            raise ValueError(
                f"CSV only has {n_frames} frames but episode_length={ep_len}. "
                f"Use a longer CSV or shorter episode."
            )

        # Sample random window
        max_start = n_frames - ep_len
        self._window_start_idx = int(self._episode_rng.randint(0, max_start + 1))
        self._frame_idx        = 0

        # Reset episode state
        self.step_count = 0
        self._offload_rows.clear()
        self._step_rows.clear()
        self.veh_spec.clear()
        self._reset_compute_heaps(now=0.0)

        self._load_frame(0)
        self._touch_motion_cache(self._live_ids)

        return self._obs_stub, {'tasks_generated': [], 'offloads': []}

    def step(self, action):
        self._frame_idx       += 1
        self.step_count       += 1
        self.global_step_count += 1

        truncated  = (self._frame_idx >= self.max_steps)
        terminated = False

        if truncated:
            self._flush_logs()
            return self._obs_stub, 0.0, terminated, truncated, {'tasks_generated': [], 'offloads': []}

        self._load_frame(self._frame_idx)
        self._touch_motion_cache(self._live_ids)
        # _ensure_vehicle_spec is called lazily per-task inside _evaluate_candidates
        # and _decide_and_schedule — no need to pre-build specs for all 18k vehicles.

        generated = self._generate_tasks_seeded(self._live_ids)

        decisions = []
        if getattr(self, 'offload_policy', 'greedy') != 'agent':
            for rec in generated:
                drec = self._decide_and_schedule(rec, self._live_ids)
                decisions.append(drec)

        task_rewards = [self._compute_task_reward(d) for d in decisions]
        step_reward  = float(np.mean(task_rewards)) if task_rewards else 0.0

        if decisions:
            self.agent_finalize_tick(decisions)

        # Count V2V peers within range for each generated task
        n_v2v_peers = 0
        if generated and self._kdtree is not None:
            vid = generated[0]["vehicle_id"]
            pos = self._pos_current.get(vid)
            if pos is not None:
                n_v2v_peers = int(self._kdtree.query_ball_point(
                    pos, self.v2v_range_m, return_length=True)) - 1

        info = {
            'tasks_generated': generated,
            'offloads':        decisions,
            'step_reward':     step_reward,
            'sim_step':        int(self._pos_step_current),
            'env_step':        int(self.step_count),
            'n_v2v_peers':     n_v2v_peers,
        }
        obs = self._make_observation(generated)
        return obs, step_reward, terminated, truncated, info

    # ── Vectorized task generation (replaces per-vehicle RNG loop) ───────────

    def _generate_tasks_seeded(self, live_ids):
        """
        Vectorized override: one RNG per frame instead of one per vehicle.

        The parent creates a fresh SeedSequence+Generator for every vehicle
        (O(N) RNG constructions = ~9s/episode at N=18k).  Here we draw all N
        Poisson values in a single numpy call, then do the same for category,
        cpu, data and deadline using pre-built probability arrays.  Episode-
        and frame-level reproducibility is preserved via the combined seed.
        """
        lam      = float(self.lambda_per_veh_per_sec) * float(self.step_length_s)
        sim_step = int(getattr(self, "_pos_step_current", self.step_count))
        salt     = int(getattr(self, "_episode_salt", 0))
        base     = int(getattr(self, "task_seed_base", 12345))

        # One RNG per (episode, frame) — cheap, reproducible
        seed = (base ^ (salt * 1_000_003) ^ (sim_step * 999_983)) & 0xFFFFFFFF
        rng  = np.random.default_rng(seed)

        # Vehicle selection:
        # If rsu_sample_k > 0 (backward compat): restrict to vehicles within RSU range
        #   of the K RSUs sampled at episode reset.
        # Otherwise (default): use all live vehicles — fragments are already filtered
        #   to RSU proximity at extraction time, so no further range filter is needed.
        if self._episode_rsu_indices is not None and self._kdtree is not None:
            rsu_range = float(self.rsu_default_range_m)
            near_rsu: set = set()
            for rsu_idx in self._episode_rsu_indices:
                xy = self._rsu_pts[rsu_idx]
                near_rsu.update(self._kdtree.query_ball_point(xy, rsu_range))
            vids = [self._kdtree_vids[i] for i in near_rsu]
        else:
            vids = list(live_ids)

        N = len(vids)
        if N == 0:
            return []

        # Vectorised Poisson draw — single C-level call instead of N Python calls
        ks = rng.poisson(lam, size=N)
        active = np.flatnonzero(ks)
        if active.size == 0:
            return []

        # Pre-build category distribution once
        cats  = list(self.category_profiles.keys())
        probs = np.array([self.category_mix[c] for c in cats], dtype=float)
        probs /= probs.sum()

        out = []
        for idx in active:
            vid = vids[idx]
            k   = int(ks[idx])
            for i in range(k):
                cat  = cats[int(rng.choice(len(cats), p=probs))]
                prof = self.category_profiles[cat]
                cpu  = float(np.clip(rng.normal(prof["cpu_cycles"][0], prof["cpu_cycles"][1]),
                                     1.0, 1e13))
                dat  = float(np.clip(rng.normal(prof["data_bytes"][0],  prof["data_bytes"][1]),
                                     1.0, int(getattr(self, "max_ul_bytes", 32 * 1024 * 1024))))
                ddl  = float(max(rng.normal(prof["deadline_s"][0], prof["deadline_s"][1]), 1e-6))
                tid  = f"{sim_step}:{vid}:{i+1}"
                rec  = {"step": int(self.global_step_count), "tid": tid,
                        "vehicle_id": vid, "category": cat,
                        "cpu_cycles": cpu, "data_bytes": dat,
                        "deadline_s": ddl, "arrival_s": float(sim_step)}
                self._task_rows.append(rec)
                self._write_task_to_csv(rec)
                out.append(rec)
        return out

    # ── Override SUMO lifecycle (no-ops) ──────────────────────────────────────

    def _start_sumo(self):
        pass   # no SUMO

    def _close_sumo(self):
        pass   # no SUMO

    def close(self):
        try:
            self._flush_logs()
        except Exception:
            pass

    # ── Reset heaps (episode boundary only) ──────────────────────────────────

    def _reset_compute_heaps(self, now: float = 0.0):
        for rid, spec in self.rsu_spec.items():
            cores = int(spec.get('cores', 1))
            spec['heap'] = [(now, i) for i in range(cores)]
            heapq.heapify(spec['heap'])
            spec['available_at'] = float(now)
        for v in self.veh_spec.values():
            v['available_at'] = float(now)
            if 'cores' in v and 'per_core_hz' in v:
                cores = int(v['cores'])
                v['heap'] = [(now, i) for i in range(cores)]
                heapq.heapify(v['heap'])
