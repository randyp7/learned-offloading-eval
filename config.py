import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
#SUMO_CONFIG = PROJECT_ROOT / "sumo" / "berlin_2023-10-18.sumocfg"
SUMO_CONFIG = PROJECT_ROOT / "sumo" / "berlin_a100_00-01.sumocfg"
#SUMO_CONFIG = PROJECT_ROOT / "sumo" / "berlin_2025-01-08.sumocfg"
#SUMO_CONFIG = PROJECT_ROOT / "sumo"/"5_final_output"/"2025"/"2025-01-08"/"berlin_2025-01-08.sumocfg"

# SUMO_HOME must point at the installed SUMO package root (…/site-packages/sumo/).
# Resolved from the environment so the repo is portable across accounts/machines:
#   1. $SUMO_HOME if already exported (recommended — set it in your SLURM script), or
#   2. $SUMO_HOME_DIR as an explicit override, else empty (env_factory raises a
#      clear error rather than silently using a stranger's home directory).
# On CSF, activate the SUMO venv (e.g. ~/sumo1.27.0) which exports SUMO_HOME for you.
SUMO_HOME = os.environ.get("SUMO_HOME", os.environ.get("SUMO_HOME_DIR", ""))
SUMO_HOME_NT = os.environ.get("SUMO_HOME", "")

# 8 RSUs along the A100 corridor (~610 m spacing) — the file the study actually
# uses. The old path (rsu_placement_berlin_urban.csv) was never committed.
RSU_CSV = PROJECT_ROOT / "data" / "rsu_data.csv"
LOG_DIR = PROJECT_ROOT / "logs_live_sumo"

# ---- Simulation controls ----
STEP_LENGTH_S = 1.0
MAX_STEPS     = 3_600
SEED          = 42

# ---- Observations ----
OBS_MAX_VEH = 50
OBS_PER_VEH = 3   # [x, y, speed]

# ---- Task generation (REALISTIC/MODERATE SCENARIO) ----
# Based on actual VEC literature: moderate task load where RSU can become a bottleneck.
# 0.3 tasks/s = ~1 task per 3s per vehicle (path planning + sensor fusion + infotainment)
# At peak with 30 vehicles + RSU profile v3 (16 GHz): ~5.4 G cycles/s = 34% utilisation;
# combined with RSU coverage gaps (range 250m, spacing 610m) this makes V2V meaningful.
LAMBDA_PER_VEH_PER_SEC = 0.3  # tasks/sec per vehicle
CATEGORY_MIX = {"autonomous": 0.35, "safety": 0.30, "infotainment": 0.35}

# ORIGINAL BASELINE (commented for reference):
# CATEGORY_PROFILES = {
#     "autonomous":   {"cpu_cycles": (2.5e9, 0.5e9), "data_bytes": (1.5e6, 5e5), "deadline_s": (0.10, 0.04)},
#     "safety":       {"cpu_cycles": (1.6e9, 3.5e8), "data_bytes": (8e5,  2.5e5), "deadline_s": (0.06, 0.03)},
#     "infotainment": {"cpu_cycles": (9e8,  2.5e8),  "data_bytes": (6e6,  2e6),   "deadline_s": (0.50, 0.25)},
# }

# REALISTIC SCENARIO: Based on VEC literature (Zhou et al., Chen et al.)
# - Safety: Collision avoidance, emergency braking (tight deadlines, light compute)
# - Autonomous: Object detection, path planning (moderate deadlines, heavy compute)
# - Infotainment: Video streaming, AR/VR (relaxed deadlines, very heavy compute)
CATEGORY_PROFILES = {
  "safety": {
    "cpu_cycles": (3.0e8, 8.0e7),    # 300 Mcycles ± 80M (collision detection)
    "data_bytes": (5.0e5, 1.5e5),    # 500 KB ± 150KB (sensor data)
    "deadline_s": (0.05, 0.015)      # 50ms ± 15ms (strict safety requirement)
  },
  "autonomous": {
    "cpu_cycles": (1.5e9, 4.0e8),    # 1500 Mcycles ± 400M (object detection/tracking)
    "data_bytes": (1.2e6, 3.0e5),    # 1.2 MB ± 300KB (camera/lidar frames)
    "deadline_s": (0.12, 0.03)       # 120ms ± 30ms (control loop requirement)
  },
  "infotainment": {
    "cpu_cycles": (5.0e9, 1.5e9),    # 5000 Mcycles ± 1500M (video processing, AR rendering)
    "data_bytes": (5.0e6, 2.0e6),    # 5 MB ± 2MB (multimedia content)
    "deadline_s": (0.50, 0.15)       # 500ms ± 150ms (user experience tolerance)
  }
}

# UTILIZATION CALCULATION (for average vehicle, task profile v1):
# Weighted avg: 0.30*300M + 0.35*1500M + 0.35*5000M = 2365 Mcycles/task
# Average vehicle: 3 cores * 1.8 GHz = 5.4 GHz total
# Vehicle utilization = (0.3 tasks/sec * 2365M cycles/task) / 5.4G Hz = 13% -> Light
# RSU utilization (16 GHz, 20 vehicles): 20 * 0.3 * 2365M / 16G Hz = 88% -> Heavy!
# RSU utilization (16 GHz, 10 vehicles): 10 * 0.3 * 2365M / 16G Hz = 44% -> Moderate
# This creates a realistic scenario where:
#   - RSU saturates during peak traffic (queue delays make V2V attractive)
#   - Coverage gaps (250m range, 610m spacing) force V2V or local in dead zones
#   - V2V provides meaningful benefit over RSU-only in both gap and saturation regimes

# ---- Offloading model (MODERATE SCENARIO: Average vehicles) ----
# Per-vehicle CPU capacity (total, not per-core). Assigned when a vehicle first appears.
# ===== LEGACY PARAMETERS (not used with multi-core, kept for compatibility) =====
VEH_CPU_HZ_MEAN = 2.0e9   # Moderate mean (up from 1.2e9)
VEH_CPU_HZ_STD  = 1.0e9   # Moderate variance
VEH_CPU_HZ_MIN  = 1.0e9   # Minimum (up from 0.5e9)

# RSU defaults (can be overridden per-RSU via columns in rsu_data.csv)
# Range: 250m matches realistic NR-V2X urban RSU coverage at 0.4W vehicle TX.
# With 8 RSUs spaced ~610m apart, this leaves ~110m gaps between RSUs where
# V2V is the only offloading alternative to local compute.
RSU_DEFAULT_CPU_HZ     = 16e9      # total RSU compute (profile v3 default)
RSU_DEFAULT_RANGE_M    = 250       # 250m realistic urban RSU coverage

# V2V link (simplified constant rate if within range)
V2V_RANGE_M  = 200 #80.0
V2V_RATE_MBPS = 50.0

# Energy model (very lightweight)
ENERGY_PER_CYCLE_J_VEH = 1.0e-9    # local compute (sender’s device)
ENERGY_PER_CYCLE_J_RSU = 1.5e-10   # RSU compute (system perspective)

# Greedy or random (keep greedy for baseline)
OFFLOAD_POLICY = "greedy"          # "greedy" | "random"

# ---- Radio & response path ----
# Result payload size (~ fraction of request payload). You can tune per category later.
RESULT_SIZE_RATIO = 0.05  # result_bytes = ratio * data_bytes default 0.20

# V2V rates (can be asymmetric; set equal if you want symmetry)
V2V_RATE_MBPS_UL = 50.0
V2V_RATE_MBPS_DL = 50.0

# RSU link (uplink/downlink can differ)
RSU_DEFAULT_UPLINK_MBPS   = 30.0
RSU_DEFAULT_DOWNLINK_MBPS = 60.0

# Radio powers (Watts): TX and RX for vehicles and RSUs
P_TX_V2V_W = 1.0        # requester TX to peer (uplink)
P_RX_VEH_W = 0.8        # vehicle RX power (when receiving)
P_TX_RSU_W = 1.2        # requester TX to RSU (uplink) and RSU TX to requester (downlink) # ~30 dBm
P_RX_RSU_W = 1.0        # RSU RX power (when receiving)

# --- Radio profile (802.11p / ITS-G5 defaults; tweak anytime) ---
PHY_NAME = "11p-10MHz"
CARRIER_HZ = 5.9e9
BANDWIDTH_HZ = 10e6

# TX/RX powers (Watts) and noise figures (dB)
P_TX_VEH_W = 0.2     # ~23 dBm
NOISE_FIG_VEH_DB = 7.0
NOISE_FIG_RSU_DB = 4.0 #5.0

# Path loss + shadowing (log-distance)
PL_EXPONENT = 2.2     # LOS-ish road; urban use 2.6–3.0
SHADOW_SIGMA_DB = 4.0 # log-normal std dev
FAST_FADING = False   # if True, enable Nakagami-m below
NAKAGAMI_M = 1.5

# CSMA/CA (11p) MAC timing
SLOT_TIME_S = 13e-6
SIFS_S = 32e-6
AIFSN = 3
CW_MIN = 15
ACK_BYTES = 14       # ACK payload for unicast
PHY_PREAMBLE_S = 40e-6
PHY_HEADER_S = 8e-6

# Channel load proxy (0..1). 0 = empty, 0.5 = moderate, 0.8 = busy
CHANNEL_LOAD = 0.3

# MCS table (11p 10MHz; rates are indicative—adjust to match your study)
MCS_TABLE = [
    # (snr_threshold_db, spectral_eff_bps_per_hz)
    (  3.0, 0.3),   # ~BPSK 1/2  -> ~3 Mbps @10MHz
    (  6.0, 0.45),  # ~BPSK 3/4  -> ~4.5 Mbps
    (  9.0, 0.6),   # ~QPSK 1/2  -> ~6 Mbps
    ( 12.0, 0.9),   # ~QPSK 3/4  -> ~9 Mbps
    ( 15.0, 1.2),   # ~16-QAM 1/2-> ~12 Mbps
    ( 18.0, 1.8),   # ~16-QAM 3/4-> ~18 Mbps
    ( 21.0, 2.4),   # ~64-QAM 2/3-> ~24 Mbps
    ( 24.0, 2.7),   # ~64-QAM 3/4-> ~27 Mbps
]

# PER curve steepness per MCS (logistic). Larger -> sharper waterfall.
PER_ALPHA = 1.2
MAX_RETRIES = 4

# ===== NR-V2X (scheduled MAC) =====
# Carrier & numerology (PC5 sidelink around 5.9 GHz; tweak as needed)
NR_CARRIER_HZ = 5_900_000_000
NR_SCS_KHZ = 30                     # 15/30/60 are common; 30 kHz is a solid default
NR_SLOT_SYMBOLS = 14                # normal CP
NR_SLOT_DURATION_S = 0.5e-3         # 0.5 ms for 30 kHz SCS
NR_BANDWIDTH_PRB_VEH = 56               # PRBs allocated for sidelink (e.g., 20 MHz @30kHz ≈ 51 PRB)
NR_BANDWIDTH_PRB_RSU = 218  

# Physical overheads
NR_OVERHEAD_DATA = 0.85             # PSSCH/PSCCH/DMRS overhead factor (effective fraction for payload)
NR_HARQ_REPETITIONS = 1             # set >1 to model repetitions for reliability

# Power & receiver (Watts, dB)
NR_P_TX_VEH_W = 0.4                 # ~26 dBm UE sidelink TX
NR_P_RX_VEH_W = 0.8
NR_P_TX_RSU_W = 2.0                 # RSU/gNB-class sidelink TX
NR_P_RX_RSU_W = 2.0
NR_NOISE_FIG_VEH_DB = 7.0
NR_NOISE_FIG_RSU_DB = 4 #5.0

# 38.901-like channel (large-scale) – scenario & stats
NR_SCENARIO = "UMi"                 # "UMi" (urban micro/streets), "UMa", or "RMa"
NR_SHADOW_SIGMA_DB_LOS = 3.0
NR_SHADOW_SIGMA_DB_NLOS = 4.0

# LOS probability model selector (per 38.901)
NR_LOS_MODEL = "UMi"                # "UMi" or "UMa" (adds distance-based LOS probability)

# CQI mapping (SNR->CQI thresholds, dB). Not standardized; default is a common sim set.
NR_CQI_THRESH_DB = [-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7, 14.1, 16.3, 18.7, 21.0, 22.7]


# ==== Vehicle CPU class distribution ====
# Based on automotive SoC market survey (2020-2023 era).
# Each class: (market_share, cores, per_core_hz_min, per_core_hz_max)
# Frequency sampled uniformly within range; cores are fixed per class.
VEH_CPU_CLASSES = [
    # (market_share, cores, hz_min,  hz_max,  label)
    (0.30, 2, 0.8e9, 1.5e9, "budget"),     # ARM Cortex-A53, NXP i.MX6
    (0.50, 4, 1.5e9, 2.5e9, "mid_range"),  # Qualcomm 820A, NXP i.MX8, Renesas R-Car M3
    (0.15, 6, 2.0e9, 3.0e9, "premium"),    # Qualcomm 8155, Renesas R-Car H3
    (0.05, 8, 2.5e9, 5.0e9, "av_ready"),   # Tesla FSD, Qualcomm 8295, NVIDIA Jetson
]
VEH_CPU_CLASS_WEIGHTS = [c[0] for c in VEH_CPU_CLASSES]

# RSUs (can be overridden per RSU via rsu_data.csv columns: cores, per_core_hz)
# Profile v3 defaults: 8 cores @ 2.0 GHz = 16 GHz (realistic roadside embedded server)
RSU_CORES_DEFAULT      = 8
RSU_PER_CORE_HZ_DEFAULT = 2.0e9

# ===== Task-level parallelism (Amdahl-style) =====
# Per category: P (parallel fraction), per-executor k_max limits.
# PARALLEL = {
#     "safety":       {"P": 0.30, "kmax_veh": 2,  "kmax_rsu": 4},
#     "autonomous":   {"P": 0.60, "kmax_veh": 4,  "kmax_rsu": 8},
#     "infotainment": {"P": 0.80, "kmax_veh": 6,  "kmax_rsu": 12},
# }

PARALLEL = {
    "safety":       {"P": 0.60, "kmax_veh": 4,  "kmax_rsu": 8},
    "autonomous":   {"P": 0.60, "kmax_veh": 4,  "kmax_rsu": 8},
    "infotainment": {"P": 0.80, "kmax_veh": 6,  "kmax_rsu": 12},
}

# Parallel efficiency (η) and per-extra-thread overhead (δ seconds)
PARALLEL_ETA_VEH = 0.80
PARALLEL_ETA_RSU = 0.90
THREAD_OVERHEAD_S = 0.0002  # 0.2 ms per extra thread

REWARD_CFG = {
    "type": "weighted",        # (future: "lagrangian", "lexicographic")
    "weights": {
        "sla": 3.0,            # Increased positive reward for meeting deadlines
        "delay": 0.3,          # Reduced delay penalty
        "energy_user": 0.15,   # Reduced energy penalties
        "energy_system": 0.05
    },
    "miss_penalty": 0.5,       # Reduced penalty for missed deadlines
    "category_weight": {       # safety-first
        "safety": 2.0,
        "autonomous": 1.0,
        "infotainment": 0.5
    },
    "aggregate": "mean",       # "mean" | "sum" | "median"
    "clip_min": -3.0,          # Less harsh clipping for PPO
    "clip_max":  5.0,          # Higher positive rewards allowed
    "eps": 1e-9,
    
    # PPO-specific reward shaping (only active when reward_type is None/default)
    # Overridden to False automatically for deadline_only and latency_only reward types.
    "ppo_mode": False,         # Flag for PPO-specific modifications
    "base_reward": 2.0,        # Base positive reward for any valid decision
    "efficiency_bonus": 1.0    # Bonus for efficient decisions (energy < local)
}

