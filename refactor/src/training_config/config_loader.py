"""Simple configuration objects"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

RSU_PROFILES = {
    1: {
        "cores": 48,
        "per_core_hz": 3.5e9,    # Total: 168 GHz (unrealistic data center)
    },
    2: {
        "cores": 16,
        "per_core_hz": 2.5e9,    # Total: 40 GHz (realistic edge server)
    },
    3: {
        "cores": 8,
        "per_core_hz": 2.0e9,    # Total: 16 GHz (realistic roadside unit, e.g. embedded server)
    },
}

def get_rsu_profile(version: int = 1) -> Dict:
    """Get RSU compute profile for specified version"""
    if version not in RSU_PROFILES:
        raise ValueError(f"Unknown rsu_profile_version: {version}. Valid: {list(RSU_PROFILES.keys())}")
    return RSU_PROFILES[version]


TASK_PROFILES = {
    1: {
        "safety": {
            "cpu_cycles": (3.0e8, 8.0e7),    # 300M ± 80M
            "data_bytes": (5.0e5, 1.5e5),    # 500 KB ± 150KB
            "deadline_s": (0.05, 0.015)      # 50ms ± 15ms
        },
        "autonomous": {
            "cpu_cycles": (1.5e9, 4.0e8),    # 1500M ± 400M
            "data_bytes": (1.2e6, 3.0e5),    # 1.2 MB ± 300KB
            "deadline_s": (0.12, 0.03)       # 120ms ± 30ms
        },
        "infotainment": {
            "cpu_cycles": (5.0e9, 1.5e9),    # 5000M ± 1500M
            "data_bytes": (5.0e6, 2.0e6),    # 5 MB ± 2MB
            "deadline_s": (0.50, 0.15)       # 500ms ± 150ms
        }
    },
    3: {
        # Based on VEC literature (Zhou et al. 2020, Liu et al. 2021)
        "safety": {
            "cpu_cycles": (1.2e8, 3.0e7),    # 120M ± 30M (collision detection)
            "data_bytes": (3.0e5, 1.0e5),    # 300 KB (preprocessed sensor)
            "deadline_s": (0.05, 0.015)      # 50ms (safety-critical)
        },
        "autonomous": {
            "cpu_cycles": (4.0e8, 1.0e8),    # 400M ± 100M (path planning)
            "data_bytes": (8.0e5, 2.0e5),    # 800 KB (feature maps)
            "deadline_s": (0.12, 0.03)       # 120ms (control loop)
        },
        "infotainment": {
            "cpu_cycles": (1.2e9, 4.0e8),    # 1200M ± 400M (video/AR)
            "data_bytes": (3.0e6, 1.0e6),    # 3 MB (compressed content)
            "deadline_s": (0.50, 0.15)       # 500ms (user tolerance)
        }
    },
    4: {
        # v3 compute/data unchanged; deadlines raised above the NR-V2X edge latency floor
        # (~225ms V2V, ~255ms local, ~310ms RSU) so SLA satisfaction is achievable and
        # discriminative between routing decisions.
        # Safety/autonomous: consistent with 3GPP TS 22.186 cooperative edge services
        # (100–500ms range).  Infotainment: 5G MEC content delivery target.
        "safety": {
            "cpu_cycles": (1.2e8, 3.0e7),    # 120M ± 30M (collision detection)
            "data_bytes": (3.0e5, 1.0e5),    # 300 KB (preprocessed sensor)
            "deadline_s": (0.30, 0.05)       # 300ms ± 50ms — V2V (~225ms) typically meets it;
                                              # RSU (~310ms) often misses; local (~255ms) borderline
        },
        "autonomous": {
            "cpu_cycles": (4.0e8, 1.0e8),    # 400M ± 100M (path planning)
            "data_bytes": (8.0e5, 2.0e5),    # 800 KB (feature maps)
            "deadline_s": (0.55, 0.09)       # 550ms ± 90ms — met by all paths under normal load;
                                              # heavy queuing events (p95 tail) cause misses
        },
        "infotainment": {
            "cpu_cycles": (1.2e9, 4.0e8),    # 1200M ± 400M (video/AR)
            "data_bytes": (3.0e6, 1.0e6),    # 3 MB (compressed content)
            "deadline_s": (2.00, 0.40)       # 2000ms ± 400ms — rarely missed
        }
    },
    5: {
        # v5: Heavy compute — local CPU (2 GHz mean) frequently misses deadlines.
        # RSU (40 GHz = 20× faster) meets deadlines comfortably, creating a clear
        # routing incentive. V2V peers (also 2 GHz) struggle similarly to local.
        # Safety:       600M cycles → 300ms at 2 GHz (borderline; misses at <2 GHz)
        #               RSU: 600M/40G = 15ms compute + tx ≈ 60ms total → well under 300ms
        # Autonomous:   1200M cycles → 600ms at 2 GHz (> 550ms deadline → most miss locally)
        #               RSU: 1200M/40G = 30ms + tx ≈ 85ms → well under 550ms
        # Infotainment: 3600M cycles → 1800ms at 2 GHz (borderline for 2000ms deadline)
        #               RSU: 3600M/40G = 90ms + tx ≈ 145ms → well under 2000ms
        "safety": {
            "cpu_cycles": (6.0e8, 1.5e8),    # 600M ± 150M (5× v4; borderline local)
            "data_bytes": (3.0e5, 1.0e5),    # 300 KB (same as v4)
            "deadline_s": (0.30, 0.05)       # 300ms ± 50ms (same as v4)
        },
        "autonomous": {
            "cpu_cycles": (1.2e9, 3.0e8),    # 1200M ± 300M (3× v4; most miss locally)
            "data_bytes": (8.0e5, 2.0e5),    # 800 KB (same as v4)
            "deadline_s": (0.55, 0.09)       # 550ms ± 90ms (same as v4)
        },
        "infotainment": {
            "cpu_cycles": (3.6e9, 1.2e9),    # 3600M ± 1200M (3× v4; borderline local)
            "data_bytes": (3.0e6, 1.0e6),    # 3 MB (same as v4)
            "deadline_s": (2.00, 0.40)       # 2000ms ± 400ms (same as v4)
        }
    }
}

def get_task_profiles(version: int = 1) -> Dict:
    """Get task profiles for specified version"""
    if version not in TASK_PROFILES:
        raise ValueError(f"Unknown task_profile_version: {version}. Valid: {list(TASK_PROFILES.keys())}")
    return TASK_PROFILES[version]


@dataclass
class NetworkArchitecture:
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = 'tanh'


@dataclass
class TrainingConfig:
    verbose: int = 1
    checkpoint_freq: int = 10000


@dataclass
class PPOHyperparameters:
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 64
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    network_architecture: NetworkArchitecture = field(default_factory=NetworkArchitecture)
    training: TrainingConfig = field(default_factory=TrainingConfig)


@dataclass
class DQNHyperparameters:
    learning_rate: float = 1e-4
    buffer_size: int = 100000
    learning_starts: int = 1000
    batch_size: int = 32
    tau: float = 1.0
    gamma: float = 0.99
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 1000
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    network_architecture: NetworkArchitecture = field(default_factory=lambda: NetworkArchitecture(activation='relu'))
    training: TrainingConfig = field(default_factory=TrainingConfig)


@dataclass
class ScenarioConfig:
    name: str = 'default'
    seed: int = 42
    date: Optional[str] = None  # Format: "YYYY-MM-DD"
    timeframe: Optional[Tuple[int, int]] = None  # Format: (start_hour, end_hour) e.g., (7, 9)
    episode_length_steps: int = 100  # SUMO steps per episode (100 steps ≈ 5700 agent decisions)
    state_space_version: int = 1  # 1=v1 (oracular, 34 features), 2=v2 (realistic, 22 features)
    task_profile_version: int = 1  # 1=original heavy, 3=realistic edge compute
    rsu_profile_version: int = 1  # 1=overpowered (168 GHz), 2=realistic (40 GHz)
    v2v_range_m: Optional[float] = None  # None = use global default (200m); 0.0 = disable V2V
    lambda_per_veh_per_sec: Optional[float] = None  # None = use global config default (0.3)
    reward_type: Optional[str] = None  # None = current energy+deadline; 'deadline_only'; 'latency_only'
    peer_selection: str = "best"  # "best" (min finish time) or "nearest" (geographic)
    veh_per_core_hz_mean: Optional[float] = None  # None → config.py default (1.8 GHz)
    veh_per_core_hz_std:  Optional[float] = None  # None → config.py default (0.4 GHz)
    veh_per_core_hz_min:  Optional[float] = None  # None → config.py default (1.2 GHz)
    rsu_range_m: Optional[float] = None           # None → config.py default (250 m)
    demand_scale: Optional[float] = None          # None → SUMO default (1.0); e.g. 0.5 = half density
    rsu_sample_k: int = 0                         # 0 = use all RSUs; N = sample N RSUs per episode (CSV only)
    fragment_hour_range: Optional[tuple] = None   # None = all fragments; (h0, h1) = daytime filter (CSV only)


@dataclass
class PolicyConfig:
    algorithm: str = 'ppo'
    hyperparameters: object = field(default_factory=PPOHyperparameters)


def create_policy_config(algorithm: str, **overrides):
    """Create policy config for algorithm"""
    if algorithm == 'ppo':
        hyperparams = PPOHyperparameters(**overrides)
    elif algorithm == 'dqn':
        hyperparams = DQNHyperparameters(**overrides)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return PolicyConfig(algorithm=algorithm, hyperparameters=hyperparams)


def create_scenario_config(
    name='default',
    seed=42,
    date: Optional[str] = None,
    timeframe: Optional[Tuple[int, int]] = None,
    episode_length_steps: int = 100,
    state_space_version: int = 1,
    task_profile_version: int = 1,
    rsu_profile_version: int = 1,
    v2v_range_m: Optional[float] = None,
    lambda_per_veh_per_sec: Optional[float] = None,
    reward_type: Optional[str] = None,
    peer_selection: str = "best",
    veh_per_core_hz_mean: Optional[float] = None,
    veh_per_core_hz_std:  Optional[float] = None,
    veh_per_core_hz_min:  Optional[float] = None,
    rsu_range_m: Optional[float] = None,
    demand_scale: Optional[float] = None,
    rsu_sample_k: int = 0,
    fragment_hour_range: Optional[tuple] = None,
):
    """
    Create scenario config

    Args:
        name: Scenario name
        seed: Random seed
        date: Date string "YYYY-MM-DD" (optional)
        timeframe: (start_hour, end_hour) tuple, e.g., (7, 9) (optional)
        episode_length_steps: Length of each episode in SUMO steps (default: 100)
        state_space_version: State space version (default: 1)
            - 1: Original with oracular features (34 features)
            - 2: Realistic/measurable only (22 features)
        task_profile_version: Task profile version (default: 1)
            - 1: Original heavy compute (impossible locally)
            - 3: Realistic edge compute (meaningful local vs offload)
        rsu_profile_version: RSU compute profile (default: 1)
            - 1: Overpowered (48 cores @ 3.5 GHz = 168 GHz)
            - 2: Realistic edge server (16 cores @ 2.5 GHz = 40 GHz)
            - 3: Lightweight roadside unit (8 cores @ 2.0 GHz = 16 GHz)
    """
    return ScenarioConfig(
        name=name,
        seed=seed,
        date=date,
        timeframe=timeframe,
        episode_length_steps=episode_length_steps,
        state_space_version=state_space_version,
        task_profile_version=task_profile_version,
        rsu_profile_version=rsu_profile_version,
        v2v_range_m=v2v_range_m,
        lambda_per_veh_per_sec=lambda_per_veh_per_sec,
        reward_type=reward_type,
        peer_selection=peer_selection,
        veh_per_core_hz_mean=veh_per_core_hz_mean,
        veh_per_core_hz_std=veh_per_core_hz_std,
        veh_per_core_hz_min=veh_per_core_hz_min,
        rsu_range_m=rsu_range_m,
        demand_scale=demand_scale,
        rsu_sample_k=rsu_sample_k,
        fragment_hour_range=fragment_hour_range,
    )
