import os
import sys
import time
from pathlib import Path

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.vec_env_offload_nrv2x_m_v01_seeded import VECEnvOffload as CoreEnv
from envs.vec_offload_env_csv_v01_seeded import VECEnvOffloadCSV as CSVEnv
from envs.offload_task_env import OffloadDecisionEnv, build_features, build_features_v2, build_features_v3, build_features_v4, build_features_v5, build_features_v6
from config import *

# Import task profiles and RSU profiles from config_loader
REFACTOR_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(REFACTOR_SRC))
from training_config.config_loader import get_task_profiles, TASK_PROFILES, get_rsu_profile, RSU_PROFILES

DEFAULT_TASK_PROFILE_VERSION = 1  # Default to v1 for backwards compatibility
DEFAULT_RSU_PROFILE_VERSION = 1   # Default to v1 for backwards compatibility

# Feature function registry
FEATURE_FUNCTIONS = {
    1: build_features,      # v1: Original with oracular features (34 features)
    2: build_features_v2,   # v2: Realistic/measurable only (22 features)
    3: build_features_v3,   # v3: v2 + traffic context (24 features)
    4: build_features_v4,   # v4: v3 + estimated processing time per candidate (27 features)
    5: build_features_v5,   # v5: v4 + V2V peer and RSU queue wait times (29 features)
    6: build_features_v6,   # v6: v5 + local CPU queue wait time (30 features)
}

DEFAULT_STATE_SPACE_VERSION = 1  # Default to v1 for backwards compatibility
from data.berlin_data_resolver import resolve_berlin_data


def ensure_sumo_env():
    """Set up SUMO environment variables.

    Prefers an already-exported SUMO_HOME (set by activating the SUMO venv or in
    the SLURM script). Falls back to config.SUMO_HOME only if it is non-empty.
    Raises a clear error rather than pointing SUMO at an empty/invalid path.
    """
    if "SUMO_HOME" not in os.environ or not os.environ["SUMO_HOME"]:
        default = SUMO_HOME if os.name != "nt" else SUMO_HOME_NT
        if not default:
            raise EnvironmentError(
                "SUMO_HOME is not set and no fallback is configured. "
                "Export SUMO_HOME to your SUMO package root "
                "(e.g. `. ~/sumo1.27.0/bin/activate`, which sets it), or set "
                "SUMO_HOME_DIR before running. See slurm/ scripts."
            )
        os.environ["SUMO_HOME"] = default
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)


def create_offloading_env(scenario_config, use_gui=False, state_space_version=None, sumocfg_override=None, pos_csv=None):
    """
    Create vectorized offloading environment

    Episodes will be episode_length_steps long. Each episode starts at the beginning
    of the specified timeframe and runs for episode_length_steps.

    Args:
        scenario_config: Configuration object with scenario parameters
        use_gui: Whether to show SUMO GUI
        state_space_version: State space version (1 or 2). If None, uses scenario_config
                            or DEFAULT_STATE_SPACE_VERSION.
            - v1: Original with oracular features (34 features)
            - v2: Realistic/measurable only (22 features)
        sumocfg_override: Optional path to a specific .sumocfg file, bypassing date resolver.

    Returns:
        DummyVecEnv wrapping the offloading environment
    """
    # Determine state space version
    if state_space_version is None:
        state_space_version = getattr(scenario_config, 'state_space_version', DEFAULT_STATE_SPACE_VERSION)

    feature_fn = FEATURE_FUNCTIONS.get(state_space_version)
    if feature_fn is None:
        raise ValueError(f"Unknown state_space_version: {state_space_version}. Valid: {list(FEATURE_FUNCTIONS.keys())}")

    # Determine task profile version
    task_profile_version = getattr(scenario_config, 'task_profile_version', DEFAULT_TASK_PROFILE_VERSION)
    task_profiles = get_task_profiles(task_profile_version)

    # Determine RSU profile version and calculate total compute
    rsu_profile_version = getattr(scenario_config, 'rsu_profile_version', DEFAULT_RSU_PROFILE_VERSION)
    rsu_profile = get_rsu_profile(rsu_profile_version)
    rsu_total_hz = rsu_profile['cores'] * rsu_profile['per_core_hz']

    # Shared constructor kwargs for both SUMO and CSV core envs
    def _core_kwargs(log_dir):
        return dict(
            rsu_csv=RSU_CSV,
            log_dir=log_dir,
            obs_max_veh=OBS_MAX_VEH,
            obs_per_veh=OBS_PER_VEH,
            max_steps=scenario_config.episode_length_steps,
            step_length_s=STEP_LENGTH_S,
            seed=scenario_config.seed,
            gui=use_gui,
            lambda_per_veh_per_sec=scenario_config.lambda_per_veh_per_sec if scenario_config.lambda_per_veh_per_sec is not None else LAMBDA_PER_VEH_PER_SEC,
            category_mix=CATEGORY_MIX,
            category_profiles=task_profiles,
            veh_cpu_hz_mean=VEH_CPU_HZ_MEAN,
            veh_cpu_hz_std=VEH_CPU_HZ_STD,
            veh_cpu_hz_min=VEH_CPU_HZ_MIN,
            v2v_range_m=scenario_config.v2v_range_m if scenario_config.v2v_range_m is not None else V2V_RANGE_M,
            v2v_rate_mbps=V2V_RATE_MBPS,
            rsu_default_cpu_hz=rsu_total_hz,
            rsu_default_cores=rsu_profile['cores'],
            rsu_default_per_core_hz=rsu_profile['per_core_hz'],
            rsu_default_range_m=getattr(scenario_config, 'rsu_range_m', None) or RSU_DEFAULT_RANGE_M,
            energy_per_cycle_j_veh=ENERGY_PER_CYCLE_J_VEH,
            energy_per_cycle_j_rsu=ENERGY_PER_CYCLE_J_RSU,
            p_tx_v2v_w=P_TX_V2V_W,
            p_tx_rsu_w=P_TX_RSU_W,
            v2v_rate_mbps_ul=V2V_RATE_MBPS_UL,
            v2v_rate_mbps_dl=V2V_RATE_MBPS_DL,
            rsu_default_uplink_mbps=RSU_DEFAULT_UPLINK_MBPS,
            rsu_default_downlink_mbps=RSU_DEFAULT_DOWNLINK_MBPS,
            result_size_ratio=RESULT_SIZE_RATIO,
            p_rx_veh_w=P_RX_VEH_W,
            p_rx_rsu_w=P_RX_RSU_W,
            offload_policy="agent",
            reward_type=getattr(scenario_config, 'reward_type', None),
            peer_selection=getattr(scenario_config, 'peer_selection', 'best'),
            veh_per_core_hz_mean=getattr(scenario_config, 'veh_per_core_hz_mean', None),
            veh_per_core_hz_std=getattr(scenario_config, 'veh_per_core_hz_std', None),
            veh_per_core_hz_min=getattr(scenario_config, 'veh_per_core_hz_min', None),
            task_seed_base=12345,
            task_rng_mode="stateless",
        )

    def _make_env():
        unique_log_dir = Path(LOG_DIR) / f"task_{os.getpid()}_{int(time.time()*1e6)%1_000_000}"
        unique_log_dir.mkdir(parents=True, exist_ok=True)

        if pos_csv is not None:
            # ── CSV replay mode (no SUMO) ─────────────────────────────────────
            print(f"Using CSV position replay: {pos_csv}")
            rsu_k = getattr(scenario_config, 'rsu_sample_k', 0) or 0
            fhr   = getattr(scenario_config, 'fragment_hour_range', None)
            core = CSVEnv(
                pos_csv=pos_csv,
                rsu_sample_k=rsu_k,
                fragment_hour_range=fhr,
                **_core_kwargs(unique_log_dir),
            )
        else:
            # ── SUMO live mode ────────────────────────────────────────────────
            ensure_sumo_env()
            if sumocfg_override:
                sumo_cfg = Path(sumocfg_override)
                timeframe_begin = None
                timeframe_end = None
                if scenario_config.timeframe:
                    timeframe_begin = scenario_config.timeframe[0] * 3600
                    timeframe_end = scenario_config.timeframe[1] * 3600
                print(f"Using override SUMO config: {sumo_cfg}")
            elif scenario_config.date:
                resolved = resolve_berlin_data(
                    date=scenario_config.date,
                    timeframe=scenario_config.timeframe,
                    project_root=PROJECT_ROOT
                )
                sumo_cfg = resolved['config_path']
                timeframe_begin = resolved['begin_time']
                timeframe_end = resolved['end_time']
                timeframe_str = f" (timeframe: {scenario_config.timeframe[0]}-{scenario_config.timeframe[1]}h)" if scenario_config.timeframe else " (full 24h)"
                print(f"Using SUMO config: {sumo_cfg}{timeframe_str}")
                print(f"  Episode length: {scenario_config.episode_length_steps} steps")
                print(f"  Episodes will sample random windows from {timeframe_begin}s to {timeframe_end}s")
            else:
                sumo_cfg = SUMO_CONFIG
                timeframe_begin = None
                timeframe_end = None
                print(f"Using default SUMO config: {sumo_cfg}")

            core = CoreEnv(
                sumo_cfg=sumo_cfg,
                begin_time=timeframe_begin,
                end_time=timeframe_end,
                traci_label=f"train-{os.getpid()}-{int(time.time()*1e6)%1_000_000}",
                demand_scale=getattr(scenario_config, 'demand_scale', None),
                **_core_kwargs(unique_log_dir),
            )

        env = OffloadDecisionEnv(core, feature_fn=feature_fn, use_action_mask=True)
        env = Monitor(env)
        return env

    print(f"  State space version: v{state_space_version} ({feature_fn.__name__})")
    print(f"  Task profile version: v{task_profile_version}")
    print(f"  RSU profile version: v{rsu_profile_version} ({rsu_profile['cores']} cores @ {rsu_profile['per_core_hz']/1e9:.1f} GHz = {rsu_total_hz/1e9:.0f} GHz total)")
    _eff_lambda = (scenario_config.lambda_per_veh_per_sec
                   if getattr(scenario_config, 'lambda_per_veh_per_sec', None) is not None
                   else LAMBDA_PER_VEH_PER_SEC)
    print(f"  RSU range: {RSU_DEFAULT_RANGE_M}m | Lambda: {_eff_lambda} tasks/s/veh (effective)")
    env = DummyVecEnv([_make_env])
    env.seed(scenario_config.seed)
    return env
