"""Minimal training orchestration"""
import os
# Pin BLAS/OMP to a single thread BEFORE importing torch/numpy heavy paths.
# The policy net is a tiny 64x64 MLP; multi-threading it on a many-core node
# makes DQN ~27x SLOWER (thread-dispatch overhead >> compute). Measured on a
# 20-core host: 41 steps/s (default threads) vs 1102 steps/s (1 thread).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import random
import torch
torch.set_num_threads(1)

REFACTOR_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REFACTOR_ROOT / "src"))

from training_config.config_loader import create_policy_config, create_scenario_config
from environment.env_factory import create_offloading_env
from train.model_factory import create_drl_model
# NOTE: models.model_registry is imported lazily inside the (optional) register
# block below — it is not present in this branch, and --no-register must work
# without it. See the `if not args.no_register:` guard.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'dqn'])
    parser.add_argument('--scenario', type=str, default='default')
    parser.add_argument('--timesteps', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no-register', action='store_true')
    parser.add_argument('--date', type=str, default=None,
                        help='Date for SUMO data in format YYYY-MM-DD (e.g., "2024-10-16")')
    parser.add_argument('--timeframe', type=str, default=None,
                        help='Timeframe as "START-END" (e.g., "7-9" for 7am-9am)')
    parser.add_argument('--state-space-version', type=int, default=1, choices=[1, 2, 3, 4, 5, 6],
                        help='State space version: 1=oracular (34), 2=realistic (22), 3=v2+context (24), 4=v3+est_proc (27), 5=v4+queue_wait (29), 6=v5+local_queue (30)')
    parser.add_argument('--task-profile-version', type=int, default=1, choices=[1, 3, 4, 5],
                        help='Task profile version: 1=original heavy, 3=realistic edge compute, 4=v3+achievable deadlines')
    parser.add_argument('--rsu-profile-version', type=int, default=1, choices=[1, 2, 3],
                        help='RSU profile version: 1=overpowered (168 GHz), 2=realistic (40 GHz), 3=roadside unit (16 GHz)')
    parser.add_argument('--episode-length', type=int, default=100,
                        help='Episode length in SUMO steps (default: 100)')
    parser.add_argument('--no-v2v', action='store_true',
                        help='Disable V2V offloading (RSU-only mode)')
    parser.add_argument('--peer-selection', type=str, default='best', choices=['best', 'nearest'],
                        help='V2V peer selection strategy: best (min finish time) or nearest (geographic)')
    parser.add_argument('--veh-per-core-hz-mean', type=float, default=None,
                        help='Vehicle per-core CPU Hz mean (default: 1.8 GHz from config.py)')
    parser.add_argument('--veh-per-core-hz-std', type=float, default=None,
                        help='Vehicle per-core CPU Hz std dev (default: 0.4 GHz from config.py)')
    parser.add_argument('--veh-per-core-hz-min', type=float, default=None,
                        help='Vehicle per-core CPU Hz minimum (default: 1.2 GHz from config.py)')
    parser.add_argument('--lambda-per-veh', type=float, default=None,
                        help='Task arrival rate per vehicle per second (default: use config.py LAMBDA_PER_VEH_PER_SEC=0.3)')
    parser.add_argument('--reward-type', type=str, default=None,
                        choices=['deadline_only', 'latency_only'],
                        help='Reward function override (default: current energy+deadline composite)')
    parser.add_argument('--sumocfg', type=str, default=None,
                        help='Override SUMO config file (bypasses date resolver, e.g. for A100 network)')
    parser.add_argument('--pos-csv', type=str, default=None,
                        help='Pre-recorded vehicle positions CSV — bypasses SUMO entirely')
    parser.add_argument('--rsu-sample-k', type=int, default=0,
                        help='CSV mode: sample K RSUs per episode (0 = use all RSUs)')
    parser.add_argument('--fragment-hours', type=int, nargs=2, default=None, metavar=('H0', 'H1'),
                        help='CSV mode: restrict fragments to start times in [H0, H1) hours (e.g. 7 13)')
    return parser.parse_args()


def parse_timeframe(timeframe_str: str):
    """Parse timeframe string like '7-9' into tuple (7, 9)"""
    if not timeframe_str:
        return None
    try:
        start, end = timeframe_str.split('-')
        return (int(start), int(end))
    except (ValueError, AttributeError):
        raise ValueError(f"Invalid timeframe format: {timeframe_str}. Expected format: 'START-END' (e.g., '7-9')")


def main():
    args = parse_args()

    timeframe = parse_timeframe(args.timeframe) if args.timeframe else None

    scenario_config = create_scenario_config(
        name=args.scenario,
        seed=args.seed,
        date=args.date,
        timeframe=timeframe,
        episode_length_steps=args.episode_length,
        state_space_version=args.state_space_version,
        task_profile_version=args.task_profile_version,
        rsu_profile_version=args.rsu_profile_version,
        v2v_range_m=0.0 if args.no_v2v else None,
        lambda_per_veh_per_sec=args.lambda_per_veh,
        reward_type=args.reward_type,
        peer_selection=args.peer_selection,
        veh_per_core_hz_mean=args.veh_per_core_hz_mean,
        veh_per_core_hz_std=args.veh_per_core_hz_std,
        veh_per_core_hz_min=args.veh_per_core_hz_min,
        rsu_sample_k=args.rsu_sample_k,
        fragment_hour_range=tuple(args.fragment_hours) if args.fragment_hours else None,
    )
    policy_config = create_policy_config(algorithm=args.algorithm)

    np.random.seed(args.seed)
    random.seed(args.seed)

    env = create_offloading_env(scenario_config, sumocfg_override=args.sumocfg,
                                pos_csv=getattr(args, 'pos_csv', None))
    env.reset()

    # Set up directories
    save_dir = REFACTOR_ROOT / "results" / "trained_models" / args.algorithm / args.scenario
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = save_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Set up tensorboard logging
    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    if args.resume:
        if args.algorithm == 'ppo':
            from stable_baselines3 import PPO
            model = PPO.load(args.resume, env=env)
        else:
            from stable_baselines3 import DQN
            model = DQN.load(args.resume, env=env)
        model.tensorboard_log = str(log_dir)
    else:
        model = create_drl_model(
            algorithm=policy_config.algorithm,
            env=env,
            hyperparameters=policy_config.hyperparameters,
            seed=args.seed,
            tensorboard_log=str(log_dir)
        )

    print(f"\n{'='*60}")
    print(f"Training {args.algorithm.upper()} on scenario: {args.scenario}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"TensorBoard logs: {log_dir}")
    print(f"{'='*60}\n")

    from train.debug_callback import DebugProgressCallback
    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

    debug_callback = DebugProgressCallback(verbose=1, log_freq=1000)

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=str(checkpoint_dir),
        name_prefix="model_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False
    )

    # Combine callbacks
    callbacks = CallbackList([debug_callback, checkpoint_callback])

    start_time = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callbacks,
                reset_num_timesteps=(args.resume is None))
    duration = time.time() - start_time

    # Save model
    model_path = run_dir / "model"
    model.save(model_path)

    print(f"\n{'='*60}")
    print(f"✓ Training complete!")
    print(f"✓ Model saved: {model_path}.zip")
    print(f"✓ Duration: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"\nView training logs:")
    print(f"  tensorboard --logdir {log_dir}")
    print(f"{'='*60}\n")

    # Register
    if not args.no_register:
        from models.model_registry import get_registry  # lazy: only if registering
        registry = get_registry()
        model_id = registry.register(
            model_path=model_path.with_suffix('.zip'),
            algorithm=args.algorithm,
            scenario=args.scenario,
            config={'simulation': {'seed': args.seed}},
            training_info={'timesteps': args.timesteps, 'duration_seconds': duration},
            log_dir=str(log_dir),
            tags=['refactor', f'state_space_v{args.state_space_version}', f'task_profile_v{args.task_profile_version}', f'rsu_v{args.rsu_profile_version}'],
            status='experimental',
            state_space_version=args.state_space_version,
            task_profile_version=args.task_profile_version
        )
        print(f"✓ Registered as: {model_id}")

    env.close()


if __name__ == "__main__":
    main()