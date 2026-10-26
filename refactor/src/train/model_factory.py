from stable_baselines3 import PPO, DQN
from torch import nn

algorithm_map = {
        'ppo': PPO,
        'dqn': DQN,
    }
activation_map = {
    'tanh': nn.Tanh,
    'relu': nn.ReLU,
    'elu': nn.ELU
}

def create_drl_model(
    algorithm: str,
    env,
    hyperparameters: dict,
    seed: int,
    tensorboard_log: str = None
):
    if algorithm.lower() not in algorithm_map:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    AlgorithmClass = algorithm_map[algorithm.lower()]

    policy_kwargs = {
        'net_arch': hyperparameters.network_architecture.net_arch,
        'activation_fn': activation_map[
            hyperparameters.network_architecture.activation
        ]
    }

    if algorithm.lower() == 'ppo':
        model = AlgorithmClass(
            'MlpPolicy',
            env,
            learning_rate=hyperparameters.learning_rate,
            n_steps=hyperparameters.n_steps,
            batch_size=hyperparameters.batch_size,
            gamma=hyperparameters.gamma,
            gae_lambda=hyperparameters.gae_lambda,
            clip_range=hyperparameters.clip_range,
            ent_coef=hyperparameters.ent_coef,
            vf_coef=hyperparameters.vf_coef,
            max_grad_norm=hyperparameters.max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=hyperparameters.training.verbose,
            seed=seed,
            tensorboard_log=tensorboard_log
        )
    elif algorithm.lower() == 'dqn':
        model = AlgorithmClass(
            'MlpPolicy',
            env,
            learning_rate=hyperparameters.learning_rate,
            buffer_size=hyperparameters.buffer_size,
            learning_starts=hyperparameters.learning_starts,
            batch_size=hyperparameters.batch_size,
            tau=hyperparameters.tau,
            gamma=hyperparameters.gamma,
            train_freq=hyperparameters.train_freq,
            gradient_steps=hyperparameters.gradient_steps,
            target_update_interval=hyperparameters.target_update_interval,
            exploration_fraction=hyperparameters.exploration_fraction,
            exploration_initial_eps=hyperparameters.exploration_initial_eps,
            exploration_final_eps=hyperparameters.exploration_final_eps,
            policy_kwargs=policy_kwargs,
            verbose=hyperparameters.training.verbose,
            seed=seed,
            tensorboard_log=tensorboard_log
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return model  