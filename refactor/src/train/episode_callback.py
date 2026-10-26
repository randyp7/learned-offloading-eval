"""
Custom callback to track and log episode rewards
This explicitly handles episode completion and logs to TensorBoard
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeRewardCallback(BaseCallback):
    """
    Custom callback for logging episode rewards to TensorBoard.

    This callback tracks episode-level statistics explicitly,
    bypassing any issues with Monitor wrapper compatibility.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = []
        self.current_lengths = []

    def _on_training_start(self) -> None:
        # Initialize tracking for each environment
        n_envs = self.training_env.num_envs
        self.current_rewards = [0.0] * n_envs
        self.current_lengths = [0] * n_envs

    def _on_step(self) -> bool:
        """
        Called at each step. Track rewards and check for episode completion.
        """
        # DEBUG: Print what's available in self.locals
        if self.num_timesteps <= 1024:  # Only print during first iteration
            print(f"\n[DEBUG @step={self.num_timesteps}] self.locals keys: {list(self.locals.keys())}")

        # Get rewards and dones from the vectorized environment
        # self.locals contains: rewards, dones, infos
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        # DEBUG: Print if we have dones
        if dones and any(dones):
            print(f"\n[DEBUG @step={self.num_timesteps}] EPISODE COMPLETE DETECTED!")
            print(f"  dones: {dones}")
            print(f"  rewards: {rewards}")
            print(f"  infos: {infos}")

        if self.locals.get('done'):
            print("EPISODE COMPLETED!!!!!")

        # Update current episode stats
        for i, (reward, done, info) in enumerate(zip(rewards, dones, infos)):
            self.current_rewards[i] += reward
            self.current_lengths[i] += 1

            # Check if episode completed
            if done:
                # Episode finished - log statistics
                ep_reward = self.current_rewards[i]
                ep_length = self.current_lengths[i]

                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)

                if self.verbose > 0:
                    print(f"Episode complete: reward={ep_reward:.2f}, length={ep_length}")

                # Log to TensorBoard
                self.logger.record("rollout/ep_rew_mean", np.mean(self.episode_rewards[-100:]))
                self.logger.record("rollout/ep_len_mean", np.mean(self.episode_lengths[-100:]))
                self.logger.record("rollout/ep_rew_last", ep_reward)
                self.logger.record("rollout/ep_count", len(self.episode_rewards))

                # Reset tracking for this environment
                self.current_rewards[i] = 0.0
                self.current_lengths[i] = 0

        return True  # Continue training

    def _on_training_end(self) -> None:
        """
        Called when training ends. Log final statistics.
        """
        if self.episode_rewards:
            self.logger.record("rollout/ep_rew_final_mean", np.mean(self.episode_rewards))
            self.logger.record("rollout/ep_rew_final_std", np.std(self.episode_rewards))
            self.logger.record("rollout/ep_len_final_mean", np.mean(self.episode_lengths))
            self.logger.record("rollout/total_episodes", len(self.episode_rewards))

            if self.verbose > 0:
                print(f"\nTraining complete: {len(self.episode_rewards)} episodes")
                print(f"  Mean reward: {np.mean(self.episode_rewards):.2f} ± {np.std(self.episode_rewards):.2f}")
                print(f"  Mean length: {np.mean(self.episode_lengths):.1f}")
