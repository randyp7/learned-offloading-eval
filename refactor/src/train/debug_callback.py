"""
Debug callback to track episode progress during training
"""
from stable_baselines3.common.callbacks import BaseCallback


class DebugProgressCallback(BaseCallback):
    """
    Logs core environment progress to help debug episode completion
    """

    def __init__(self, verbose=1, log_freq=1000):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.last_core_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0 or self.num_timesteps == 1:
            # Access core env through wrappers: DummyVecEnv -> Monitor -> OffloadDecisionEnv -> CoreEnv
            try:
                core_env = self.training_env.envs[0].env.e
                core_step = core_env.step_count
                max_steps = core_env.max_steps
                progress = (core_step / max_steps) * 100 if max_steps > 0 else 0

                print(f"\n[Progress @{self.num_timesteps} agent steps]")
                print(f"  Core env: step {core_step}/{max_steps} ({progress:.1f}%)")
                print(f"  Core steps since last check: {core_step - self.last_core_step}")

                self.last_core_step = core_step

                # Check if we're close to episode end
                if core_step >= max_steps - 5:
                    print(f"  ⚠️  Near episode end!")

            except Exception as e:
                print(f"  Error accessing core env: {e}")

        return True
