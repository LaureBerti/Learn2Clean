import time
from typing import Any, Dict, Tuple, SupportsFloat

import gymnasium as gym
import wandb


class WandbLoggingWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that automatically logs transitions, rewards, and metrics
    to Weights & Biases (WandB).

    It specifically looks for metrics provided in the 'info' dictionary returned
    by the environment (e.g., 'current_score', 'data_distance').
    """

    def __init__(self, env: gym.Env, log_freq: int = 1):
        """
        Initialize the WandB wrapper.

        Args:
            env: The gymnasium environment to wrap.
            log_freq: Frequency of step-level logging (default: 1, logs every step).
                      Increase this value to reduce network traffic for long episodes.
        """
        super().__init__(env)
        self.log_freq = log_freq
        self.step_count = 0
        self.episode_reward = 0.0
        self.episode_start_time = 0.0

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Resets the environment and the episode-level trackers.
        """
        obs, info = self.env.reset(**kwargs)

        self.episode_reward = 0.0
        self.episode_start_time = time.time()
        self.step_count = 0  # Reset step count for the new episode

        # Log initial state metrics if available (e.g., initial_score from Learn2CleanEnv)
        if wandb.run and "initial_score" in info:
            wandb.log({"episode/initial_score": info["initial_score"]})

        return obs, info

    def step(
        self, action: Any
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Intercepts the environment step, updates statistics, and logs to WandB.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.step_count += 1
        self.episode_reward += reward

        # Ensure WandB is active before trying to log
        if wandb.run:
            # 1. Step-level Logging (Real-time monitoring)
            if self.step_count % self.log_freq == 0:
                log_dict = {
                    "step/reward": reward,
                    "step/action_index": action,  # Useful for histograms of actions taken
                    "step/cumulative_reward": self.episode_reward,
                }

                # Dynamically extract numeric metrics from the 'info' dict
                # Learn2CleanEnv returns 'current_score' on success
                for key, value in info.items():
                    if isinstance(value, (int, float)):
                        # Prefix with 'info/' to keep the dashboard organized
                        log_dict[f"info/{key}"] = value

                wandb.log(log_dict)

            # 2. Episode-end Logging (Summary metrics)
            if terminated or truncated:
                duration = time.time() - self.episode_start_time

                # Default metrics
                summary_dict = {
                    "episode/total_reward": self.episode_reward,
                    "episode/duration_seconds": duration,
                    "episode/length": self.step_count,
                }

                # Extract specific metrics relevant to Learn2Clean if available
                # using .get() defaults ensures compatibility if keys are missing
                if "current_score" in info:
                    summary_dict["episode/final_accuracy"] = info["current_score"]

                if "data_distance" in info:
                    summary_dict["episode/final_distance"] = info["data_distance"]

                wandb.log(summary_dict)

        return obs, reward, terminated, truncated, info
