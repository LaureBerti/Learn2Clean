
from __future__ import annotations

import logging
import random
from typing import List, Optional

import numpy as np

from learn2clean_v3.envs.sequential_cleaning_env_v3 import SequentialCleaningEnvV3
from learn2clean_v3.types import Transition

logger = logging.getLogger(__name__)


class OfflineRLWrapper:

    def __init__(
        self,
        env: SequentialCleaningEnvV3,
        max_sequences: int = 500,
        max_seq_len: int = 5,
        seed: int = 42,
    ) -> None:
        self._env = env
        self._max_sequences = max_sequences
        self._max_seq_len = min(max_seq_len, env._max_steps)
        self._seed = seed
        self._buffer: List[Transition] = []


    def build_buffer(self) -> List[Transition]:
        rng = random.Random(self._seed)
        n_actions = self._env.action_space.n
        self._buffer = []

        logger.info(
            "Building offline buffer: %d sequences, max_len=%d",
            self._max_sequences,
            self._max_seq_len,
        )

        for seq_idx in range(self._max_sequences):
            obs, _ = self._env.reset()
            seq_len = rng.randint(1, self._max_seq_len)
            action_pool = list(range(n_actions))

            for _ in range(seq_len):
                if not action_pool:
                    break
                action = rng.choice(action_pool)
                action_pool = [a for a in action_pool if a != action]

                next_obs, reward, terminated, truncated, info = self._env.step(action)
                done = terminated or truncated

                self._buffer.append(
                    Transition(
                        obs=obs.astype(np.float32),
                        action=action,
                        reward=float(reward),
                        next_obs=next_obs.astype(np.float32),
                        done=done,
                        info=info,
                    )
                )
                obs = next_obs
                if done:
                    break

            if (seq_idx + 1) % 100 == 0:
                logger.info("  %d / %d sequences done", seq_idx + 1, self._max_sequences)

        logger.info("Buffer ready: %d transitions collected.", len(self._buffer))
        return self._buffer


    def train_offline_dqn(
        self,
        total_timesteps: int = 20_000,
        learning_rate: float = 1e-4,
        batch_size: int = 64,
        verbose: int = 1,
    ):
        try:
            from stable_baselines3 import DQN
            from stable_baselines3.common.buffers import ReplayBuffer as SB3Buffer
        except ImportError as exc:
            raise ImportError("stable-baselines3 is required for offline DQN training.") from exc

        if not self._buffer:
            raise RuntimeError("Buffer is empty — call build_buffer() first.")

        from stable_baselines3.common.vec_env import DummyVecEnv
        vec_env = DummyVecEnv([lambda: self._env])

        model = DQN(
            "MlpPolicy",
            vec_env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            buffer_size=len(self._buffer) + 1000,
            learning_starts=0,
            verbose=verbose,
            seed=self._seed,
        )

        obs_dim = self._buffer[0].obs.shape[0]
        logger.info("Filling SB3 replay buffer with %d transitions...", len(self._buffer))
        for t in self._buffer:
            model.replay_buffer.add(
                obs=t.obs.reshape(1, -1),
                next_obs=t.next_obs.reshape(1, -1),
                action=np.array([[t.action]]),
                reward=np.array([t.reward]),
                done=np.array([float(t.done)]),
                infos=[t.info],
            )

        logger.info("Training offline DQN for %d timesteps...", total_timesteps)
        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        return model


    @property
    def buffer(self) -> List[Transition]:
        return list(self._buffer)

    def buffer_stats(self) -> dict:
        if not self._buffer:
            return {}
        rewards = [t.reward for t in self._buffer]
        return {
            "n_transitions": len(self._buffer),
            "mean_reward": float(np.mean(rewards)),
            "max_reward": float(np.max(rewards)),
            "min_reward": float(np.min(rewards)),
            "std_reward": float(np.std(rewards)),
        }
