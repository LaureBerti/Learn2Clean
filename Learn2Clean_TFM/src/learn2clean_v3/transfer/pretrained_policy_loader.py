"""
PretrainedPolicyLoader — V3 improvement #5.

Enables curriculum / transfer learning by initialising an SB3 agent's
policy network from a checkpoint trained on a different (or simpler) dataset.

Rationale
---------
- Training on Titanic builds a policy that learns "impute missing → scale → dedup".
- That policy provides a warm start for a new dataset with similar structure.
- Layer freezing allows fine-tuning only the top layers on the new task.

Usage
-----
    loader = PretrainedPolicyLoader(checkpoint_path="outputs/titanic_ppo.zip")
    model = loader.load_into(new_env, algorithm_class=PPO)
    model.learn(total_timesteps=10_000)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Type

import gymnasium as gym

logger = logging.getLogger(__name__)


class PretrainedPolicyLoader:
    """
    Parameters
    ----------
    checkpoint_path : str | Path
        Path to a saved SB3 model (.zip file).
    freeze_layers : int
        Number of MLP layers to freeze from the bottom up (0 = no freezing).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        freeze_layers: int = 0,
    ) -> None:
        self._path = Path(checkpoint_path)
        self._freeze_layers = freeze_layers

        if not self._path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self._path}")

    def load_into(
        self,
        target_env: gym.Env,
        algorithm_class: Optional[Any] = None,
        **model_kwargs: Any,
    ) -> Any:
        """
        Load a checkpoint and adapt it to *target_env*.

        Parameters
        ----------
        target_env : gym.Env
            The new environment to train in.
        algorithm_class : type | None
            SB3 algorithm class (e.g. PPO, DQN).  Auto-detected from the
            checkpoint when None.
        **model_kwargs
            Extra kwargs forwarded to the SB3 model constructor.

        Returns
        -------
        SB3 model ready for fine-tuning with ``.learn()``.
        """
        try:
            from stable_baselines3.common.base_class import BaseAlgorithm
        except ImportError as exc:
            raise ImportError("stable-baselines3 is required.") from exc

        # Auto-detect algorithm if not supplied
        if algorithm_class is None:
            algorithm_class = self._detect_algorithm()

        logger.info("Loading checkpoint: %s  →  %s", self._path, algorithm_class.__name__)

        # Load the source model (env mismatch is expected; we extract weights only)
        source_model = algorithm_class.load(str(self._path), env=None)

        # Create a fresh model for the target env
        target_model = algorithm_class(
            "MlpPolicy",
            target_env,
            **model_kwargs,
        )

        # Transfer policy weights (best-effort — ignores dimension mismatches)
        transferred = self._transfer_weights(source_model, target_model)
        logger.info(
            "Transferred %d / %d parameter tensors.",
            transferred,
            len(list(target_model.policy.parameters())),
        )

        if self._freeze_layers > 0:
            self._freeze(target_model)

        return target_model

    # ------------------------------------------------------------------

    def _detect_algorithm(self) -> Any:
        """Infer the SB3 algorithm class from the checkpoint filename."""
        from stable_baselines3 import A2C, DQN, PPO

        name = self._path.stem.lower()
        if "ppo" in name:
            return PPO
        if "dqn" in name:
            return DQN
        if "a2c" in name:
            return A2C
        logger.warning(
            "Cannot detect algorithm from filename '%s'; defaulting to PPO.", self._path.stem
        )
        return PPO

    def _transfer_weights(self, source: Any, target: Any) -> int:
        """Copy matching weight tensors from source → target policy."""
        import torch

        source_state = source.policy.state_dict()
        target_state = target.policy.state_dict()
        transferred = 0

        for key in target_state:
            if key in source_state:
                src_shape = source_state[key].shape
                tgt_shape = target_state[key].shape
                if src_shape == tgt_shape:
                    target_state[key] = source_state[key].clone()
                    transferred += 1
                else:
                    logger.debug(
                        "Shape mismatch for '%s': src=%s tgt=%s — skipped.",
                        key, src_shape, tgt_shape,
                    )

        target.policy.load_state_dict(target_state)
        return transferred

    def _freeze(self, model: Any) -> None:
        """Freeze the bottom ``freeze_layers`` layers of the policy MLP."""
        import torch.nn as nn

        linear_layers = [
            m for m in model.policy.modules() if isinstance(m, nn.Linear)
        ]
        to_freeze = linear_layers[: self._freeze_layers]
        for layer in to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

        n_frozen = sum(1 for p in model.policy.parameters() if not p.requires_grad)
        logger.info(
            "Froze %d layers (%d parameters).",
            len(to_freeze), n_frozen,
        )
