
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Type

import gymnasium as gym

logger = logging.getLogger(__name__)


class PretrainedPolicyLoader:

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
        try:
            from stable_baselines3.common.base_class import BaseAlgorithm
        except ImportError as exc:
            raise ImportError("stable-baselines3 is required.") from exc

        if algorithm_class is None:
            algorithm_class = self._detect_algorithm()

        logger.info("Loading checkpoint: %s  →  %s", self._path, algorithm_class.__name__)

        source_model = algorithm_class.load(str(self._path), env=None)

        target_model = algorithm_class(
            "MlpPolicy",
            target_env,
            **model_kwargs,
        )

        transferred = self._transfer_weights(source_model, target_model)
        logger.info(
            "Transferred %d / %d parameter tensors.",
            transferred,
            len(list(target_model.policy.parameters())),
        )

        if self._freeze_layers > 0:
            self._freeze(target_model)

        return target_model


    def _detect_algorithm(self) -> Any:
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
