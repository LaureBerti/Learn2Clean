"""Abstract observer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np

from learn2clean_v3.types import ActionHistory, Features, ObservationVector, OptionalTarget


class BaseObserver(ABC):
    """Strategy interface for converting env state → agent observation."""

    @abstractmethod
    def observation_space(self, n_actions: int) -> gym.Space:
        """Return the Gymnasium observation space."""

    @abstractmethod
    def observe(
        self,
        X: Features,
        y: OptionalTarget,
        action_history: ActionHistory,
        n_actions: int,
    ) -> ObservationVector:
        """Convert current state to a flat numpy observation vector."""
