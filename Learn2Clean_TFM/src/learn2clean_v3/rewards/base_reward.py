"""Abstract base class for all reward functions."""

from __future__ import annotations

from abc import ABC, abstractmethod

from learn2clean_v3.types import Features, OptionalTarget


class BaseReward(ABC):
    """
    A reward function maps (X, y) → float.

    The environment calls ``reset()`` at the start of each episode and
    ``__call__(X, y)`` after each action step.
    """

    _name_override: str = ""   # set via ``reward.set_name(...)`` to disambiguate instances

    @abstractmethod
    def reset(self, X_initial: Features, y_initial: OptionalTarget) -> None:
        """Store initial state for per-episode normalisation."""

    @abstractmethod
    def __call__(self, X: Features, y: OptionalTarget) -> float:
        """Return a scalar reward in [−1, 1] (convention, not enforced)."""

    def set_name(self, name: str) -> "BaseReward":
        """Override the display name (useful when two instances share a class)."""
        self._name_override = name
        return self

    @property
    def name(self) -> str:
        return self._name_override or self.__class__.__name__
