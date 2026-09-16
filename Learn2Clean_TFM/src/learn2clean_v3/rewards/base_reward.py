
from __future__ import annotations

from abc import ABC, abstractmethod

from learn2clean_v3.types import Features, OptionalTarget


class BaseReward(ABC):

    _name_override: str = ""

    @abstractmethod
    def reset(self, X_initial: Features, y_initial: OptionalTarget) -> None:
        pass

    @abstractmethod
    def __call__(self, X: Features, y: OptionalTarget) -> float:
        pass

    def set_name(self, name: str) -> "BaseReward":
        self._name_override = name
        return self

    @property
    def name(self) -> str:
        return self._name_override or self.__class__.__name__
