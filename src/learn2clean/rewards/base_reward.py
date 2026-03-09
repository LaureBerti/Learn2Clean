from abc import ABC, abstractmethod

from learn2clean.types import Features, OptionalFeatures, OptionalTarget


class BaseReward(ABC):
    def __init__(self, initial_X: OptionalFeatures, initial_y: OptionalTarget):
        self.initial_X = initial_X.copy() if hasattr(initial_X, "copy") else initial_X
        self.initial_y = initial_y.copy() if hasattr(initial_y, "copy") else initial_y

    @abstractmethod
    def reset(self):
        """
        Reset any internal state of the reward calculator.
        Must be called at the start of each episode.
        """
        pass

    @abstractmethod
    def __call__(self, X: Features, y: OptionalTarget) -> float:
        """
        Calculate the absolute utility score of the current dataset.

        Args:
            X: Current features.
            y: Current target.

        Returns:
            float: A score representing data quality (e.g., between 0.0 and 1.0).
        """
        pass
