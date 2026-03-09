import numpy as np
from learn2clean.rewards.base_reward import BaseReward
from learn2clean.types import Features, OptionalTarget


class CompletenessRetentionReward(BaseReward):
    """
    Reward based on Data Completeness and Row Retention.
    Formula: Completeness * sqrt(Retention)
    """

    def __init__(self, initial_X: Features, initial_y: OptionalTarget):
        super().__init__(initial_X, initial_y)
        self.initial_rows = len(initial_X)

    def reset(self) -> None:
        pass

    def __call__(self, X: Features, y: OptionalTarget) -> float:
        if X is None or X.empty:
            return -1.0
        if X.size == 0:
            return 0.0

        retention = len(X) / self.initial_rows
        missing = X.isna().sum().sum()
        completeness = 1.0 - (missing / X.size)

        return float(completeness * np.sqrt(retention))
