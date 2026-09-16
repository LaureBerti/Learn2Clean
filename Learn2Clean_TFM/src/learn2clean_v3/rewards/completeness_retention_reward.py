
from __future__ import annotations

import math

from learn2clean_v3.rewards.base_reward import BaseReward
from learn2clean_v3.types import Features, OptionalTarget


class CompletenessRetentionReward(BaseReward):

    def __init__(self) -> None:
        self._original_n_rows: int = 1
        self._original_n_cells: int = 1

    def reset(self, X_initial: Features, y_initial: OptionalTarget) -> None:
        self._original_n_rows = max(len(X_initial), 1)
        self._original_n_cells = max(X_initial.size, 1)

    def __call__(self, X: Features, y: OptionalTarget) -> float:
        if X is None or len(X) == 0:
            return -1.0
        n_missing = int(X.isna().sum().sum())
        total = X.size or 1
        completeness = 1.0 - n_missing / total
        retention = len(X) / self._original_n_rows
        return completeness * math.sqrt(retention)
