from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, SelectKBest

from .base_feature_selector import BaseFeatureSelector


class MutualInformationSelector(BaseFeatureSelector):
    """
    Feature selection using Mutual Information for classification.

    This action selects the top-k features with the highest estimated mutual
    information with respect to the target column using scikit-learn's
    `SelectKBest` and the `mutual_info_classif` scoring function.

    Notes
    -----
    - This method is non-parametric and can capture non-linear relationships.
    - Features (X) should be scaled appropriately (e.g., Z-score, Min-Max).
    - The target (y) must be categorical or discrete.
    """

    def __init__(
        self,
        k: int = 10,
        random_state: int | None = 42,
        n_neighbors: int = 3,
        **params: Any
    ) -> None:
        self.k = k
        self.random_state = random_state
        self.n_neighbors = n_neighbors

        def score_function(
            X: pd.DataFrame, y: pd.Series
        ) -> tuple[np.ndarray, np.ndarray]:
            scores = mutual_info_classif(
                X, y, random_state=self.random_state, n_neighbors=self.n_neighbors
            )
            return scores, np.ones_like(scores)

        super().__init__(
            selector=SelectKBest(score_func=score_function, k=self.k), **params
        )
