from typing import Any

from sklearn.feature_selection import chi2, SelectKBest

from .base_feature_selector import BaseFeatureSelector


class ChiSquareSelector(BaseFeatureSelector):
    """
    Feature selection using the Chi-Square test (Chi²).

    This action selects the top-k features with the highest Chi-Square statistic
    relative to the target column, using scikit-learn's `SelectKBest` and the
    `chi2` scoring function.

    The Chi-Square test measures the dependence between stochastic variables.
    A higher Chi-Square value indicates a greater dependency between the feature and the target.

    Notes
    -----
    - Features (X) must be non-negative (e.g., counts or frequencies).
    - The target (y) must be categorical or discrete, as the test is based on
      contingency tables, typically used for classification problems.
    - The `fit` and `transform` logic is inherited from `BaseFeatureSelector`.
    """

    def __init__(self, k: int = 10, **params: Any) -> None:
        self.k = k
        selector_instance = SelectKBest(score_func=chi2, k=self.k)
        super().__init__(selector=selector_instance, **params)
