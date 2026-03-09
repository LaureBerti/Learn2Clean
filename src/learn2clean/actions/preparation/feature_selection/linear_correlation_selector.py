from typing import Any

from sklearn.feature_selection import f_regression, SelectKBest

from .base_feature_selector import BaseFeatureSelector


class LinearCorrelationSelector(BaseFeatureSelector):
    """
    Feature selection using Linear Correlation (LC).

    This action selects the top-k features most correlated with a numeric target
    column using scikit-learn's `SelectKBest` and the `f_regression` scoring
    function (F-statistic based on linear correlation).

    The selection logic is implemented in the inherited `fit` method, and
    `transform` applies the selection using the fitted state.

    Notes
    -----
    - This action requires a numeric target column.
    - Features available for selection are determined by `self.columns` and `self.exclude`.
    """

    def __init__(self, k: int = 10, **params: Any) -> None:
        self.k = k
        selector_instance = SelectKBest(score_func=f_regression, k=self.k)
        super().__init__(selector=selector_instance, **params)
