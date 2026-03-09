from typing import Any

from sklearn.feature_selection import VarianceThreshold

from learn2clean.actions.preparation.feature_selection.base_feature_selector import (
    BaseFeatureSelector,
)


class VarianceThresholdSelector(BaseFeatureSelector):
    """
    Feature selection using Variance Thresholding.

    This action removes all features whose variance is below a given threshold.
    It is a non-supervised method, as the selection depends only on the features (X),
    not the target (y).

    Notes
    -----
    - Features with zero variance (i.e., constant features) are removed by default
      if threshold is 0.0 (the default value).
    - Features should be scaled appropriately (e.g., Z-score or Min-Max) before
      using a non-zero threshold, as variance is scale-dependent.
    """

    def __init__(self, threshold: float = 0.0, **params: Any) -> None:
        self.threshold = threshold
        super().__init__(selector=VarianceThreshold(threshold=self.threshold), **params)
