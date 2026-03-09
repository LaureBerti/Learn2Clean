from abc import ABC

import numpy as np

from learn2clean.distance.base_moment import BaseStatisticalMomentDistance


class _BaseVariance(BaseStatisticalMomentDistance, ABC):
    @property
    def statistic_method_name(self) -> str:
        return "var"  # Pandas shortcut for variance()


class MeanVarianceDistance(_BaseVariance):
    """
    Calculates the absolute difference of the mean Variance across columns.

    This metric is highly sensitive to Scaling actions.
    (e.g., MinMaxScaler will drastically reduce the variance of a feature).
    """

    name = "MeanVariance"

    def _calculate_from_vectors(self, vec_p: np.ndarray, vec_q: np.ndarray) -> float:
        return abs(vec_q.mean() - vec_p.mean())


class MaxVarianceDistance(_BaseVariance):
    """
    Calculates the absolute difference of the maximum Variance.

    Useful for detecting if the feature with the largest spread (magnitude)
    has been scaled down or normalized.
    """

    name = "MaxVariance"

    def _calculate_from_vectors(self, vec_p: np.ndarray, vec_q: np.ndarray) -> float:
        # Variance is always non-negative, so inner abs() is technically optional
        # but kept for consistency with other metrics.
        return abs(np.max(vec_q) - np.max(vec_p))


class MedianVarianceDistance(_BaseVariance):
    """
    Calculates the absolute difference of the median Variance.
    Represents the typical change in data spread/dispersion.
    """

    name = "MedianVariance"

    def _calculate_from_vectors(self, vec_p: np.ndarray, vec_q: np.ndarray) -> float:
        return abs(np.median(vec_q) - np.median(vec_p))
