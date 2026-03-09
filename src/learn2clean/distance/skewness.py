from abc import ABC

import numpy as np

from learn2clean.distance.base_moment import BaseStatisticalMomentDistance


# --- INTERMEDIATE CLASS (Configuration) ---
class _BaseSkewness(BaseStatisticalMomentDistance, ABC):
    @property
    def statistic_method_name(self) -> str:
        return "skew"


# --- CONCRETE IMPLEMENTATIONS ---
class MeanSkewnessDistance(_BaseSkewness):
    """
    Calculates the absolute difference of the mean Skewness across columns.
    Represents the average shift in distribution asymmetry.
    """

    name = "MeanSkewness"

    def _calculate_from_vectors(self, vec_p: np.ndarray, vec_q: np.ndarray) -> float:
        return abs(vec_q.mean() - vec_p.mean())


class MaxSkewnessDistance(_BaseSkewness):
    """
    Calculates the absolute difference of the maximum absolute Skewness.

    Useful for detecting if the 'worst' outliers (most skewed columns)
    have been addressed.
    Formula: | Max(|Q|) - Max(|P|) |
    """

    name = "MaxSkewness"

    def _calculate_from_vectors(self, vec_p: np.ndarray, vec_q: np.ndarray) -> float:
        # Compare maximum magnitude
        return abs(np.max(np.abs(vec_q)) - np.max(np.abs(vec_p)))


class MedianSkewnessDistance(_BaseSkewness):
    """
    Calculates the absolute difference of the median Skewness.
    Robust to isolated outliers or single extreme columns.
    """

    name = "MedianSkewness"

    def _calculate_from_vectors(self, vec_p: np.ndarray, vec_q: np.ndarray) -> float:
        return abs(np.median(vec_q) - np.median(vec_p))
