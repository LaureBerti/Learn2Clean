from abc import ABC

import numpy as np

from learn2clean.distance.base_moment import BaseStatisticalMomentDistance


# --- INTERMEDIATE CLASS ---
class _BaseKurtosis(BaseStatisticalMomentDistance, ABC):
    @property
    def statistic_method_name(self) -> str:
        return "kurt"  # Pandas shortcut for kurtosis()


# --- CONCRETE IMPLEMENTATIONS ---
class MeanKurtosisDistance(_BaseKurtosis):
    """
    Calculates the absolute difference of the mean Kurtosis across columns.
    Represents the global change in tailedness/outliers.
    """

    name = "MeanKurtosis"

    def _calculate_from_vectors(self, vec_p: np.ndarray, vec_q: np.ndarray) -> float:
        return abs(vec_q.mean() - vec_p.mean())


class MaxKurtosisDistance(_BaseKurtosis):
    """
    Calculates the absolute difference of the maximum absolute Kurtosis.

    This is the strictest metric for Outlier Removal tasks, as it focuses
    on the column with the heaviest tails.
    """

    name = "MaxKurtosis"

    def _calculate_from_vectors(self, vec_p: np.ndarray, vec_q: np.ndarray) -> float:
        return abs(np.max(np.abs(vec_q)) - np.max(np.abs(vec_p)))


class MedianKurtosisDistance(_BaseKurtosis):
    """
    Calculates the absolute difference of the median Kurtosis.
    Provides a robust view of the 'typical' column's tailedness.
    """

    name = "MedianKurtosis"

    def _calculate_from_vectors(self, vec_p: np.ndarray, vec_q: np.ndarray) -> float:
        return abs(np.median(vec_q) - np.median(vec_p))
