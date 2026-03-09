from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from learn2clean.distance.base_distance import BaseDistance


class BaseStatisticalMomentDistance(BaseDistance, ABC):
    """
    Abstract base class for distance metrics based on statistical moments
    (Skewness, Kurtosis, Variance, etc.).

    Responsibilities:
    1. Extract the requested statistic (e.g., skew, kurtosis) from all numeric columns.
    2. Handle parameters (axis, skipna).
    3. Clean NaNs and provide clean numpy vectors to child classes.
    """

    DEFAULT_PARAMS: dict[str, Any] = {
        "axis": 0,  # Calculate column-wise
        "skipna": True,  # Ignore NaNs during calculation
    }

    @property
    @abstractmethod
    def statistic_method_name(self) -> str:
        """
        Name of the Pandas method to call (e.g., 'skew', 'kurt', 'var').
        """
        pass

    def _get_vector(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extracts the cleaned vector of statistics for the given DataFrame.
        """
        axis = self.params.get("axis", 0)
        skipna = self.params.get("skipna", True)

        method_name = self.statistic_method_name

        # Safety check: Verify that the method exists on the DataFrame
        if not hasattr(df, method_name):
            raise ValueError(f"Pandas DataFrame has no method '{method_name}'")

        stat_func = getattr(df, method_name)

        # Calculation ('numeric_only=True' is vital to ignore string columns)
        series = stat_func(axis=axis, skipna=skipna, numeric_only=True)

        # Conversion and cleaning of NaNs (e.g., from constant or empty columns)
        values = series.to_numpy()
        valid_values = values[~np.isnan(values)]

        return valid_values

    @abstractmethod
    def _calculate_from_vectors(self, vec_p: np.ndarray, vec_q: np.ndarray) -> float:
        """
        Comparison logic for the two vectors (P and Q).
        Must be implemented by child classes (Mean, Max, Median...).
        """
        pass

    def _calculate_metric(self, df_p: pd.DataFrame, df_q: pd.DataFrame) -> float:
        """
        Template method that orchestrates the calculation.
        """
        vec_p = self._get_vector(df_p)
        vec_q = self._get_vector(df_q)

        # Handle empty cases (e.g., empty dataset or no numeric columns)
        if len(vec_p) == 0 or len(vec_q) == 0:
            return 0.0

        return self._calculate_from_vectors(vec_p, vec_q)
