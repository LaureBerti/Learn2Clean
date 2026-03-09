from abc import ABC, abstractmethod
from typing import Tuple, Any

import numpy as np
import pandas as pd

from learn2clean.distance.base_distance import BaseDistance


class BaseBinningDistance(BaseDistance, ABC):
    """
    Base abstract class for metrics that require binning or probability alignment
    (e.g., Chi-Squared, KL Divergence, Jensen-Shannon).

    It handles the conversion of raw data into aligned probability vectors (histograms).
    """

    DEFAULT_PARAMS: dict[str, Any] = {
        "bins": 50,  # Number of bins for numeric data
        "epsilon": 1e-10,  # Smoothing factor available for subclasses
    }

    def _get_aligned_probabilities(
        self, u: pd.Series, v: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms two Series into aligned probability vectors summing to 1.

        Note: This method returns raw probabilities. It does NOT apply epsilon smoothing automatically.
        Subclasses should use self.params['epsilon'] if they need to handle zeros.
        """
        is_numeric = pd.api.types.is_numeric_dtype(u) and pd.api.types.is_numeric_dtype(
            v
        )
        bins = self.params.get("bins", 50)

        if is_numeric:
            # 1. Numeric: Histogram alignment
            global_min = min(u.min(), v.min())
            global_max = max(u.max(), v.max())

            # Edge case: constant columns
            if global_min == global_max:
                return np.array([1.0]), np.array([1.0])

            common_bins = np.linspace(global_min, global_max, bins + 1)
            p_hist, _ = np.histogram(u, bins=common_bins, density=False)
            q_hist, _ = np.histogram(v, bins=common_bins, density=False)

        else:
            # 2. Categorical: Vocabulary alignment
            u_str, v_str = u.astype(str), v.astype(str)
            all_cats = np.union1d(u_str.unique(), v_str.unique())

            # Count occurrences
            p_hist = u_str.value_counts().reindex(all_cats, fill_value=0).to_numpy()
            q_hist = v_str.value_counts().reindex(all_cats, fill_value=0).to_numpy()

        # 3. Normalization (Counts -> Probabilities)
        p_sum = np.sum(p_hist)
        q_sum = np.sum(q_hist)

        p_probs = p_hist / p_sum if p_sum > 0 else p_hist
        q_probs = q_hist / q_sum if q_sum > 0 else q_hist

        return p_probs, q_probs

    @abstractmethod
    def _calculate_from_probs(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Logic specific to the metric (Chi2, KL, JS, etc.).
        Receives two normalized probability vectors of equal length.
        """
        pass

    def _calculate_metric(self, df_p: pd.DataFrame, df_q: pd.DataFrame) -> float:
        """
        Orchestrates the column-by-column comparison.
        """
        common_cols = df_p.columns.intersection(df_q.columns)
        distances = []

        for col in common_cols:
            # Data preparation: Drop NaNs to focus on distribution shape
            u = df_p[col].dropna()
            v = df_q[col].dropna()

            if u.empty or v.empty:
                continue

            try:
                p_probs, q_probs = self._get_aligned_probabilities(u, v)
                dist = self._calculate_from_probs(p_probs, q_probs)
                distances.append(dist)
            except Exception as e:
                # Log warning but continue (robustness)
                self.log.warning(f"Metric calculation failed for column {col}: {e}")
                pass

        return float(np.mean(distances)) if distances else 0.0
