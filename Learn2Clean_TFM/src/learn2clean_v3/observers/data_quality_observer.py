"""
DataQualityObserver — V3 improvement #2.

Produces a richer state representation by combining:
  - Dataset shape stats (rows, cols)
  - Per-column missing rates (mean + max)
  - Distribution shift from the reference (Wasserstein distance)
  - Skewness and kurtosis summary statistics
  - Duplicate ratio
  - Class balance (supervised only)
  - Action history (binary vector)

This gives the RL agent much more information about *why* the data needs
cleaning than V2's simple DataStatsObserver.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from scipy import stats

from learn2clean_v3.observers.base_observer import BaseObserver
from learn2clean_v3.types import ActionHistory, Features, ObservationVector, OptionalTarget


class DataQualityObserver(BaseObserver):
    """
    Parameters
    ----------
    include_drift : bool
        Include Wasserstein distance from reference distribution.
    include_skewness : bool
        Include mean absolute skewness of numeric columns.
    include_kurtosis : bool
        Include mean absolute excess kurtosis of numeric columns.
    include_missing_per_column : bool
        Include (mean missing rate, max missing rate) across columns.
    """

    # Fixed-size feature vector components and their widths
    _N_SHAPE = 2          # rows_ratio, col_count
    _N_MISSING = 2        # mean_missing, max_missing
    _N_DIST = 2           # skewness, kurtosis
    _N_DRIFT = 1          # wasserstein_drift
    _N_BALANCE = 1        # class_balance (0 if unsupervised)
    _N_DUPLICATE = 1      # duplicate_ratio

    def __init__(
        self,
        include_drift: bool = True,
        include_skewness: bool = True,
        include_kurtosis: bool = True,
        include_missing_per_column: bool = True,
    ) -> None:
        self._include_drift = include_drift
        self._include_skewness = include_skewness
        self._include_kurtosis = include_kurtosis
        self._include_missing_per_col = include_missing_per_column

        self._reference_X: Optional[Features] = None
        self._original_n_rows: int = 1

    def set_reference(self, X: Features) -> None:
        """Called once at env.reset() to capture the original distribution."""
        self._reference_X = X.copy()
        self._original_n_rows = max(len(X), 1)

    def _feature_dim(self) -> int:
        dim = self._N_SHAPE + self._N_MISSING + self._N_BALANCE + self._N_DUPLICATE
        if self._include_drift:
            dim += self._N_DRIFT
        if self._include_skewness:
            dim += 1
        if self._include_kurtosis:
            dim += 1
        return dim

    def observation_space(self, n_actions: int) -> gym.Space:
        dim = self._feature_dim() + n_actions   # + action history
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)

    def observe(
        self,
        X: Features,
        y: OptionalTarget,
        action_history: ActionHistory,
        n_actions: int,
    ) -> ObservationVector:
        numeric = X.select_dtypes(include="number")
        n_rows = max(len(X), 1)

        # --- Shape ---
        row_ratio = n_rows / self._original_n_rows
        col_count = len(X.columns) / max(1, len(self._reference_X.columns)) if self._reference_X is not None else 1.0

        # --- Missing ---
        missing_rates = X.isna().mean()
        mean_missing = float(missing_rates.mean())
        max_missing = float(missing_rates.max()) if len(missing_rates) > 0 else 0.0

        # --- Duplicates ---
        dup_ratio = float(X.duplicated().sum() / n_rows)

        # --- Class balance ---
        balance = 0.0
        if y is not None:
            try:
                y_arr = np.asarray(y)
                if len(y_arr) > 0:
                    unique, counts = np.unique(y_arr, return_counts=True)
                    if len(unique) > 1:
                        balance = float(counts.min() / counts.max())
            except Exception:
                balance = 0.0

        features: list[float] = [row_ratio, col_count, mean_missing, max_missing, dup_ratio, balance]

        # --- Distribution stats ---
        if self._include_skewness and len(numeric.columns) > 0:
            skew_vals = numeric.apply(lambda c: abs(c.dropna().skew()) if len(c.dropna()) > 2 else 0.0)
            features.append(float(skew_vals.mean()))

        if self._include_kurtosis and len(numeric.columns) > 0:
            kurt_vals = numeric.apply(lambda c: abs(c.dropna().kurtosis()) if len(c.dropna()) > 3 else 0.0)
            features.append(float(kurt_vals.mean()))

        # --- Wasserstein drift from reference ---
        if self._include_drift:
            drift = self._wasserstein_drift(numeric)
            features.append(drift)

        # --- Action history (binary) ---
        history_vec = np.zeros(n_actions, dtype=np.float32)
        for idx in action_history:
            if 0 <= idx < n_actions:
                history_vec[idx] = 1.0

        obs = np.array(features, dtype=np.float32)
        return np.concatenate([obs, history_vec])

    def _wasserstein_drift(self, numeric: pd.DataFrame) -> float:
        if self._reference_X is None or len(numeric.columns) == 0:
            return 0.0
        ref_numeric = self._reference_X.select_dtypes(include="number")
        shared_cols = [c for c in numeric.columns if c in ref_numeric.columns]
        if not shared_cols:
            return 0.0

        distances: list[float] = []
        for col in shared_cols:
            curr = numeric[col].dropna().values
            ref = ref_numeric[col].dropna().values
            if len(curr) < 2 or len(ref) < 2:
                continue
            try:
                # Wasserstein distance via scipy (1D marginals)
                w = float(stats.wasserstein_distance(curr, ref))
                # Normalise by std of reference to make it scale-invariant
                ref_std = float(np.std(ref)) or 1.0
                distances.append(min(w / ref_std, 5.0))   # cap at 5
            except Exception:
                continue

        return float(np.mean(distances)) if distances else 0.0
