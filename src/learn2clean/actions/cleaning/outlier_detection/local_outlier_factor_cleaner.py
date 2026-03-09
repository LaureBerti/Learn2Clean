from typing import Any, ClassVar

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from ...data_frame_action import DataFrameAction


class LocalOutlierFactorCleaner(DataFrameAction):
    """
    Implements Local Outlier Factor (LOF) outlier detection and cleaning.

    This action applies LOF univariately (column-by-column) to detect low-density
    values relative to their neighbors.

    It supports two modes compatible with ML pipelines:
    1. Stateful (fit/transform): Uses 'novelty detection' mode. The model learns
       the training data distribution and detects deviations in new data.
    2. Stateless (transform only): Uses standard outlier detection to find anomalies
       within the current batch of data.

    References:
        - Breunig, M. M., et al. (2000). LOF: identifying density-based local outliers.
    """

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "n_neighbors": 20,
        "contamination": "auto",
        "method": "mask",  # Options: 'drop', 'mask'
    }

    def __init__(self, **params: Any) -> None:
        """
        Initialize the LocalOutlierFactorCleaner.

        Parameters:
            n_neighbors (int): Number of neighbors to use by default.
            contamination (float | str): The proportion of outliers in the data set.
            method (str): Strategy to handle outliers ('drop' or 'mask').
        """
        super().__init__(**params)
        # Dictionary to store fitted LOF models for each column
        self.models_: dict[str, LocalOutlierFactor] = {}

    def fit(
        self, df: pd.DataFrame, y: pd.Series | None = None
    ) -> "LocalOutlierFactorCleaner":
        """
        Fit a LocalOutlierFactor model for each selected numeric column.
        Enables 'novelty=True' for future predictions.
        """
        # Strict restriction to numeric columns
        target_cols = self.select_columns(df, numeric_only=True)
        self._fitted_columns = target_cols

        n_neighbors = self.params["n_neighbors"]
        contamination = self.params["contamination"]
        self.models_ = {}

        for col in target_cols:
            # 1. Drop NaNs because LOF cannot handle missing values
            # 2. Use .values to convert to numpy array immediately (Avoid feature name warning)
            series_valid = df[col].dropna()

            # Safety check: LOF needs at least 2 samples to define neighbors
            if len(series_valid) < 2:
                self.log_warning(f"Column '{col}' has < 2 samples. Skipping LOF fit.")
                continue

            # 3. Reshape to (n_samples, 1) required by sklearn for single feature
            X = series_valid.values.reshape(-1, 1)

            # Adjust n_neighbors for small datasets
            current_neighbors = min(n_neighbors, len(X) - 1 if len(X) > 1 else 1)

            if current_neighbors < n_neighbors:
                self.log_debug(
                    f"Adjusting n_neighbors to {current_neighbors} for '{col}' (samples={len(X)})"
                )

            # novelty=True is essential for the fit/transform paradigm
            lof = LocalOutlierFactor(
                n_neighbors=current_neighbors,
                contamination=contamination,
                novelty=True,
            )
            lof.fit(X)
            self.models_[col] = lof

        self.log_info(
            f"Fitted LOF models (novelty=True) on {len(self.models_)} columns."
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers using LOF.
        Uses stored models if available (Novelty), else fits on the fly (Outlier).
        """
        df_clean = df.copy()

        # Use helper to get columns compatible with fit state
        target_cols = self._get_fitted_columns(df, numeric_only=True)

        method = self.params["method"]
        if method not in ["drop", "mask"]:
            raise ValueError(f"Invalid method '{method}'. Supported: 'drop', 'mask'.")

        use_fitted_models = bool(self.models_)
        cols_to_process = [col for col in target_cols if col in df_clean.columns]

        if not cols_to_process:
            self.log_warning("No matching columns found for LOF cleaning.")
            return df_clean

        self.log_info(
            f"Applying LOF cleaning (method='{method}') on {len(cols_to_process)} columns. "
            f"Mode: {'Stateful (Novelty)' if use_fitted_models else 'Stateless (Outlier Detection)'}."
        )

        rows_with_outliers = pd.Series(False, index=df_clean.index)
        total_outliers_count = 0

        for col in cols_to_process:
            series_valid = df_clean[col].dropna()

            if len(series_valid) < 2:
                continue

            # Ensure consistent Numpy shape (N, 1)
            X = series_valid.values.reshape(-1, 1)
            indices = series_valid.index

            preds: np.ndarray

            if use_fitted_models:
                if col not in self.models_:
                    continue
                lof = self.models_[col]
                # Predict: 1 = inlier, -1 = outlier
                preds = lof.predict(X)
            else:
                # Stateless: Standard Outlier Detection on current batch
                n_neighbors_param = self.params["n_neighbors"]
                current_neighbors = min(n_neighbors_param, len(X) - 1)

                lof = LocalOutlierFactor(
                    n_neighbors=current_neighbors,
                    contamination=self.params["contamination"],
                    novelty=False,  # Important: standard outlier detection
                )
                preds = lof.fit_predict(X)

            # Identify outliers (-1)
            is_outlier = preds == -1
            outlier_indices = indices[is_outlier]
            count = len(outlier_indices)
            total_outliers_count += count

            if count > 0:
                if method == "drop":
                    # Mark rows for deletion (logical OR)
                    rows_with_outliers.loc[outlier_indices] = True
                elif method == "mask":
                    # Mask immediately
                    df_clean.loc[outlier_indices, col] = np.nan

        # Finalize 'drop' method outside the loop to avoid index shifting issues during iteration
        if method == "drop":
            num_dropped = rows_with_outliers.sum()
            if num_dropped > 0:
                df_clean = df_clean[~rows_with_outliers]
                self.log_info(
                    f"Dropped {num_dropped} rows containing at least one LOF outlier."
                )
            else:
                self.log_info("No rows dropped.")
        else:
            self.log_info(f"Masked {total_outliers_count} outlier values using LOF.")

        return df_clean
