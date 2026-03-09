from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ...data_frame_action import DataFrameAction


class ZScoreOutlierCleaner(DataFrameAction):
    """
    Implements the Z-score-based (ZSB) outlier detection and cleaning method.

    This action detects outliers by calculating the Z-score for each value in
    numeric columns. A value is considered an outlier if its Z-score (absolute value)
    exceeds a specified threshold.

    References:
        - Learn2Clean Paper: Section 3.1 "Outlier detection" (ZSB).
        - Standard definition: z = (x - mean) / std

    Attributes:
        stats_ (dict[str, dict[str, float]]):
            Stores the 'mean' and 'std' for each fitted column to ensure
            consistency when transforming new data (e.g., test sets).
    """

    # Default configuration
    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "threshold": 3.0,
        "method": "mask",  # Options: 'drop', 'mask', 'clip'
        "numeric_only": True,  # Z-score applies only to numeric data
    }

    def __init__(self, **params: Any) -> None:
        """
        Initialize the ZScoreOutlierCleaner.

        Parameters:
            threshold (float): The Z-score threshold. Values with |z| > threshold are outliers.
                               Default is 3.0.
            method (str): The strategy to handle outliers.
                          - 'drop': Remove rows containing at least one outlier.
                          - 'mask': Replace outliers with NaN (useful for subsequent imputation).
                          - 'clip': Cap values at the threshold limits (mean +/- threshold * std).
                          Default is 'mask'.
            numeric_only (bool): If True, restricts action to numeric columns. Default is True.
        """
        super().__init__(**params)
        self.stats_: dict[str, dict[str, float]] = {}

    def fit(
        self, df: pd.DataFrame, y: pd.Series | None = None
    ) -> "ZScoreOutlierCleaner":
        """
        Compute the mean and standard deviation for each selected column.

        These statistics are stored to ensure that the definition of an 'outlier'
        remains consistent when transforming subsequent datasets (e.g., test set).

        Args:
            df (pd.DataFrame): The training data.
            y (pd.Series, optional): Ignored.

        Returns:
            self: The fitted instance.
        """
        # Select columns (filtering by numeric_only is handled by the base class logic + params)
        target_cols = self.select_columns(
            df, numeric_only=self.params.get("numeric_only", True)
        )
        self._fitted_columns = target_cols

        # Calculate statistics for the selected columns
        # We verify that columns are numeric to avoid errors during calculation
        valid_cols = [
            col for col in target_cols if pd.api.types.is_numeric_dtype(df[col])
        ]

        if len(valid_cols) < len(target_cols):
            self.log_warning(
                f"Skipping {len(target_cols) - len(valid_cols)} non-numeric columns requested for Z-Score."
            )

        self.stats_ = {}
        for col in valid_cols:
            self.stats_[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
            }

        self.log_info(
            f"Fitted Z-Score stats on {len(self.stats_)} columns. "
            f"Threshold={self.params['threshold']}."
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers in the DataFrame using fitted statistics.

        Args:
            df (pd.DataFrame): The data to clean.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        # Create a copy to avoid SettingWithCopy warnings and side effects
        df_clean = df.copy()

        # Determine which columns to process
        # If fit() was called, use _fitted_columns; otherwise, fallback to dynamic selection
        target_cols = self._get_fitted_columns(
            df, numeric_only=self.params.get("numeric_only", True)
        )

        # Retrieve parameters
        threshold = self.params["threshold"]
        method = self.params["method"]

        # Validations
        valid_methods = {"drop", "mask", "clip"}
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. Expected one of {valid_methods}"
            )

        # If stats_ is empty (fitless execution), compute them on the fly for the current batch
        use_fitted_stats = bool(self.stats_)

        cols_to_process = [col for col in target_cols if col in df_clean.columns]
        if not cols_to_process:
            self.log_warning("No matching columns found for Z-Score cleaning.")
            return df_clean

        self.log_info(
            f"Applying Z-Score cleaning (threshold={threshold}, method='{method}') "
            f"on {len(cols_to_process)} columns. "
            f"Using {'fitted' if use_fitted_stats else 'current batch'} statistics."
        )

        # Keep track of rows to drop (if method is 'drop')
        rows_with_outliers = pd.Series(False, index=df_clean.index)
        total_outliers_count = 0

        for col in cols_to_process:
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                continue

            # Get stats
            if use_fitted_stats and col in self.stats_:
                mean = self.stats_[col]["mean"]
                std = self.stats_[col]["std"]
            else:
                mean = df_clean[col].mean()
                std = df_clean[col].std()

            # Avoid division by zero if std is 0 (constant column)
            if std == 0:
                continue

            # Calculate Z-score: z = (x - mean) / std
            z_scores = (df_clean[col] - mean) / std

            # Identify outliers
            is_outlier = z_scores.abs() > threshold
            outlier_count = is_outlier.sum()
            total_outliers_count += outlier_count

            if outlier_count > 0:
                if method == "drop":
                    rows_with_outliers |= is_outlier

                elif method == "mask":
                    df_clean.loc[is_outlier, col] = np.nan

                elif method == "clip":
                    # Clip values to the boundaries defined by the threshold
                    lower_bound = mean - threshold * std
                    upper_bound = mean + threshold * std
                    df_clean[col] = df_clean[col].clip(
                        lower=lower_bound, upper=upper_bound
                    )

        # Finalize 'drop' method
        if method == "drop":
            num_dropped = rows_with_outliers.sum()
            if num_dropped > 0:
                df_clean = df_clean[~rows_with_outliers]
                self.log_info(
                    f"Dropped {num_dropped} rows containing at least one outlier."
                )
            else:
                self.log_info("No rows dropped (no outliers found).")
        else:
            self.log_info(
                f"Handled {total_outliers_count} outlier values using '{method}'."
            )

        return df_clean
