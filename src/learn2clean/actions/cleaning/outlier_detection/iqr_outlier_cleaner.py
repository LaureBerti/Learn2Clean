from typing import Any, ClassVar, Literal

import pandas as pd
import numpy as np

from learn2clean.actions.data_frame_action import DataFrameAction


class IQROutlierCleaner(DataFrameAction):
    """
    Implements Inter Quartile Range (IQR) outlier detection and cleaning.

    The IQR method defines outliers as data points that fall below Q1 - k*IQR
    or above Q3 + k*IQR, where Q1 and Q3 are the 25th and 75th percentiles,
    and k is a scaling factor (typically 1.5).

    Attributes:
        stats_ (dict[str, dict[str, float]]):
            Stores the calculated boundaries ('lower_bound', 'upper_bound') for each
            fitted column to ensure consistency when transforming new data.
    """

    # Default configuration
    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "factor": 1.5,  # The 'k' multiplier. 1.5 is standard, 3.0 for extreme outliers.
        "method": "mask",  # Options: 'drop', 'mask', 'clip'
        "numeric_only": True,
    }

    def __init__(self, **params: Any) -> None:
        """
        Initialize the IQROutlierCleaner.

        Parameters:
            factor (float): The multiplier for the IQR range. Default is 1.5.
            method (str): The strategy to handle outliers.
                          - 'drop': Remove rows containing at least one outlier.
                          - 'mask': Replace outliers with NaN.
                          - 'clip': Cap values at the calculated lower/upper bounds.
            numeric_only (bool): If True, restricts action to numeric columns.
        """
        super().__init__(**params)
        self.stats_: dict[str, dict[str, float]] = {}

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> "IQROutlierCleaner":
        """
        Compute the Q1, Q3, and acceptance boundaries for each selected column.

        Storing the boundaries ensures that outliers in the test set are detected
        based on the distribution of the training set.

        Args:
            df (pd.DataFrame): The training data.
            y (pd.Series, optional): Ignored.

        Returns:
            self: The fitted instance.
        """
        target_cols = self.select_columns(
            df, numeric_only=self.params.get("numeric_only", True)
        )
        self._fitted_columns = target_cols

        # Filter valid numeric columns
        valid_cols = [
            col for col in target_cols if pd.api.types.is_numeric_dtype(df[col])
        ]

        if len(valid_cols) < len(target_cols):
            self.log_warning(
                f"Skipping {len(target_cols) - len(valid_cols)} non-numeric columns requested for IQR."
            )

        factor = self.params["factor"]
        self.stats_ = {}

        for col in valid_cols:
            # Calculate Quantiles
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            # Calculate Boundaries
            lower_bound = q1 - (factor * iqr)
            upper_bound = q3 + (factor * iqr)

            self.stats_[col] = {
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
            }

        self.log_info(
            f"Fitted IQR stats on {len(self.stats_)} columns. Factor={factor}."
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers using IQR boundaries.

        Args:
            df (pd.DataFrame): The data to clean.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        df_clean = df.copy()

        # Determine columns to process
        target_cols = self._get_fitted_columns(
            df, numeric_only=self.params.get("numeric_only", True)
        )

        method = self.params["method"]
        valid_methods = {"drop", "mask", "clip"}
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. Expected one of {valid_methods}"
            )

        # Check if we use fitted stats or compute on the fly
        use_fitted_stats = bool(self.stats_)

        # Intersection of requested columns and existing columns
        cols_to_process = [col for col in target_cols if col in df_clean.columns]

        if not cols_to_process:
            self.log_warning("No matching columns found for IQR cleaning.")
            return df_clean

        # For fitless execution (on the fly), we need the factor
        factor = self.params["factor"]

        self.log_info(
            f"Applying IQR cleaning (factor={factor}, method='{method}') "
            f"on {len(cols_to_process)} columns."
        )

        rows_with_outliers = pd.Series(False, index=df_clean.index)
        total_outliers_count = 0

        for col in cols_to_process:
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                continue

            # Determine bounds
            if use_fitted_stats and col in self.stats_:
                lower_bound = self.stats_[col]["lower_bound"]
                upper_bound = self.stats_[col]["upper_bound"]
            else:
                # Compute on the fly
                q1 = df_clean[col].quantile(0.25)
                q3 = df_clean[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (factor * iqr)
                upper_bound = q3 + (factor * iqr)

            # Identify outliers
            is_outlier = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            outlier_count = is_outlier.sum()
            total_outliers_count += outlier_count

            if outlier_count > 0:
                if method == "drop":
                    rows_with_outliers |= is_outlier

                elif method == "mask":
                    df_clean.loc[is_outlier, col] = np.nan

                elif method == "clip":
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
