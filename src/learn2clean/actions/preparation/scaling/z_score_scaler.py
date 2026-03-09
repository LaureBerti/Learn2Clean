from typing import Any, Self

import pandas as pd
from sklearn.preprocessing import StandardScaler

from ...data_frame_action import DataFrameAction


class ZScoreScaler(DataFrameAction):
    """
    Z-score normalization using scikit-learn's StandardScaler.

    This transformation standardizes numeric columns by removing the mean
    and scaling to unit variance:

        z = (x - mean) / standard_deviation

    Parameters passed to this class are forwarded to
    `sklearn.preprocessing.StandardScaler` (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html),
    including:
        - with_mean: center the data before scaling
        - with_std: scale data to unit variance

    Columns to transform are determined by `self.columns` and `self.exclude`.
    If no columns are specified, all numeric columns are scaled.
    """

    def __init__(self, **params: Any) -> None:
        """
        Initialize ZScoreScaling action with optional StandardScaler parameters.

        Parameters
        ----------
        **params : Any
            Optional parameters to override DEFAULT_PARAMS.
        """
        super().__init__(**params)
        self.scaler: StandardScaler = StandardScaler(**self.params)

    def fit(self, df: pd.DataFrame, y: Any = None) -> Self:
        """
        Fit the action by selecting numeric columns and computing the mean and
        standard deviation statistics using scikit-learn's StandardScaler.
        """
        self._fitted_columns = self.select_columns(df, numeric_only=True)

        if not self._fitted_columns:
            self.log_warning(f" Fit skipped: No numeric columns found for scaling.")
            return self

        self.scaler.fit(df[self._fitted_columns])

        self.log_info(
            f" Fit: Calculated Mean/StdDev on {len(self._fitted_columns)} columns."
        )
        # StandardScaler stores the mean in 'mean_' attribute
        self.log_debug(f" Fitted Mean values: {self.scaler.mean_}")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Z-Score scaling to numeric columns using the fitted scaler.
        """
        df_copy = df.copy()

        cols_to_scale: list[str] = self._fitted_columns

        # Check if the scaler has been fitted (StandardScaler sets 'mean_' after fit)
        if not cols_to_scale or not hasattr(self.scaler, "mean_"):
            self.log_warning(
                f" Transform skipped: Action was not fitted or no numeric columns found."
            )
            return df_copy

        # Perform the scaling transformation
        df_copy[cols_to_scale] = self.scaler.transform(df_copy[cols_to_scale])

        self.log_info(f" Scaling applied to {len(cols_to_scale)} columns.")

        return df_copy
