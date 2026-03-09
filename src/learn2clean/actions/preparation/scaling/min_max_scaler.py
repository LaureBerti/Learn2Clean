from typing import Any, Self

import pandas as pd
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler

from ...data_frame_action import DataFrameAction


class MinMaxScaler(DataFrameAction):
    """
    Scales numeric columns of a DataFrame to a specified range using
    scikit-learn's MinMaxScaler.

    This is a STATEFUL action: Min/Max values are computed during fit()
    and applied during transform().

    Attributes
    ----------
    DEFAULT_PARAMS : dict[str, Any]
        Default parameters for the MinMaxScaler:
        - feature_range: tuple (min, max), default=(0, 1)
        - clip: bool, default=False
    """

    def __init__(self, **params: Any) -> None:
        """
        Initialize the MinMaxScaling action.
        """
        super().__init__(**params)
        self.scaler: SklearnMinMaxScaler = SklearnMinMaxScaler(**self.params)

    def fit(self, df: pd.DataFrame, y: Any = None) -> Self:
        """
        Fit the action by selecting numeric columns and computing min/max statistics
        using scikit-learn's MinMaxScaler.
        """
        self._fitted_columns = self.select_columns(df, numeric_only=True)

        if not self._fitted_columns:
            self.log_warning(f" Fit skipped: No numeric columns found for scaling.")
            return self

        self.scaler.fit(df[self._fitted_columns])

        self.log_info(
            f" Fit: Calculated Min/Max on {len(self._fitted_columns)} columns."
        )
        self.log_debug(f" Fitted Min values: {self.scaler.data_min_}")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply MinMax scaling to numeric columns using the fitted scaler.
        """
        df_copy = df.copy()

        cols_to_scale = self._fitted_columns

        if not cols_to_scale or not hasattr(self.scaler, "data_min_"):
            self.log_warning(
                f" Transform skipped: Action was not fitted or no numeric columns found."
            )
            return df_copy

        df_copy[cols_to_scale] = self.scaler.transform(df_copy[cols_to_scale])

        self.log_info(f" Scaling applied to {len(cols_to_scale)} columns.")

        return df_copy
