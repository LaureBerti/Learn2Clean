from typing import Any, Self, ClassVar

import numpy as np
import pandas as pd

from learn2clean.actions.data_frame_action import DataFrameAction


class Log10Scaler(DataFrameAction):
    """
    Applies a Logarithm base 10 (Log10) transformation: log10(x + C)
    to reduce positive skewness in numeric columns.

    This is a STATEFUL action: The minimum shift value (C) is computed
    during fit() to handle non-positive values (0 or negative values)
    in the training data.

    Attributes:
        _shift_value (float): The constant C calculated during fit()
                              such that all x + C > 0.
        _fitted_columns (list[str]): Columns selected during fit.
    """

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "shift_epsilon": 1e-6,
    }

    def __init__(self, **params: Any) -> None:
        """
        Initialize the Log10Scaler action.
        """
        super().__init__(**params)
        self._shift_value: float = 0.0

    def fit(self, df: pd.DataFrame, y: Any = None) -> Self:
        """
        Fit the action by selecting numeric columns and calculating the
        shift constant C required to ensure all log arguments are positive.

        C = max(0, -min(X_train)) + epsilon
        """
        self._fitted_columns = self.select_columns(df, numeric_only=True)

        if not self._fitted_columns:
            self.log_warning(f"Fit skipped: No numeric columns found for scaling.")
            return self

        X_fit = df[self._fitted_columns]

        min_val = X_fit.min().min()

        if min_val <= 0:
            self._shift_value = abs(min_val) + self.params.get(
                "shift_epsilon", self.DEFAULT_PARAMS["shift_epsilon"]
            )
        else:
            self._shift_value = 0.0

        self.log_info(
            f"Fit: Calculated shift C = {self._shift_value:.6f} on {len(self._fitted_columns)} columns."
        )
        self.log_debug(f"Global minimum value detected: {min_val}")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Log10 transformation: log10(x + C) to numeric columns using the fitted shift C.
        """
        df_copy = df.copy()

        cols_to_scale: list[str] = self._fitted_columns

        if not cols_to_scale:
            self.log_warning(f"Transform skipped: Action was not fitted.")
            return df_copy

        shift_C = self._shift_value

        try:
            df_copy[cols_to_scale] = np.log10(df_copy[cols_to_scale] + shift_C)
        except Exception as e:
            self.log_error(f"Error during log10 transformation: {e}")
            return df_copy

        self.log_info(
            f"Log10 transformation applied to {len(cols_to_scale)} columns with C={shift_C:.6f}."
        )

        return df_copy
