import math
from typing import Any, Self

import pandas as pd

from ...data_frame_action import DataFrameAction


class DecimalScaler(DataFrameAction):
    """
    Decimal Scaling normalization.

    This is a STATEFUL action: The scaling factor (10^j) must be computed during
    fit() on the training data and stored.

    Each numeric value x in a column is scaled by a power of 10:
        x' = x / 10^j
    where j is the smallest integer such that max(|x|) < 10^j.
    """

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._divisors: dict[str, float] = {}

    def fit(self, df: pd.DataFrame, y: Any = None) -> Self:
        """
        Fit the action by selecting numeric columns and calculating the
        appropriate divisor (10^j) for each column.
        """
        self._fitted_columns = self.select_columns(df, numeric_only=True)
        self._divisors = {}

        if not self._fitted_columns:
            self.log_warning(f"Fit skipped: No numeric columns found for scaling.")
            return self

        for col in self._fitted_columns:
            max_val = df[col].abs().max()
            self.log_debug(f"Scaling {col}: {max_val}")
            if pd.isna(max_val) or max_val <= 0:
                self.log_warning(
                    f"Fit: Cannot scale column '{col}'. Max absolute value is {max_val}."
                )
                continue

            try:
                log_val = math.log10(max_val)
                j = math.ceil(log_val)

                if log_val == j:
                    j += 1

                divisor = 10 ** int(j)
                self._divisors[col] = divisor
                self.log_debug(
                    f"Fit: Column '{col}'. max|x|={max_val:.4f}, j={j}, Divisor={divisor}"
                )
            except ValueError as e:
                self.log_error(f"Fit: Log calculation failed for column '{col}': {e}")
                continue

        self.log_info(
            f" Fit: Calculated scaling factors for {len(self._divisors)} columns."
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Decimal Scaling normalization using the fitted divisors.
        """
        df_copy = df.copy()

        cols_to_scale = self._fitted_columns

        if not cols_to_scale or not self._divisors:
            self.log_warning(
                f"Transform skipped: Action was not fitted or no numeric columns found."
            )
            return df_copy

        self.log_info(f"Applying scaling to {len(self._divisors)} columns.")

        for col, divisor in self._divisors.items():
            if col in df_copy.columns:
                df_copy[col] = df_copy[col] / divisor
            else:
                self.log_warning(
                    f"Transform: Column '{col}' was fitted but not found in the input DataFrame."
                )

        return df_copy
