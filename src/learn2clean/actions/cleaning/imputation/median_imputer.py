"""
Median Imputation module for Learn2Clean.

Implements Median value imputation using Scikit-Learn's SimpleImputer.
Robust baseline for numerical data with outliers or skewed distributions.
"""

import inspect
from typing import Any, Self, ClassVar

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from ...data_frame_action import DataFrameAction


class MedianImputer(DataFrameAction):
    """
    Median Imputation Action.

    Replaces missing values with the median of each column.
    This strategy is robust to outliers, unlike MeanImputer.
    """

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "missing_values": np.nan,
        "strategy": "median",  # Enforced
        "fill_value": None,
        "keep_empty_features": False,
    }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.imputer: SimpleImputer | None = None

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """
        Learns the median of each numerical column.
        """
        # Median strategy only works for numeric data
        self._fitted_columns = self.select_columns(df, numeric_only=True)

        if not self._fitted_columns:
            self.log_warning("MedianImputer: No numerical columns found.")
            return self

        valid_params = self._filter_params(SimpleImputer)
        valid_params["strategy"] = "median"

        self.imputer = SimpleImputer(**valid_params)

        try:
            self.imputer.fit(df[self._fitted_columns])
            self.log_debug(
                f"MedianImputer fitted on {len(self._fitted_columns)} columns."
            )
        except Exception as e:
            self.log_error(f"MedianImputer fit failed: {e}")
            raise

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the learned medians to the DataFrame.
        """
        if not self._fitted_columns or self.imputer is None:
            self.log_warning("MedianImputer: Not fitted. Returning original.")
            return df

        missing = set(self._fitted_columns) - set(df.columns)
        if missing:
            raise ValueError(
                f"MedianImputer: Missing columns in transform input: {missing}"
            )

        df_res = df.copy()

        if not df_res[self._fitted_columns].isna().any().any():
            return df_res

        try:
            X_imputed = self.imputer.transform(df_res[self._fitted_columns])
            df_res.loc[:, self._fitted_columns] = X_imputed
            self.log_info(
                f"Median imputation applied to {len(self._fitted_columns)} columns."
            )
        except Exception as e:
            self.log_error(f"MedianImputer transform failed: {e}")
            raise

        return df_res

    def _filter_params(self, cls: type) -> dict[str, Any]:
        sig = inspect.signature(cls.__init__)
        valid = set(sig.parameters.keys()) - {"self", "kwargs"}
        return {k: v for k, v in self.params.items() if k in valid}
