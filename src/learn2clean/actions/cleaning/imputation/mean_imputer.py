"""
Mean Imputation module for Learn2Clean.

Implements Mean value imputation using Scikit-Learn's SimpleImputer.
Strictly for numerical data.
"""

import inspect
from typing import Any, Self, ClassVar

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from ...data_frame_action import DataFrameAction


class MeanImputer(DataFrameAction):
    """
    Mean Imputation Action.

    Replaces missing values with the mean of each column.
    This is the standard baseline for numerical imputation.
    """

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "missing_values": np.nan,
        "strategy": "mean",  # Enforced
        "fill_value": None,
        "keep_empty_features": False,
    }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.imputer: SimpleImputer | None = None

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """
        Learns the mean of each numerical column.
        """
        # Mean strategy only makes sense for numeric data
        self._fitted_columns = self.select_columns(df, numeric_only=True)

        if not self._fitted_columns:
            self.log_warning("MeanImputer: No numerical columns found.")
            return self

        valid_params = self._filter_params(SimpleImputer)
        # We strictly enforce 'mean' to match the class semantic
        valid_params["strategy"] = "mean"

        self.imputer = SimpleImputer(**valid_params)

        try:
            self.imputer.fit(df[self._fitted_columns])
            self.log_debug(
                f"MeanImputer fitted on {len(self._fitted_columns)} columns."
            )
        except Exception as e:
            self.log_error(f"MeanImputer fit failed: {e}")
            raise

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the learned means to the DataFrame.
        """
        if not self._fitted_columns or self.imputer is None:
            self.log_warning("MeanImputer: Not fitted. Returning original.")
            return df

        # Schema validation
        missing = set(self._fitted_columns) - set(df.columns)
        if missing:
            raise ValueError(
                f"MeanImputer: Missing columns in transform input: {missing}"
            )

        df_res = df.copy()

        # Optimization: Check if there is actually work to do
        if not df_res[self._fitted_columns].isna().any().any():
            return df_res

        try:
            X_imputed = self.imputer.transform(df_res[self._fitted_columns])
            df_res.loc[:, self._fitted_columns] = X_imputed
            self.log_info(
                f"Mean imputation applied to {len(self._fitted_columns)} columns."
            )
        except Exception as e:
            self.log_error(f"MeanImputer transform failed: {e}")
            raise

        return df_res

    def _filter_params(self, cls: type) -> dict[str, Any]:
        """Helper to filter self.params against a class constructor signature."""
        sig = inspect.signature(cls.__init__)
        valid = set(sig.parameters.keys()) - {"self", "kwargs"}
        return {k: v for k, v in self.params.items() if k in valid}
