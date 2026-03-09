"""
MF Imputation module for Learn2Clean.

Implements Most Frequent value imputation using Scikit-Learn's SimpleImputer.
Suitable for both categorical and numerical data.
"""

import inspect
from typing import Any, Self, ClassVar

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from ...data_frame_action import DataFrameAction


class MFImputer(DataFrameAction):
    """
    Most Frequent (Mode) Imputation Action.

    Replaces missing values with the most frequent value in each column.
    This is the preferred strategy for categorical/text data where mean/median
    are not applicable.
    """

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "missing_values": np.nan,
        "strategy": "most_frequent",
        "fill_value": None,
        "keep_empty_features": False,
    }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.imputer: SimpleImputer | None = None

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        # MF works on ALL types (numeric and object), so we don't force numeric_only=True
        # unless specified by the user in params.
        self._fitted_columns = self.select_columns(df)

        if not self._fitted_columns:
            self.log_warning("MF: No columns selected for fitting.")
            return self

        valid_params = self._filter_params(SimpleImputer)
        # Enforce 'most_frequent' strategy regardless of user input to match class name
        valid_params["strategy"] = "most_frequent"

        self.imputer = SimpleImputer(**valid_params)

        try:
            self.imputer.fit(df[self._fitted_columns])
            self.log_debug(f"MF fitted on {len(self._fitted_columns)} columns.")
        except Exception as e:
            self.log_error(f"MF fit failed: {e}")
            raise

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted_columns or self.imputer is None:
            self.log_warning("MF: Not fitted. Returning original.")
            return df

        # Schema validation
        missing = set(self._fitted_columns) - set(df.columns)
        if missing:
            raise ValueError(f"MF: Missing columns in transform input: {missing}")

        df_res = df.copy()

        # Optimization: check for NaNs before calling sklearn overhead
        if not df_res[self._fitted_columns].isna().any().any():
            return df_res

        try:
            # SimpleImputer returns a numpy array, we assign it back safely
            X_imputed = self.imputer.transform(df_res[self._fitted_columns])
            df_res.loc[:, self._fitted_columns] = X_imputed
            self.log_info(f"MF imputed {len(self._fitted_columns)} columns.")
        except Exception as e:
            self.log_error(f"MF transform failed: {e}")
            raise

        return df_res

    def _filter_params(self, cls: type) -> dict[str, Any]:
        """Helper to filter self.params against a class constructor signature."""
        sig = inspect.signature(cls.__init__)
        valid = set(sig.parameters.keys()) - {"self", "kwargs"}
        return {k: v for k, v in self.params.items() if k in valid}
