"""
EM Imputation module for Learn2Clean.

Implements multivariate imputation using Scikit-Learn's IterativeImputer.
Refactored for robustness, type safety, and strict parameter handling.
"""

import inspect
from typing import Any, Self, ClassVar

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

from ...data_frame_action import DataFrameAction


class EMImputer(DataFrameAction):
    """
    EM-style Imputation Action.

    Uses a BayesianRidge estimator to iteratively impute missing values.
    Best suited for:
    - Data with linear correlations.
    - Numerical data with Gaussian-like distributions.
    - Scenarios requiring fast convergence.
    """

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "max_iter": 20,
        "tol": 1e-3,
        "imputation_order": "ascending",
        "random_state": 42,
        "verbose": 0,
    }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.imputer: IterativeImputer | None = None

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        self._fitted_columns = self.select_columns(df, numeric_only=True)

        if not self._fitted_columns:
            self.log_warning("EM: No numerical columns found.")
            return self

        # Secure parameter passing
        valid_params = self._filter_params(IterativeImputer)

        # Explicitly enforce BayesianRidge if not set, to guarantee "EM" behavior
        if "estimator" not in valid_params or valid_params["estimator"] is None:
            valid_params["estimator"] = BayesianRidge()

        self.imputer = IterativeImputer(**valid_params)

        X = df[self._fitted_columns].to_numpy(dtype=np.float64)
        try:
            self.imputer.fit(X)
            self.log_debug(f"EM fitted on {len(self._fitted_columns)} columns.")
        except Exception as e:
            self.log_error(f"EM fit failed: {e}")
            raise

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted_columns or self.imputer is None:
            self.log_warning("EM: Not fitted or no columns. Returning original.")
            return df

        # Schema validation
        missing = set(self._fitted_columns) - set(df.columns)
        if missing:
            raise ValueError(f"EM: Missing columns in transform input: {missing}")

        df_res = df.copy()
        X = df_res[self._fitted_columns].to_numpy(dtype=np.float64)

        # Optimization: Skip if no NaNs
        if not np.isnan(X).any():
            return df_res

        try:
            X_imputed = self.imputer.transform(X)
            df_res.loc[:, self._fitted_columns] = X_imputed
            self.log_info(f"EM imputed {len(self._fitted_columns)} columns.")
        except Exception as e:
            self.log_error(f"EM transform failed: {e}")
            raise

        return df_res

    def _filter_params(self, cls: type) -> dict[str, Any]:
        """Helper to filter self.params against a class constructor signature."""
        sig = inspect.signature(cls.__init__)
        valid = set(sig.parameters.keys()) - {"self", "kwargs"}
        return {k: v for k, v in self.params.items() if k in valid}
