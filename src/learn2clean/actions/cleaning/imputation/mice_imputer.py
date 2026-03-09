"""
MICE Imputation module for Learn2Clean.

Implements MICE using IterativeImputer with an ExtraTreesRegressor.
This approach is non-parametric and can model complex, non-linear relationships.
"""

import inspect
from typing import Any, Self, ClassVar

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

# IterativeImputer is still experimental in sklearn
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from ...data_frame_action import DataFrameAction


class MICEImputer(DataFrameAction):
    """
    MICE (Non-Linear) Imputation Action.

    Uses an ExtraTreesRegressor to iteratively impute missing values.
    Best suited for:
    - Complex datasets with non-linear interactions.
    - Data where assumptions of normality do not hold.
    - Scenarios where accuracy is prioritized over speed.
    """

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "max_iter": 50,  # Trees converge faster but are costlier per iter
        "tol": 1e-3,
        "imputation_order": "descending",  # Impute most complete cols first (often helps trees)
        "random_state": 42,
        "n_nearest_features": None,
        "verbose": 0,
    }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.imputer: IterativeImputer | None = None

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        self._fitted_columns = self.select_columns(df, numeric_only=True)

        if not self._fitted_columns:
            self.log_warning("MICE: No numerical columns found.")
            return self

        valid_params = self._filter_params(IterativeImputer)

        # The Key Difference: Default to ExtraTreesRegressor
        if "estimator" not in valid_params or valid_params["estimator"] is None:
            # fast and robust tree ensemble
            valid_params["estimator"] = ExtraTreesRegressor(
                n_estimators=50,
                random_state=self.params.get("random_state", 42),
                n_jobs=-1,
            )

        self.imputer = IterativeImputer(**valid_params)

        X = df[self._fitted_columns].to_numpy(dtype=np.float64)
        try:
            self.imputer.fit(X)

            if self.imputer.n_iter_ == self.params.get("max_iter", 10):
                self.log_warning(
                    f"MICE reached max_iter ({self.imputer.n_iter_}). "
                    "Convergence might not be optimal. Consider increasing max_iter."
                )
            else:
                self.log_debug(
                    f"MICE converged in {self.imputer.n_iter_} iterations using "
                    f"{type(valid_params['estimator']).__name__}."
                )

        except Exception as e:
            self.log_error(f"MICE fit failed: {e}")
            raise

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted_columns or self.imputer is None:
            self.log_warning("MICE: Not fitted or no columns. Returning original.")
            return df

        missing = set(self._fitted_columns) - set(df.columns)
        if missing:
            raise ValueError(f"MICE: Missing columns in transform input: {missing}")

        df_res = df.copy()
        X = df_res[self._fitted_columns].to_numpy(dtype=np.float64)

        if not np.isnan(X).any():
            return df_res

        try:
            X_imputed = self.imputer.transform(X)
            df_res.loc[:, self._fitted_columns] = X_imputed

            self.log_info(
                f"MICE imputed values in {len(self._fitted_columns)} columns."
            )

        except Exception as e:
            self.log_error(f"MICE transform failed: {e}")
            raise

        return df_res

    def _filter_params(self, cls: type) -> dict[str, Any]:
        """Helper to filter self.params against a class constructor signature."""
        sig = inspect.signature(cls.__init__)
        valid_keys = set(sig.parameters.keys()) - {"self", "kwargs"}
        return {k: v for k, v in self.params.items() if k in valid_keys}
