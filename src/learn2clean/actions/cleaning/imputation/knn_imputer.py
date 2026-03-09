"""
KNN Imputation module for Learn2Clean.

Implements K-Nearest Neighbors imputation using Scikit-Learn's KNNImputer.
Strictly for numerical data (distance-based).
"""

import inspect
from typing import Any, Self, ClassVar

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer as SklearnKNNImputer

from ...data_frame_action import DataFrameAction


class KNNImputer(DataFrameAction):
    """
    K-Nearest Neighbors (KNN) Imputation Action.

    Imputes missing values using the weighted or unweighted mean of the k-nearest
    neighbors found in the training set.
    """

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "n_neighbors": 5,
        "weights": "uniform",  # or 'distance'
        "metric": "nan_euclidean",
        "keep_empty_features": False,
    }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.imputer: SklearnKNNImputer | None = None

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        # KNN requires numerical data to compute distances
        self._fitted_columns = self.select_columns(df, numeric_only=True)

        if not self._fitted_columns:
            self.log_warning("KNN: No numerical columns found.")
            return self

        valid_params = self._filter_params(SklearnKNNImputer)
        self.imputer = SklearnKNNImputer(**valid_params)

        X = df[self._fitted_columns].to_numpy(dtype=np.float64)
        try:
            self.imputer.fit(X)
            self.log_debug(f"KNN fitted on {len(self._fitted_columns)} columns.")
        except Exception as e:
            self.log_error(f"KNN fit failed: {e}")
            raise

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted_columns or self.imputer is None:
            self.log_warning("KNN: Not fitted. Returning original.")
            return df

        missing = set(self._fitted_columns) - set(df.columns)
        if missing:
            raise ValueError(f"KNN: Missing columns in transform input: {missing}")

        df_res = df.copy()
        X = df_res[self._fitted_columns].to_numpy(dtype=np.float64)

        if not np.isnan(X).any():
            return df_res

        try:
            X_imputed = self.imputer.transform(X)
            df_res.loc[:, self._fitted_columns] = X_imputed
            self.log_info(f"KNN imputed {len(self._fitted_columns)} columns.")
        except Exception as e:
            self.log_error(f"KNN transform failed: {e}")
            raise

        return df_res

    def _filter_params(self, cls: type) -> dict[str, Any]:
        """Helper to filter self.params against a class constructor signature."""
        sig = inspect.signature(cls.__init__)
        valid = set(sig.parameters.keys()) - {"self", "kwargs"}
        return {k: v for k, v in self.params.items() if k in valid}
