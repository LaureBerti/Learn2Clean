"""
ParameterizedAction — V3 improvement #3.

Extends DataFrameAction so that each action exposes a structured
hyperparameter space.  The environment can pass continuous or discrete
parameter vectors alongside the action index, enabling a mixed
discrete-continuous action space (e.g., PPO with Dict action spaces).

Concrete examples shipped with V3:

    ParameterizedImputer  — strategy (mean/median/knn/mice) + n_neighbors
    ParameterizedOutlier  — method (iqr/zscore) + threshold
    ParameterizedScaler   — method (minmax/zscore/quantile)
"""

from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler

import pandera as pa

from learn2clean_v3.actions.data_frame_action import DataFrameAction
from learn2clean_v3.types import Features, OptionalTarget, ParamSpec


class ParameterizedAction(DataFrameAction):
    """
    A DataFrameAction whose behaviour is controlled by typed hyperparameters.

    Subclasses declare their parameter space via ``param_specs``.
    The env calls ``set_params(**kwargs)`` before ``__call__``.
    """

    @property
    @abstractmethod
    def param_specs(self) -> List[ParamSpec]:
        """Declare the hyperparameter space."""

    def set_params(self, **params: Any) -> "ParameterizedAction":
        """Apply hyperparameters; returns self for chaining."""
        valid = {s.name for s in self.param_specs}
        for k, v in params.items():
            if k not in valid:
                raise ValueError(f"{self.__class__.__name__}: unknown param '{k}'")
            setattr(self, f"_{k}", v)
        self.reset()   # force re-fit with new params
        return self

    def default_params(self) -> Dict[str, Any]:
        return {s.name: s.default for s in self.param_specs}

    def current_params(self) -> Dict[str, Any]:
        return {s.name: getattr(self, f"_{s.name}", s.default) for s in self.param_specs}

    def clone_with(self, **params: Any) -> "ParameterizedAction":
        """Return a deep-copy with updated parameters."""
        clone = deepcopy(self)
        return clone.set_params(**params)


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------

class ParameterizedImputer(ParameterizedAction):
    """
    Missing-value imputation with a selectable strategy.

    Hyperparameters
    ---------------
    strategy : str
        One of "mean", "median", "most_frequent", "knn".
    n_neighbors : int
        Number of neighbours for KNN imputation (ignored for others).
    """

    _STRATEGIES = ("mean", "median", "most_frequent", "knn")

    def __init__(
        self,
        strategy: str = "mean",
        n_neighbors: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(dtype_filter="numeric", **kwargs)
        self._strategy = strategy
        self._n_neighbors = n_neighbors
        self._imputer: Any = None

    @property
    def param_specs(self) -> List[ParamSpec]:
        return [
            ParamSpec(
                name="strategy",
                dtype="categorical",
                choices=list(self._STRATEGIES),
                default="mean",
            ),
            ParamSpec(
                name="n_neighbors",
                dtype="int",
                low=1,
                high=20,
                default=5,
            ),
        ]

    def fit(self, df: Features, y: OptionalTarget = None) -> "ParameterizedImputer":
        self._fitted_columns = self._select_columns(df)
        if self._strategy == "knn":
            self._imputer = KNNImputer(n_neighbors=self._n_neighbors)
        else:
            self._imputer = SimpleImputer(strategy=self._strategy)
        cols = [c for c in self._fitted_columns if c in df.columns]
        if cols:
            self._imputer.fit(df[cols])
        self._is_fitted = True
        return self

    def transform(self, df: Features) -> Features:
        result = df.copy()
        cols = [c for c in self._fitted_columns if c in df.columns]
        if cols and self._imputer is not None:
            result[cols] = self._imputer.transform(result[cols])
        return result


class ParameterizedOutlierCleaner(ParameterizedAction):
    """
    Outlier removal with a selectable detection method and threshold.

    Hyperparameters
    ---------------
    method : str
        "iqr" or "zscore".
    threshold : float
        IQR multiplier (IQR method) or z-score cut-off (zscore method).
    """

    def __init__(
        self,
        method: str = "iqr",
        threshold: float = 1.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(dtype_filter="numeric", **kwargs)
        self._method = method
        self._threshold = threshold
        self._bounds: Dict[str, tuple] = {}

    @property
    def param_specs(self) -> List[ParamSpec]:
        return [
            ParamSpec(
                name="method",
                dtype="categorical",
                choices=["iqr", "zscore"],
                default="iqr",
            ),
            ParamSpec(
                name="threshold",
                dtype="float",
                low=0.5,
                high=5.0,
                default=1.5,
            ),
        ]

    def fit(self, df: Features, y: OptionalTarget = None) -> "ParameterizedOutlierCleaner":
        self._fitted_columns = self._select_columns(df)
        self._bounds = {}
        for col in self._fitted_columns:
            series = df[col].dropna()
            if self._method == "iqr":
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                self._bounds[col] = (q1 - self._threshold * iqr, q3 + self._threshold * iqr)
            else:
                mean, std = series.mean(), series.std()
                self._bounds[col] = (mean - self._threshold * std, mean + self._threshold * std)
        self._is_fitted = True
        return self

    def transform(self, df: Features) -> Features:
        result = df.copy()
        mask = pd.Series(True, index=result.index)
        for col, (lo, hi) in self._bounds.items():
            if col in result.columns:
                mask &= result[col].between(lo, hi) | result[col].isna()
        return result[mask]


class ParameterizedScaler(ParameterizedAction):
    """
    Feature scaling with a selectable normalisation method.

    Hyperparameters
    ---------------
    method : str
        "minmax", "zscore", or "quantile".
    quantile_output : str
        Output distribution for quantile scaler: "uniform" or "normal".
    """

    def __init__(
        self,
        method: str = "minmax",
        quantile_output: str = "uniform",
        **kwargs: Any,
    ) -> None:
        super().__init__(dtype_filter="numeric", **kwargs)
        self._method = method
        self._quantile_output = quantile_output
        self._scaler: Any = None

    @property
    def param_specs(self) -> List[ParamSpec]:
        return [
            ParamSpec(
                name="method",
                dtype="categorical",
                choices=["minmax", "zscore", "quantile"],
                default="minmax",
            ),
            ParamSpec(
                name="quantile_output",
                dtype="categorical",
                choices=["uniform", "normal"],
                default="uniform",
            ),
        ]

    def fit(self, df: Features, y: OptionalTarget = None) -> "ParameterizedScaler":
        self._fitted_columns = self._select_columns(df)
        if self._method == "minmax":
            self._scaler = MinMaxScaler()
        elif self._method == "zscore":
            self._scaler = StandardScaler()
        else:
            self._scaler = QuantileTransformer(output_distribution=self._quantile_output)
        cols = [c for c in self._fitted_columns if c in df.columns]
        if cols:
            self._scaler.fit(df[cols])
        self._is_fitted = True
        return self

    def transform(self, df: Features) -> Features:
        result = df.copy()
        cols = [c for c in self._fitted_columns if c in df.columns]
        if cols and self._scaler is not None:
            result[cols] = self._scaler.transform(result[cols])
        return result


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class ParameterizedDeduplicator(ParameterizedAction):
    """
    Remove exact duplicate rows from the feature matrix.

    Hyperparameters
    ---------------
    keep : str
        Which occurrence to retain when a duplicate is found.
        "first" — keep the first occurrence (default).
        "last"  — keep the last occurrence.
    subset : str
        Column subset used to identify duplicates.
        "all" — all columns must match (exact row duplicate).
        "numeric" — only numeric columns are compared.
    """

    def __init__(
        self,
        keep: str = "first",
        subset: str = "all",
        **kwargs: Any,
    ) -> None:
        # dtype_filter="all" so the action sees the full DataFrame
        super().__init__(dtype_filter="all", **kwargs)
        self._keep = keep
        self._subset = subset

    @property
    def param_specs(self) -> List[ParamSpec]:
        return [
            ParamSpec(
                name="keep",
                dtype="categorical",
                choices=["first", "last"],
                default="first",
            ),
            ParamSpec(
                name="subset",
                dtype="categorical",
                choices=["all", "numeric"],
                default="all",
            ),
        ]

    def fit(self, df: Features, y: OptionalTarget = None) -> "ParameterizedDeduplicator":
        # Nothing to fit — deduplication is purely data-driven at transform time.
        self._is_fitted = True
        return self

    def transform(self, df: Features) -> Features:
        if self._subset == "numeric":
            num_cols: Optional[List[str]] = df.select_dtypes(include=[np.number]).columns.tolist()
            subset_arg = num_cols if num_cols else None
        else:
            subset_arg = None  # pandas default: all columns

        result = df.drop_duplicates(subset=subset_arg, keep=self._keep)
        return result.reset_index(drop=True)
