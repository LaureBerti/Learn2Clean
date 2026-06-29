"""
DataFrameAction — base class for all cleaning operations in Learn2Clean V3.

V3 improvements over V2:
- Pandera schema validation after every transform (optional, configurable)
- Raises DataValidationError on schema regression instead of silently passing bad data
- Cleaner logging via LoggingMixin
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union

import pandas as pd
import pandera as pa

from learn2clean_v3.types import Features, OptionalTarget

logger = logging.getLogger(__name__)


class DataValidationError(RuntimeError):
    """Raised when a transform produces data that violates the attached schema."""


class DataFrameAction(ABC):
    """
    Abstract base class for all DataFrame cleaning/preparation actions.

    Subclasses must implement ``transform(df)``.  The ``fit`` step is optional
    (column statistics computed on training split) and defaults to a no-op.

    Parameters
    ----------
    columns : list[str] | None
        Explicit list of columns to target. ``None`` means 'all eligible'.
    exclude_columns : list[str] | None
        Columns to exclude from targeting.
    dtype_filter : str | None
        Restrict to columns of this dtype family ('numeric', 'object', 'datetime').
    schema : pa.DataFrameSchema | None
        Pandera schema to validate the result after ``transform``.
        Validation is skipped if ``None``.
    strict_schema : bool
        If True, schema errors raise ``DataValidationError``.
        If False, they are logged as warnings and the result is still returned.
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
        dtype_filter: Optional[str] = None,
        schema: Optional[pa.DataFrameSchema] = None,
        strict_schema: bool = True,
    ) -> None:
        self._columns = columns
        self._exclude_columns = exclude_columns or []
        self._dtype_filter = dtype_filter
        self._schema = schema
        self._strict_schema = strict_schema
        self._fitted_columns: List[str] = []
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: Features, y: OptionalTarget = None) -> "DataFrameAction":
        """Learn column statistics from *df*.  Override in subclass as needed."""
        self._fitted_columns = self._select_columns(df)
        self._is_fitted = True
        return self

    @abstractmethod
    def transform(self, df: Features) -> Features:
        """Apply the cleaning/preparation operation.  Must return a new DataFrame."""

    def fit_transform(self, df: Features, y: OptionalTarget = None) -> Features:
        return self.fit(df, y).transform(df)

    def __call__(self, df: Features, y: OptionalTarget = None) -> Features:
        """Fit-transform shortcut with post-transform schema validation."""
        if not self._is_fitted:
            self.fit(df, y)

        result = self.transform(df)

        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"{self.__class__.__name__}.transform() must return a DataFrame, "
                f"got {type(result)}"
            )

        self._validate(result)
        return result

    def reset(self) -> None:
        """Reset fit state (called between RL episodes)."""
        self._fitted_columns = []
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Column selection helpers
    # ------------------------------------------------------------------

    def _select_columns(self, df: Features) -> List[str]:
        if self._columns is not None:
            candidates = [c for c in self._columns if c in df.columns]
        elif self._dtype_filter == "numeric":
            candidates = df.select_dtypes(include="number").columns.tolist()
        elif self._dtype_filter == "object":
            candidates = df.select_dtypes(include="object").columns.tolist()
        elif self._dtype_filter == "datetime":
            candidates = df.select_dtypes(include="datetime").columns.tolist()
        else:
            candidates = df.columns.tolist()

        return [c for c in candidates if c not in self._exclude_columns]

    @property
    def fitted_columns(self) -> List[str]:
        return list(self._fitted_columns)

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------

    def _validate(self, df: Features) -> None:
        if self._schema is None:
            return
        try:
            self._schema.validate(df, lazy=True)
        except pa.errors.SchemaErrors as exc:
            msg = (
                f"[{self.__class__.__name__}] post-transform schema validation failed:\n"
                f"{exc.failure_cases}"
            )
            if self._strict_schema:
                raise DataValidationError(msg) from exc
            logger.warning(msg)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        cols = self._columns or "all"
        return f"{self.__class__.__name__}(columns={cols})"

    @property
    def name(self) -> str:
        return self.__class__.__name__
