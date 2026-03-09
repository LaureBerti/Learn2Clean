"""
DataFrameAction module for Learn2Clean.

Defines the abstract base class `DataFrameAction`, which serves as the
foundation for all DataFrame transformations.
Subclasses should implement their specific transformation logic by overriding the `tranform()` method.
This architecture ensures compatibility with:
1. The Scikit-learn API model (fit/transform for stateful actions).
2. Functional pipeline execution (via the __call__ method).
3. Configuration management systems (like Hydra).
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Self, ClassVar

import pandas as pd

from learn2clean.types import Features, OptionalTarget
from learn2clean.utils.action_logging_mixin import ActionLoggingMixin


class DataFrameAction(ABC, ActionLoggingMixin):
    """
    Abstract base class for all transformations applied to pandas DataFrames
    within the Learn2Clean pipeline.

    It provides a consistent, stateful interface (fit/transform) and manages
    essential metadata, configuration, and parameter resolution.
    """

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {}

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(self, **params: Any) -> None:
        """
        Initialize the DataFrameAction, manage parameters, and calculate module metadata.

        Initializes the logging system by calling the LoggingMixin constructor.

        Parameters
        ----------
        **params : Any
            User-specified parameters. These can include action configuration
            (`columns`, `exclude`, `name`) and parameters specific to the
            transformation logic (which are stored in `self.params`).
        """
        super().__init__()

        # --- Configuration Attributes ---
        self.name: str = params.pop("name", self.__class__.__name__)
        self.columns: list[str] | None = params.pop("columns", None)
        self.exclude: list[str] | None = params.pop("exclude", None)

        # Internal state for Scikit-learn API compliance
        self._fitted_columns: list[str] = []

        # --- Parameter Resolution ---
        # Merge class defaults with user-supplied parameters, allowing user input to override.
        self.params: dict[str, Any] = {**self.__class__.DEFAULT_PARAMS, **params}

        # --- Module Metadata Calculation (Logical Path) ---
        # Calculates the path segments (e.g., 'preparation.scaling') for logging/config.
        parts = self.__module__.split(".")
        self.logical_path: str = ".".join(parts[2 : len(parts) - 1])

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Allows calling the action instance directly as a function (e.g., `action(df)`).

        This method serves as the entry point for pipeline execution, wrapping
        `self.transform()` with necessary logging, timing, and error handling.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to be transformed.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame returned by `self.transform()`.
        """
        start_time = time.time()
        self.log_info(f"START transform")
        try:
            # Executes the core transformation logic implemented in the subclass
            result_df = self.transform(df)
            duration = time.time() - start_time
            self.log_info(f"END in {duration:.4f} seconds.")
            return result_df

        except Exception as e:
            self.log_error(f"FATAL: failed during transformation: {e}")
            # Re-raises the exception to ensure the pipeline stops
            raise

    # -------------------------------------------------------------------------
    # Sklearn-like API (State Management)
    # -------------------------------------------------------------------------

    def fit(self, df: Features, y: OptionalTarget = None) -> Self:
        """
        Fits the action on the DataFrame, typically used to compute and store state.

        Default behavior:
        - Only selects the target columns based on the DataFrame structure and
          user configuration (self.columns/self.exclude).
        - Stores the result in `self._fitted_columns`.
        - Subclasses must override this method to compute statistics (e.g., means,
          quantiles, thresholds) and store them for use during `transform`.

        Parameters
        ----------
        df : pd.DataFrame
            The training data used to compute the state.
        y : pd.Series, optional
            Target values (ignored by default, but required for Sklearn API compatibility).

        Returns
        -------
        self : DataFrameAction
            The fitted action instance.
        """
        self._fitted_columns = self.select_columns(df)
        self.log_debug(
            f"Fitted with parameters: {self.params}. "
            f"Selected columns (default fit): {self._fitted_columns}"
        )
        return self

    # -------------------------------------------------------------------------
    # Abstract method: subclasses must implement the core transformation
    # -------------------------------------------------------------------------
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the specific transformation logic to the DataFrame.

        This is a required abstract method; subclasses must implement this
        method, using the state stored during the `fit` call if the action is stateful.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to be transformed (usually test or unseen data).

        Returns
        -------
        pd.DataFrame
            The resulting transformed DataFrame.
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Column selection logic
    # -------------------------------------------------------------------------
    def select_columns(
        self,
        df: pd.DataFrame,
        numeric_only: bool = False,
        categorical_only: bool = False,
    ) -> list[str]:
        """
        Determines the final set of columns to be targeted by the action.

        Selection is based on three hierarchical criteria:
        1. Explicit inclusion: Use `self.columns` if provided.
        2. Explicit exclusion: Columns in `self.exclude` are always removed.
        3. Data type filtering: Filter by `numeric_only` or `categorical_only`.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame used for checking column existence and data types.
        numeric_only : bool, optional
            If True, selects only columns with numeric data types. Defaults to False.
        categorical_only : bool, optional
            If True, selects only columns that are NOT numeric. Defaults to False.

        Returns
        -------
        list[str]
            List of column names selected for transformation.
        """
        if self.columns is not None:
            inclusion_set = set(self.columns)
        else:
            inclusion_set = set(df.columns)

        exclusion_set = set(self.exclude or [])
        cols: list[str] = []
        for c in df.columns:
            if self.columns is not None and c not in inclusion_set:
                continue
            if c in exclusion_set:
                continue
            if numeric_only and not pd.api.types.is_numeric_dtype(df[c]):
                continue
            if categorical_only and pd.api.types.is_numeric_dtype(df[c]):
                continue
            cols.append(c)

        return cols

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _get_fitted_columns(
        self,
        df: pd.DataFrame,
        numeric_only: bool = False,
        categorical_only: bool = False,
    ) -> list[str]:
        """
        Retrieves the list of columns the action should operate on during transformation.

        It prioritizes the columns stored during `fit` (`self._fitted_columns`)
        to maintain state consistency. If `fit` was not called (fitless mode),
        it falls back to selecting columns dynamically based on the input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame (used only as a fallback for fitless mode).
        numeric_only : bool, optional
            Fallback filter: used if `_fitted_columns` is empty.
        categorical_only : bool, optional
            Fallback filter: used if `_fitted_columns` is empty.

        Returns
        -------
        list[str]
            List of column names to be transformed.
        """
        # Prioritize _fitted_columns (stateful operation)
        if self._fitted_columns:
            return self._fitted_columns

        # Fallback for fitless mode (dynamic selection)
        return self.select_columns(df, numeric_only, categorical_only)

    def __repr__(self) -> str:
        """Return a readable string representation of the action instance."""
        cls = self.__class__.__name__
        return (
            f"{cls}(name={self.name!r}, "
            f"path={self.logical_path!r}, "
            f"columns={self.columns}, exclude={self.exclude}, "
            f"params={self.params})"
        )
