from abc import ABC
from typing import Self, Protocol, TypeAlias, Any

import numpy as np
import pandas as pd

from learn2clean.actions.data_frame_action import DataFrameAction


class FeatureSelectorProtocol(Protocol):
    """Protocol defining the interface for Scikit-Learn feature selectors."""

    def fit(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series | None = None
    ) -> "FeatureSelectorProtocol": ...
    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray: ...
    def get_support(self, indices: bool = False) -> np.ndarray: ...
    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray: ...


SklearnSelector: TypeAlias = FeatureSelectorProtocol


class BaseFeatureSelector(DataFrameAction, ABC):
    """
    Base class for feature selection actions using Scikit-Learn selectors.

    This class handles the boilerplate of:
    1. Selecting numeric columns for the selector.
    2. Fitting the selector (supervised or unsupervised).
    3. Transforming the data while PRESERVING other columns (target, IDs, etc.).
    """

    _selector: SklearnSelector | None = None
    _selected_columns: list[str]

    def __init__(self, selector: SklearnSelector, **params: Any) -> None:
        super().__init__(**params)
        self._selector = selector

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """
        Fit the feature selector on the provided data.

        Args:
            df: The input DataFrame.
            y: The target Series (required for supervised feature selection).
        """
        if self._selector is None:
            raise RuntimeError("Selector (self.selector) must be initialized.")

        # 1. Select only numeric columns from the input DataFrame
        self._fitted_columns = self.select_columns(df, numeric_only=True)

        if not self._fitted_columns:
            self.log_warning("No numeric columns found for feature selection.")
            self._selected_columns = []
            return self

        X = df[self._fitted_columns]

        # 2. Initialize and fit Selector
        # Try-except block handles both supervised (needs y) and unsupervised (no y) selectors
        try:
            self._selector.fit(X, y)
        except TypeError:
            self._selector.fit(X)

        # 3. Determine the selected column names
        # Check if get_feature_names_out exists (modern sklearn) or use get_support
        if hasattr(self._selector, "get_feature_names_out"):
            self._selected_columns = list(
                self._selector.get_feature_names_out(self._fitted_columns)
            )
        else:
            mask = self._selector.get_support()
            self._selected_columns = [
                col for col, keep in zip(self._fitted_columns, mask) if keep
            ]

        self.log_debug(
            f"Selected {len(self._selected_columns)} features out of {len(self._fitted_columns)}"
        )

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the feature selection to the DataFrame.
        Preserves non-selected columns (Target, IDs, Categorical) by re-concatenating them.
        """
        if getattr(self, "_selected_columns", None) is None:
            raise RuntimeError("The action has not been fitted yet.")

        # 1. Identify columns to process
        cols_to_screen = self._fitted_columns

        # Safety check: if fit found nothing, we return original df
        if not cols_to_screen:
            return df.copy()

        # 2. Identify "Pass-through" columns (The Fix)
        # We drop the columns we are about to screen/filter.
        # Everything remaining (Target, Strings, etc.) must be kept.
        passthrough_cols = df.drop(columns=cols_to_screen, errors="ignore")

        # 3. Apply Selection
        # We ensure strict alignment by selecting exact columns
        X_in = df[cols_to_screen]
        X_new_array = self._selector.transform(X_in)

        # Create DataFrame with selected features
        df_selected = pd.DataFrame(
            X_new_array, columns=self._selected_columns, index=df.index
        )

        # 4. RECOMBINE
        # We merge the kept features with the pass-through columns
        result_df = pd.concat([df_selected, passthrough_cols], axis=1)

        return result_df
