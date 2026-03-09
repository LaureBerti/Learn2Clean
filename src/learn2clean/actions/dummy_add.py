"""
DummyAdd module for Learn2Clean.

Defines the `DummyAdd` action, which is a simple transformation that
increments numeric columns by a fixed value. It inherits from `DataFrameAction`
and serves as a minimal example of a concrete action implementation.
"""

from typing import Self

import pandas as pd

from .data_frame_action import DataFrameAction
from ..types import Features, OptionalTarget


class DummyAdd(DataFrameAction):
    """
    DummyAdd action for incrementing numeric columns by a fixed value.

    This action adds a fixed increment to all numeric columns
    selected during the fitting phase. Since DummyAdd is a stateless
    action, the fit phase only determines the target columns.

    Example
    -------
    >>> action = DummyAdd(increment=2)
    >>> df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    >>> action.transform(df)
       a  b
    0  3  5
    1  4  6
    """

    name = "DummyAdd"

    DEFAULT_PARAMS = {"increment": 1}

    def fit(self, df: Features, y: OptionalTarget = None) -> Self:
        """
        Fits the action by identifying the numeric columns to transform.
        """
        self._fitted_columns = self.select_columns(df, numeric_only=True)
        self.log_info(
            f"Fit: Stored {len(self._fitted_columns)} columns "
            f"for increment ({self.params.get('increment', 1)}): {self._fitted_columns}"
        )
        self.log_debug(f"Fit: Parameters used: {self.params}")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the DummyAdd transformation to numeric columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with numeric columns incremented by the specified value.
        """
        df_copy = df.copy()
        numeric_columns = self._get_fitted_columns(df_copy, numeric_only=True)

        if not numeric_columns:
            self.log_warning(f"No numeric columns selected for transformation.")
            return df_copy

        increment: int = self.params.get("increment", self.DEFAULT_PARAMS["increment"])

        self.log_info(
            f"Applying increment of {increment} to {len(numeric_columns)} columns."
        )

        df_copy[numeric_columns] = df_copy[numeric_columns] + increment

        self.log_debug(f"Columns processed: {numeric_columns}")

        return df_copy
