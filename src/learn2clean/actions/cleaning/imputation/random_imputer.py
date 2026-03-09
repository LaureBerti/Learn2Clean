"""
Random Imputation module for Learn2Clean.

Implements Random Sample Imputation.
Replaces missing values by a random selection from the observed values in the training set.
"""

from typing import Any, Self, ClassVar

import numpy as np
import pandas as pd

from ...data_frame_action import DataFrameAction


class RandomImputer(DataFrameAction):
    """
    Random Sample Imputation Action.

    Replaces missing values by sampling randomly from the valid observed values
    of the same column learned during `fit`.

    Pros: Preserves the mean and variance of the original distribution.
    Cons: Adds randomness; does not capture correlations between features.
    """

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "random_state": 42,
    }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        # Dictionary to store the pool of valid values for each column
        self._observed_values: dict[str, np.ndarray] = {}
        self._rng: np.random.Generator | None = None

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """
        Stores the valid (non-missing) values for each column to serve as a sampling pool.
        """
        self._fitted_columns = self.select_columns(df)

        if not self._fitted_columns:
            self.log_warning("RandomImputer: No columns selected for fitting.")
            return self

        # Initialize Random Generator
        seed = self.params.get("random_state", 42)
        self._rng = np.random.default_rng(seed)

        # Store observed values (drop NaNs)
        # OPTIMIZATION: If data is huge, we could store a reservoir sample instead of all values.
        for col in self._fitted_columns:
            valid_vals = df[col].dropna().to_numpy()
            if len(valid_vals) > 0:
                self._observed_values[col] = valid_vals
            else:
                self.log_warning(
                    f"RandomImputer: Column '{col}' is fully empty during fit. Cannot impute later."
                )

        self.log_debug(
            f"RandomImputer learned distributions for {len(self._observed_values)} columns."
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills NaNs by sampling from the fitted observed values.
        """
        if not self._fitted_columns or not self._observed_values:
            self.log_warning("RandomImputer: Not fitted or empty. Returning original.")
            return df

        # Schema validation
        missing = set(self._fitted_columns) - set(df.columns)
        if missing:
            raise ValueError(
                f"RandomImputer: Missing columns in transform input: {missing}"
            )

        df_res = df.copy()

        for col in self._fitted_columns:
            # Skip if column was empty during fit
            if col not in self._observed_values:
                continue

            # Identify missing locations
            mask_nan = df_res[col].isna()
            n_missing = mask_nan.sum()

            if n_missing > 0:
                # Sample from the learned pool
                fill_values = self._rng.choice(
                    self._observed_values[col], size=n_missing, replace=True
                )
                df_res.loc[mask_nan, col] = fill_values

        self.log_info(
            f"Random imputation applied to {len(self._fitted_columns)} columns."
        )
        return df_res
