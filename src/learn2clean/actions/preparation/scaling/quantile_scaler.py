from typing import Any, Self

import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from ...data_frame_action import DataFrameAction


class QuantileScaler(DataFrameAction):
    """
    Quantile normalization using scikit-learn's QuantileTransformer.

    This transformation maps the data to a uniform or normal distribution
    (depending on 'output_distribution'). It estimates the quantiles of
    the training data and maps new data points to these quantiles.

    Parameters passed to this class are forwarded to
    `sklearn.preprocessing.QuantileTransformer` (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html),
    including:
        - n_quantiles: number of quantiles to be used, default=1000
        - output_distribution: 'uniform' or 'normal', default='uniform'
        - random_state: determines random number generation for stochastic aspects

    Columns to transform are determined by `self.columns` and `self.exclude`.
    If no columns are specified, all numeric columns are scaled.
    """

    def __init__(self, **params: Any) -> None:
        """
        Initialize QuantileScaling action with optional QuantileTransformer parameters.

        Parameters
        ----------
        **params : Any
            Optional parameters to override DEFAULT_PARAMS.
        """
        super().__init__(**params)
        # Use QuantileTransformer for Quantile Scaling
        self.scaler: QuantileTransformer = QuantileTransformer(**self.params)

    def fit(self, df: pd.DataFrame, y: Any = None) -> Self:
        """
        Fit the action by selecting numeric columns and computing the quantiles
        using scikit-learn's QuantileTransformer.
        """
        self._fitted_columns = self.select_columns(df, numeric_only=True)

        if not self._fitted_columns:
            self.log_warning(f"Fit skipped: No numeric columns found for scaling.")
            return self

        n_samples = len(df)
        current_n_quantiles = self.scaler.get_params().get("n_quantiles")
        new_n_quantiles = min(current_n_quantiles, n_samples)
        if new_n_quantiles != current_n_quantiles:
            self.log_debug(
                f"Adjusting n_quantiles from {current_n_quantiles} to {new_n_quantiles} "
                f"based on {n_samples} samples."
            )
            self.scaler.set_params(n_quantiles=new_n_quantiles)

        # The QuantileTransformer requires data as a NumPy array for fitting
        self.scaler.fit(df[self._fitted_columns].values)

        self.log_info(
            f"Fit: Calculated quantiles on {len(self._fitted_columns)} columns."
        )
        # QuantileTransformer stores the number of samples seen in 'n_samples_seen_'
        self.log_debug(f"Fitted number of quantiles: {self.scaler.n_quantiles_}")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Quantile scaling to numeric columns using the fitted transformer.
        """
        df_copy = df.copy()

        cols_to_scale: list[str] = self._fitted_columns

        # Check if the transformer has been fitted (QuantileTransformer sets 'quantiles_' after fit)
        if not cols_to_scale or not hasattr(self.scaler, "quantiles_"):
            self.log_warning(
                f"Transform skipped: Action was not fitted or no numeric columns found."
            )
            return df_copy

        # Perform the scaling transformation using NumPy array
        scaled_data = self.scaler.transform(df_copy[cols_to_scale].values)

        # Assign the transformed NumPy array back to the DataFrame
        df_copy[cols_to_scale] = pd.DataFrame(
            scaled_data, index=df_copy.index, columns=cols_to_scale
        )

        self.log_info(f"Scaling applied to {len(cols_to_scale)} columns.")

        return df_copy
