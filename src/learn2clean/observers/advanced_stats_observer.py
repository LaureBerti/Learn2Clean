import numpy as np
import pandas as pd
from gymnasium import spaces
from scipy.stats import kurtosis, skew

from learn2clean.observers.base_observer import BaseObserver
from learn2clean.types import Features, OptionalTarget


class AdvancedStatsObserver(BaseObserver):
    """
    An Observer that extracts advanced statistical meta-features from the dataset.

    This observer goes beyond basic shape metrics and calculates statistical properties
    of the numeric features. This is particularly useful for agents that need to decide
    on preprocessing steps like Scaling, Log-Transformation, or Outlier Removal.

    Observation Vector Structure (Box(5,)):
        1. Mean Skewness: Indicates asymmetry. High values suggest LogTransform/Scaling needed.
        2. Mean Kurtosis: Indicates tail heaviness. High values suggest outliers (RobustScaler).
        3. Mean Correlation: Average absolute off-diagonal correlation. Indicates redundancy.
        4. Data Sparsity: Ratio of zero-values in the numeric data.
        5. Label Balance: Ratio of the minority class (if target is categorical), else 0.0.

    It returns a Gymnasium Dict space containing:
        - 'advanced_stats': Box(5,)
        - 'action_history': Box(n_actions,)
    """

    def get_observation_space(self) -> spaces.Dict:
        """
        Returns the Gymnasium Dict space definition.

        Returns:
            spaces.Dict: Keys 'advanced_stats' and 'action_history'.

        Raises:
            ValueError: If 'n_actions' is not set (inherited attribute).
        """
        if self.n_actions <= 0:
            raise ValueError(
                "n_actions must be positive. Ensure the Environment has "
                "initialized the observer with the correct action space size."
            )

        return spaces.Dict(
            {
                # [mean_skew, mean_kurtosis, mean_correlation, data_sparsity, label_balance]
                # Range is technically (-inf, inf) for skew/kurtosis, though usually bounded.
                "advanced_stats": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
                ),
                # Binary mask of executed actions
                "action_history": spaces.Box(
                    low=0, high=1, shape=(self.n_actions,), dtype=np.float32
                ),
            }
        )

    def observe(
        self,
        X: Features,
        y: OptionalTarget,
        action_history: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Calculates advanced statistics (Skew, Kurtosis, Corr) and formats history.

        Args:
            X: Current features DataFrame.
            y: Current target variable.
            action_history: Binary vector of past actions.

        Returns:
            dict[str, np.ndarray]: Composite observation.
        """
        # 1. Select numeric data for statistics
        X_num = X.select_dtypes(include=[np.number])

        if X_num.empty:
            # Fallback if no numeric columns exist
            stats = np.zeros(5, dtype=np.float32)
        else:
            # SKEWNESS
            # Measures asymmetry of the probability distribution.
            # nan_policy='omit' ignores NaNs during calculation.
            # np.nanmean handles cases where a column might return NaN skew (e.g., constant value).
            skew_vals = skew(X_num, axis=0, nan_policy="omit")
            avg_skew = np.nanmean(skew_vals) if len(skew_vals) > 0 else 0.0

            # KURTOSIS
            # Measures the "tailedness" of the probability distribution.
            kurt_vals = kurtosis(X_num, axis=0, nan_policy="omit")
            avg_kurt = np.nanmean(kurt_vals) if len(kurt_vals) > 0 else 0.0

            # SPARSITY
            # Ratio of zero values to total elements.
            sparsity = (X_num == 0).sum().sum() / X_num.size

            # CORRELATION
            # Average absolute correlation between features (excluding diagonal).
            if X_num.shape[1] > 1:
                corr_matrix = X_num.corr().abs()
                # Create a mask to select off-diagonal elements
                mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                # Calculate mean of valid correlations
                avg_corr = corr_matrix.values[mask].mean()
            else:
                avg_corr = 0.0

            # LABEL BALANCE (Only if y exists and is categorical/object)
            if y is not None and not pd.api.types.is_numeric_dtype(y):
                # Calculate the frequency of the minority class
                # value_counts(normalize=True) returns ratios summing to 1.
                counts = y.value_counts(normalize=True)
                balance_score = counts.min() if not counts.empty else 0.0
            else:
                # Default for Regression or Unsupervised tasks
                balance_score = 0.0

            # Replace any remaining NaNs with 0.0 (safety net) and cast to float32
            stats = np.array(
                [avg_skew, avg_kurt, avg_corr, sparsity, balance_score],
                dtype=np.float32,
            )
            stats = np.nan_to_num(stats, nan=0.0)

        # 2. History Handling
        if action_history is None:
            hist_vector = np.zeros(self.n_actions, dtype=np.float32)
        else:
            hist_vector = action_history.astype(np.float32)

        return {
            "advanced_stats": stats,
            "action_history": hist_vector,
        }
