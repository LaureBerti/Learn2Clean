from typing import Any

import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from learn2clean.evaluation.regression.base_learner import BaseLearner


class KernelRidgeRegression(BaseLearner):
    """
    Implements Kernel Ridge Regression.

    Uses GridSearchCV to tune the best hyperparameters (alpha and gamma)
    and applies StandardScaler as a necessary preprocessing step for
    kernel methods (RBF kernel).
    """

    def __init__(self, k_folds: int = 5) -> None:
        """Initializes the Kernel Ridge Learner."""
        # Calls BaseLearner.__init__(), which sets self.model=None and self.scaler=None.
        super().__init__()
        self.k_folds = k_folds
        self.best_params: dict[str, Any] = {}
        # Parameter grid for grid search (RBF kernel optimization)
        self.param_grid = {
            "alpha": [1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10, 100],
            "gamma": [1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10],
        }

    def fit(
        self, X: pd.DataFrame, y: pd.Series
    ) -> "KernelRidgeRegression":  # Added return type annotation
        """
        Fits the Kernel Ridge model using GridSearchCV on scaled features.

        It first fits the StandardScaler on X, then performs the grid search.
        The fitted StandardScaler is saved to self.scaler for consistent prediction.
        """
        # 1. Initialize and Fit StandardScaler (Crucial for Kernel Methods)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        # Convert back to DataFrame to preserve indices/column compatibility
        X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

        # 2. Setup and Run GridSearchCV on Scaled Data
        kr = KernelRidge(kernel="rbf")
        grid_search = GridSearchCV(
            estimator=kr,
            param_grid=self.param_grid,
            scoring="neg_mean_squared_error",
            cv=self.k_folds,
            n_jobs=-1,
        )
        grid_search.fit(X_scaled, y)  # Fitting on scaled data

        # 3. Store Results
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        return self
