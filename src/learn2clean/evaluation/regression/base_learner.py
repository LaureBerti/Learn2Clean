from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


class BaseLearner(BaseEstimator, RegressorMixin):
    """
    Base class for all regression models in the Learn2Clean pipeline.

    It establishes a standardized interface (fit, predict, evaluate)
    and includes support for a feature scaler (StandardScaler, etc.),
    which is essential for kernel-based methods like Kernel Ridge Regression.
    """

    def __init__(self) -> None:
        """
        Initializes the learner with a placeholder for the trained model
        and the feature scaler.
        """
        self.model: Any | None = None
        # Placeholder for a fitted scaler (e.g., StandardScaler) used by child classes
        # that require feature scaling (LASSO, KernelRidge, etc.).
        self.scaler: Any | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseLearner":
        """
        Fits the regression model to the data. Must be implemented by child classes.

        The child class should handle any necessary scaling (self.scaler.fit_transform)
        before training the model.
        """
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for the input data X.

        It automatically handles feature scaling if a scaler is present
        and adapts to statsmodels' OLS prediction mechanism.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")

        # --- Step 1: Handle Feature Scaling ---
        # If a scaler is present (set by a child class like KernelRidge),
        # the input features MUST be transformed before prediction.
        if self.scaler is not None:
            # Apply the transformation (DO NOT re-fit)
            X_processed = self.scaler.transform(X)
            # Reconvert to DataFrame to preserve column indexing/naming
            X_processed = pd.DataFrame(X_processed, index=X.index, columns=X.columns)
        else:
            X_processed = X

        # --- Step 2: Handle OLS Prediction (statsmodels) ---
        if isinstance(self.model, sm.regression.linear_model.RegressionResultsWrapper):
            # statsmodels OLS requires the constant (intercept) to be added manually
            X_with_const = sm.add_constant(X_processed, has_constant="add")
            return self.model.predict(X_with_const).values

        # --- Step 3: Standard scikit-learn Prediction ---
        return self.model.predict(X_processed)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[float, float]:
        """
        Calculates standard performance metrics (MSE and R2 score).

        The method relies on self.predict(), which handles feature scaling automatically.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): True target values.

        Returns:
            Tuple[float, float]: (Mean Squared Error, R2 Score)
        """
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2
