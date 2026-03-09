import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.model_selection import cross_val_score

from learn2clean.evaluation.regression.base_learner import BaseLearner


class GradientBoostingRegressor(BaseLearner):
    """
    Implements Gradient Boosting Regressor (GBR) as a replacement for MARS,
    a robust non-linear method. Inherits the standard interface from BaseLearner.
    """

    def __init__(self, k_folds: int = 5) -> None:
        # Calls BaseLearner.__init__(), which now initializes self.model=None and self.scaler=None.
        super().__init__()
        self.k_folds = k_folds
        # Note: self.cv_score is specific to this class; BaseLearner handles standard evaluation (MSE, R2).
        self.cv_score: float | None = None

    def fit(
        self, X: pd.DataFrame, y: pd.Series
    ) -> "GradientBoostingRegressor":  # Added return type annotation
        """
        Fits the GBR model and calculates the cross-validated RMSE.

        Since GBR is tree-based, no feature scaling is applied (self.scaler remains None).
        """
        gbr_model = GBR(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        )
        # GBR fits directly on the raw X data
        gbr_model.fit(X, y)
        self.model = gbr_model

        # Calculate Cross-Validated Root Mean Squared Error (RMSE)
        neg_mse_scores = cross_val_score(
            gbr_model,
            X,
            y,
            scoring="neg_mean_squared_error",
            cv=self.k_folds,
            n_jobs=-1,
        )

        self.cv_score = np.sqrt(-neg_mse_scores).mean()
        return self
