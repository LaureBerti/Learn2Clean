import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

from learn2clean.evaluation.regression.base_learner import BaseLearner


class LeastAbsoluteShrinkageAndSelectionOperator(BaseLearner):
    """
    Implements LASSO regression with built-in Cross-Validation (LassoCV)
    to automatically find the optimal alpha. Inherits the standard interface from BaseLearner.
    """

    def __init__(self, k_folds: int = 5) -> None:
        """
        Initializes the LASSO learner and defines the parameter search space.

        Calls BaseLearner.__init__(), setting self.model=None and self.scaler=None.
        """
        super().__init__()
        self.k_folds = k_folds
        # Define a broad range of alphas for the search
        self.alphas = np.logspace(-4, 0, 100)

    def fit(
        self, X: pd.DataFrame, y: pd.Series
    ) -> "LeastAbsoluteShrinkageAndSelectionOperator":
        """
        Fits the LassoCV model.

        Since LASSO uses an L1 penalty, feature scaling is often recommended,
        but LassoCV can often handle unscaled data if regularization is strong.
        No explicit scaling is applied here, keeping self.scaler = None.
        """
        # LassoCV automatically searches for the best alpha using cross-validation
        lasso_cv = LassoCV(
            alphas=self.alphas,
            cv=self.k_folds,
            random_state=42,
            n_jobs=-1,  # Use all processors
        ).fit(X, y)
        self.model = lasso_cv
        return self

    @property
    def best_alpha(self) -> float:
        """Returns the optimal alpha found by cross-validation."""
        if self.model and hasattr(self.model, "alpha_"):
            return self.model.alpha_
        return np.nan
