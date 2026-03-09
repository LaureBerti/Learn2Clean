import pandas as pd
import statsmodels.api as sm

from learn2clean.evaluation.regression.base_learner import BaseLearner


class OrdinaryLeastSquares(BaseLearner):
    """
    Implements Ordinary Least Squares (OLS) regression using the statsmodels library.
    Statsmodels is preferred for its detailed statistical summary outputs.
    """

    def __init__(self) -> None:
        """
        Initializes the OLS learner.
        Calls BaseLearner.__init__(), setting self.model=None and self.scaler=None.
        """
        super().__init__()
        # Attribute specific to OLS to store the full regression summary
        self.summary: str | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "OrdinaryLeastSquares":
        """
        Fits the OLS model using statsmodels.

        Since OLS is not sensitive to feature scale (coefficients adapt),
        no explicit scaling is performed (self.scaler remains None).

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Target values.

        Returns:
            "OrdinaryLeastSquares": The fitted instance (self).
        """
        # Statsmodels requires a constant (intercept) to be added manually
        X_with_const = sm.add_constant(X, has_constant="add")

        ols_model = sm.OLS(y, X_with_const).fit()
        self.model = ols_model
        # Store the statistical summary for detailed output
        self.summary = ols_model.summary().as_text()
        return self
