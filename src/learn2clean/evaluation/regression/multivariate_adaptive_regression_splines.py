# TODO Requires pyearth
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import cross_val_score
#
# from learn2clean.evaluation.regression.base_learner import BaseLearner
#
#
# class MultivariateAdaptiveRegressionSplines(BaseLearner):
#     """
#     Implements MARS (Multivariate Adaptive Regression Splines)
#     using the pyearth library.
#     """
#
#     def __init__(self, k_folds: int = 5) -> None:
#         super().__init__()
#         self.k_folds = k_folds
#         self.cv_score: float | None = None
#
#     def fit(self, X: pd.DataFrame, y: pd.Series):
#         # MARS model setup
#         mars_model = Earth(max_terms=10, max_degree=1)
#         mars_model.fit(X, y)
#         self.model = mars_model
#
#         # Calculate cross-validated performance (e.g., RMSE)
#         # Note: cross_val_score uses the base scikit-learn model interface
#         neg_mse_scores = cross_val_score(
#             mars_model,
#             X,
#             y,
#             scoring="neg_mean_squared_error",
#             cv=self.k_folds,
#             n_jobs=-1,
#         )
#         # Convert Negative MSE to RMSE (Mean of RMSE scores)
#         self.cv_score = np.sqrt(-neg_mse_scores).mean()
#         return self
