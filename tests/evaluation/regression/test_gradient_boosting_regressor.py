import pytest
import numpy as np
import pandas as pd
from typing import Tuple
from learn2clean.evaluation.regression.gradient_boosting_regressor import (
    GradientBoostingRegressor,
)
from sklearn.ensemble import GradientBoostingRegressor as GRB


# --- Fixture for Simple Non-Linear Data (Same as used before) ---
@pytest.fixture(scope="module")
def synthetic_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Generates a simple non-linear dataset: y = x^2 + noise.
    Ideal for tree-based methods.
    """
    np.random.seed(42)
    n_samples = 100
    X = pd.DataFrame(
        {
            "feature_1": np.linspace(0, 10, n_samples),
            "feature_2_noise": np.random.uniform(10, 20, n_samples),
        }
    )
    # Target with non-linear feature_1**2 and low noise
    y = X["feature_1"] ** 2 + np.random.normal(0, 1.0, n_samples)
    y = pd.Series(y, name="target")

    # Split
    X_train, X_test = X.iloc[:80], X.iloc[80:]
    y_train, y_test = y.iloc[:80], y.iloc[80:]

    return X_train, y_train, X_test, y_test


# --- GBR Tests ---


def test_gbr_initialization():
    """Checks correct attribute initialization."""
    k_folds = 5
    gbr = GradientBoostingRegressor(k_folds=k_folds)
    assert gbr.k_folds == k_folds
    assert gbr.model is None
    assert gbr.cv_score is None


def test_gbr_fit_and_model_creation(synthetic_data):
    """Verifies that the fit method trains a GradientBoostingRegressor model."""
    X_train, y_train, _, _ = synthetic_data
    gbr = GradientBoostingRegressor(k_folds=3)

    gbr.fit(X_train, y_train)

    # 1. Model must be a trained GBR instance
    assert gbr.model is not None
    assert isinstance(gbr.model, GRB)


def test_gbr_cv_score_calculation(synthetic_data):
    """Checks that the cross-validated RMSE is calculated and is positive."""
    X_train, y_train, _, _ = synthetic_data
    gbr = GradientBoostingRegressor(k_folds=5)

    gbr.fit(X_train, y_train)

    # CV score must be calculated
    assert gbr.cv_score is not None
    assert isinstance(gbr.cv_score, float)
    assert gbr.cv_score > 0


# def test_gbr_evaluation_performance(synthetic_data):
#     """Checks that GBR achieves high R2 performance on non-linear data."""
#     X_train, y_train, X_test, y_test = synthetic_data
#     gbr = GradientBoostingRegressor(k_folds=3)
#
#     gbr.fit(X_train, y_train)
#
#     _, r2 = gbr.evaluate(X_test, y_test)
#
#     # R2 should be very high, proving the model captured the x^2 relationship
#     assert r2 > 0.90
