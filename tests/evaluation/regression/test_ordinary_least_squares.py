import pytest
import numpy as np
import pandas as pd
from typing import Tuple
from learn2clean.evaluation.regression.ordinary_least_squares import (
    OrdinaryLeastSquares,
)
from statsmodels.regression.linear_model import RegressionResultsWrapper


# --- Fixture for Simple Linear Data ---
@pytest.fixture(scope="module")
def simple_linear_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Generates deterministic linear data: y = 2*x1 + 3*x2 + noise.
    Ideal for OLS and LASSO.
    """
    np.random.seed(42)
    n_samples = 100
    X = pd.DataFrame(
        {
            "feature_1": np.random.rand(n_samples) * 10,
            "feature_2": np.random.rand(n_samples) * 5,
            "feature_3_noise": np.random.rand(n_samples),  # Irrelevant feature
        }
    )
    # Target with low noise
    y = 2 * X["feature_1"] + 3 * X["feature_2"] + np.random.normal(0, 0.5, n_samples)
    y = pd.Series(y, name="target")

    # Split
    X_train, X_test = X.iloc[:80], X.iloc[80:]
    y_train, y_test = y.iloc[:80], y.iloc[80:]

    return X_train, y_train, X_test, y_test


# --- OLS Tests ---


def test_ols_initialization():
    """Checks correct attribute initialization."""
    ols = OrdinaryLeastSquares()
    assert ols.model is None
    assert ols.summary is None


def test_ols_fit_and_model_creation(simple_linear_data):
    """Verifies that the fit method trains a statsmodels OLS model."""
    X_train, y_train, _, _ = simple_linear_data
    ols = OrdinaryLeastSquares()

    ols.fit(X_train, y_train)

    # 1. Model must be a trained statsmodels result wrapper
    assert ols.model is not None
    assert isinstance(ols.model, RegressionResultsWrapper)
    # 2. Summary must be generated
    assert isinstance(ols.summary, str)


def test_ols_evaluation_performance(simple_linear_data):
    """Checks that OLS achieves high performance on simple linear data."""
    X_train, y_train, X_test, y_test = simple_linear_data
    ols = OrdinaryLeastSquares()

    ols.fit(X_train, y_train)

    mse, r2 = ols.evaluate(X_test, y_test)

    # R2 should be very high due to low noise
    assert r2 > 0.95
    # MSE should be very low
    assert mse < 1.0
