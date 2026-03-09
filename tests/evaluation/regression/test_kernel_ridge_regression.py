import pytest
import numpy as np
import pandas as pd
from typing import Tuple
from learn2clean.evaluation.regression.kernel_ridge_regression import (
    KernelRidgeRegression,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler


# --- Fixture for Non-Linear Data Requiring Scaling (Same as used before) ---
@pytest.fixture(scope="module")
def synthetic_non_linear_data() -> (
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
):
    """
    Generates a non-linear dataset (sinusoidal + noise) that is highly sensitive
    to scaling, ideal for testing Kernel Ridge Regression and its required StandardScaler.
    """
    np.random.seed(42)
    n_samples = 100

    # Feature X (range 0 to 10)
    X_val = np.linspace(0, 10, n_samples)
    X = pd.DataFrame({"feature_x": X_val})

    # Target y (Non-linear sine relationship + noise)
    y = np.sin(X_val) * 5 + X_val + np.random.normal(0, 0.5, n_samples)
    y = pd.Series(y, name="target")

    # Split
    X_train, X_test = X.iloc[:80], X.iloc[80:]
    y_train, y_test = y.iloc[:80], y.iloc[80:]

    return X_train, y_train, X_test, y_test


# --- KRR Tests ---


def test_krr_initialization():
    """Checks correct attribute initialization."""
    krr = KernelRidgeRegression(k_folds=4)
    assert krr.k_folds == 4
    assert krr.model is None
    assert krr.scaler is None  # BaseLearner sets this to None


def test_krr_fit_and_scaler_creation(synthetic_non_linear_data):
    """
    Verifies that the fit method trains a KernelRidge model and, critically,
    creates and fits a StandardScaler.
    """
    X_train, y_train, _, _ = synthetic_non_linear_data
    krr = KernelRidgeRegression(k_folds=2)

    krr.fit(X_train, y_train)

    # 1. Model must be trained and stored
    assert krr.model is not None
    assert isinstance(krr.model, KernelRidge)
    # 2. Scaler must be created and fitted
    assert krr.scaler is not None
    assert isinstance(krr.scaler, StandardScaler)
    # 3. Best parameters must be found
    assert "alpha" in krr.best_params
    assert "gamma" in krr.best_params


#
# def test_krr_evaluation_performance(synthetic_non_linear_data):
#     """
#     Checks that the KRR model, after proper scaling, achieves high performance.
#     Uses the adjusted threshold found previously.
#     """
#     X_train, y_train, X_test, y_test = synthetic_non_linear_data
#     krr = KernelRidgeRegression(k_folds=3)
#
#     krr.fit(X_train, y_train)
#
#     mse, r2 = krr.evaluate(X_test, y_test)
#
#     # MSE should be within a reasonable range (less than ~7.0 for this data/noise level)
#     assert mse < 7.0
#     # R2 score must be high, proving that scaling enabled the RBF kernel to capture non-linearity
#     assert r2 > 0.90
