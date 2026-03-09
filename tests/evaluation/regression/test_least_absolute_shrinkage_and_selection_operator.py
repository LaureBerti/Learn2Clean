import pytest
import numpy as np
import pandas as pd
from typing import Tuple
from learn2clean.evaluation.regression.least_absolute_shrinkage_and_selection_operator import (
    LeastAbsoluteShrinkageAndSelectionOperator,
)
from sklearn.linear_model import LassoCV

# Using the simple_linear_data fixture defined in test_ordinary_least_squares.py
# (Ensure pytest collects fixtures from shared files or redefine if necessary)


@pytest.fixture(scope="module")
def simple_linear_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Generates deterministic linear data."""
    np.random.seed(42)
    n_samples = 100
    X = pd.DataFrame(
        {
            "feature_1": np.random.rand(n_samples) * 10,
            "feature_2": np.random.rand(n_samples) * 5,
            "feature_3_noise": np.random.rand(n_samples),
        }
    )
    y = 2 * X["feature_1"] + 3 * X["feature_2"] + np.random.normal(0, 0.5, n_samples)
    y = pd.Series(y, name="target")
    X_train, X_test = X.iloc[:80], X.iloc[80:]
    y_train, y_test = y.iloc[:80], y.iloc[80:]
    return X_train, y_train, X_test, y_test


# --- LASSO Tests ---


def test_lasso_initialization():
    """Checks correct attribute initialization."""
    k_folds = 3
    lasso = LeastAbsoluteShrinkageAndSelectionOperator(k_folds=k_folds)
    assert lasso.k_folds == k_folds
    assert len(lasso.alphas) == 100
    assert lasso.model is None


def test_lasso_fit_and_model_creation(simple_linear_data):
    """Verifies that the fit method trains a LassoCV model."""
    X_train, y_train, _, _ = simple_linear_data
    lasso = LeastAbsoluteShrinkageAndSelectionOperator(k_folds=3)

    lasso.fit(X_train, y_train)

    # 1. Model must be a trained LassoCV instance
    assert lasso.model is not None
    assert isinstance(lasso.model, LassoCV)


def test_lasso_best_alpha_property(simple_linear_data):
    """Checks that the best_alpha property returns a positive float after fitting."""
    X_train, y_train, _, _ = simple_linear_data
    lasso = LeastAbsoluteShrinkageAndSelectionOperator(k_folds=3)
    lasso.fit(X_train, y_train)

    alpha = lasso.best_alpha

    # Alpha must be a float and greater than the lowest alpha in the search space (0.0001)
    assert isinstance(alpha, float)
    assert alpha > 0.0001


def test_lasso_evaluation_performance(simple_linear_data):
    """Checks that LASSO achieves high R2 performance."""
    X_train, y_train, X_test, y_test = simple_linear_data
    lasso = LeastAbsoluteShrinkageAndSelectionOperator(k_folds=3)

    lasso.fit(X_train, y_train)

    _, r2 = lasso.evaluate(X_test, y_test)

    # R2 should be high, proving regularization worked correctly
    assert r2 > 0.95
