import numpy as np

from learn2clean.actions import EMImputer
from tests.utils.actions.assert_action_pipeline import assert_action_pipeline


def test_em_imputation_learns_linear_correlation() -> None:
    """
    Transductive test: The imputer learns and transforms on the same data.
    Simple case: B = 2 * A.
    """
    input_data = {
        "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        "B": [2.0, 4.0, np.nan, 8.0, 10.0],  # Missing at index 2
        "C_cat": ["keep", "me", "safe", "please", "!"],
    }

    expected_data = {
        "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        "B": [2.0, 4.0, 6.0, 8.0, 10.0],  # 3.0 * 2 = 6.0
        "C_cat": ["keep", "me", "safe", "please", "!"],
    }

    assert_action_pipeline(
        EMImputer(random_state=42, max_iter=50),
        transform_data=input_data,
        expected_result=expected_data,
        fit_data=input_data,  # Fit on self
        numeric_allclose=True,
        atol=0.1,
    )


def test_em_imputation_generalization_split() -> None:
    """
    Generalization test (Inductive):
    We train on one dataset (Fit) and apply to a NEW dataset (Transform).
    This is crucial to validate that the model has correctly 'learned' the relationship B = A + 10.
    """
    # Train: Relationship B = A + 10
    fit_data = {
        "A": [1.0, 2.0, 3.0, 10.0],
        "B": [11.0, 12.0, 13.0, 20.0],
    }

    # Test: New data with missing B
    transform_data = {
        "A": [5.0],
        "B": [np.nan],  # Expected: 15.0
    }

    expected_data = {
        "A": [5.0],
        "B": [15.0],
    }

    assert_action_pipeline(
        EMImputer(random_state=42),
        transform_data=transform_data,
        expected_result=expected_data,
        fit_data=fit_data,  # Fit on separate data
        numeric_allclose=True,
        atol=0.1,
    )


def test_em_imputation_multivariate_dependency() -> None:
    """
    Multivariate test: C depends on A and B (C = A + B).
    EM must use both present columns to infer the third one.
    """
    input_data = {
        "A": [1.0, 2.0, 3.0, 4.0],
        "B": [1.0, 1.0, 1.0, 1.0],
        "C": [2.0, 3.0, np.nan, 5.0],  # Missing, should be 3 + 1 = 4
    }

    expected_data = {
        "A": [1.0, 2.0, 3.0, 4.0],
        "B": [1.0, 1.0, 1.0, 1.0],
        "C": [2.0, 3.0, 4.0, 5.0],
    }

    assert_action_pipeline(
        EMImputer(random_state=42),
        transform_data=input_data,
        expected_result=expected_data,
        fit_data=input_data,
        numeric_allclose=True,
        atol=0.1,
    )


def test_em_no_missing_values_pass_through() -> None:
    """
    Idempotency test: If no data is missing, the data must not be modified.
    Verifies the integrity of clean data.
    """
    data = {
        "A": [1.0, 2.0, 3.0],
        "B": [4.0, 5.0, 6.0],
        "C_text": ["x", "y", "z"],
    }

    assert_action_pipeline(
        EMImputer(),
        transform_data=data,
        expected_result=data,  # Must be identical to input
        fit_data=data,
        numeric_allclose=True,
    )


def test_em_ignores_non_numeric_columns() -> None:
    """
    Safety test: Verifies that the imputer silently ignores non-numeric columns
    and does not crash if there is nothing to do.
    """
    data = {
        "Category": ["A", "B", "C"],
        "Label": ["Yes", "No", np.nan],  # Missing but textual -> Must be ignored by EM
    }

    # Expected result is identical to input because EM only touches numerics
    assert_action_pipeline(
        EMImputer(),
        transform_data=data,
        expected_result=data,
        fit_data=data,
    )
