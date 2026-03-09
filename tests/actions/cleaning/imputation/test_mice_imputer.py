import numpy as np

from learn2clean.actions import MICEImputer
from tests.utils.actions.assert_action_pipeline import assert_action_pipeline


def test_mice_imputation_learns_non_linear_pattern() -> None:
    """
    Transductive test: Validates that MICE (Tree-based) captures non-linear patterns.
    Pattern: Step function. If A <= 5, B = 10; If A > 5, B = 100.
    Linear EM would fail to capture this sharp jump accurately.
    """
    input_data = {
        "A": [1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 9.0],
        "B": [
            10.0,
            10.0,
            10.0,
            10.0,
            100.0,
            np.nan,
            100.0,
        ],  # Missing at index 5 (A=8.0 -> Should be 100)
        "C_cat": ["a", "b", "c", "d", "e", "f", "g"],
    }

    expected_data = {
        "A": [1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 9.0],
        "B": [10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0],  # Imputed 100.0
        "C_cat": ["a", "b", "c", "d", "e", "f", "g"],
    }

    assert_action_pipeline(
        MICEImputer(random_state=42, max_iter=20, n_estimators=50),
        transform_data=input_data,
        expected_result=expected_data,
        fit_data=input_data,
        numeric_allclose=True,
        atol=0.5,
    )


def test_mice_imputation_generalization_split() -> None:
    """
    Generalization test (Inductive):
    Trains on a dataset representing a conditional logic and applies it to new data.
    Rule learned: If A is close to 1, B=0; If A is close to 10, B=1.
    """
    # Train: Binary-like relationship
    fit_data = {
        "A": [1.0, 1.1, 1.2, 9.8, 9.9, 10.0],
        "B": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    }

    # Test: New data points clearly falling into the two clusters
    transform_data = {
        "A": [1.05, 9.95],
        "B": [np.nan, np.nan],  # Expected: ~0.0 and ~1.0
    }

    expected_data = {
        "A": [1.05, 9.95],
        "B": [0.0, 1.0],
    }

    assert_action_pipeline(
        MICEImputer(random_state=42, n_estimators=50),
        transform_data=transform_data,
        expected_result=expected_data,
        fit_data=fit_data,
        numeric_allclose=True,
        atol=0.1,
    )


def test_mice_imputation_multivariate_interaction() -> None:
    """
    Multivariate interaction test: B depends on A, and C depends on the interaction of A and B.
    Example: XOR-like or simple interaction. Here: C is high only if A and B are high.
    """
    input_data = {
        "A": [0, 0, 10, 10, 0, 10],
        "B": [0, 10, 0, 10, 0, 10],
        # C is 100 only if A=10 AND B=10, else 0
        "C": [0, 0, 0, 100, 0, np.nan],  # Last row: A=10, B=10 -> C should be 100
    }

    expected_data = {
        "A": [0, 0, 10, 10, 0, 10],
        "B": [0, 10, 0, 10, 0, 10],
        "C": [0, 0, 0, 100, 0, 100],
    }

    assert_action_pipeline(
        MICEImputer(random_state=42, n_estimators=50),
        transform_data=input_data,
        expected_result=expected_data,
        fit_data=input_data,
        numeric_allclose=True,
        atol=1.0,
    )


def test_mice_no_missing_values_pass_through() -> None:
    """
    Idempotency test: Ensures that MICE leaves complete datasets untouched.
    """
    data = {
        "A": [1.0, 5.0, 10.0],
        "B": [0.0, 50.0, 100.0],
        "C_text": ["cat", "dog", "bird"],
    }

    assert_action_pipeline(
        MICEImputer(),
        transform_data=data,
        expected_result=data,
        fit_data=data,
        numeric_allclose=True,
    )


def test_mice_ignores_non_numeric_columns() -> None:
    """
    Safety test: Verifies that MICE ignores non-numeric columns and does not raise errors
    when processing mixed types.
    """
    data = {
        "ID": ["u1", "u2", "u3"],
        "Value": [
            10.0,
            20.0,
            np.nan,
        ],  # Will be imputed (likely mean if no other correlation)
        "Comment": ["ok", "good", np.nan],  # Should be ignored
    }

    # Since there are no other useful numeric cols to predict 'Value',
    # MICE (like IterativeImputer) usually falls back to simple mean/median for that col
    # or predicts based on index if not shuffled.
    # Here we just check that it runs and 'Comment' is NOT touched.

    # We define expectation only for structure/non-numeric preservation check
    # We don't assert exact value of 'Value' here as it depends on fallback strategy

    assert_action_pipeline(
        MICEImputer(),
        transform_data=data,
        expected_result={
            "ID": ["u1", "u2", "u3"],
            "Value": [
                10.0,
                20.0,
                15.0,
            ],  # Will be imputed (likely mean if no other correlation)
            "Comment": ["ok", "good", np.nan],  # Should be ignored
        },
        fit_data=data,
    )
