import numpy as np
from learn2clean.actions import MeanImputer
from tests.utils.actions.assert_action_pipeline import assert_action_pipeline


def test_mean_imputation_standard() -> None:
    """
    Standard test: Replaces NaNs with the mean of the column (from fit data).
    """
    # Mean of A is (1+2+3)/3 = 2.0
    # Mean of B is (10+20+60)/3 = 30.0
    input_data = {"A": [1.0, 2.0, 3.0, np.nan], "B": [10.0, 20.0, 60.0, np.nan]}

    expected_data = {"A": [1.0, 2.0, 3.0, 2.0], "B": [10.0, 20.0, 60.0, 30.0]}

    assert_action_pipeline(
        MeanImputer(),
        transform_data=input_data,
        expected_result=expected_data,
        fit_data=input_data,
        numeric_allclose=True,
    )


def test_mean_imputation_ignores_categorical() -> None:
    """
    Safety test: Should strictly ignore categorical/text columns.
    """
    input_data = {
        "Val": [1.0, 3.0, np.nan],  # Mean = 2.0
        "Cat": ["A", "B", np.nan],  # Should remain NaN
    }

    expected_data = {"Val": [1.0, 3.0, 2.0], "Cat": ["A", "B", np.nan]}

    assert_action_pipeline(
        MeanImputer(),
        transform_data=input_data,
        expected_result=expected_data,
        fit_data=input_data,
        numeric_allclose=True,
    )
