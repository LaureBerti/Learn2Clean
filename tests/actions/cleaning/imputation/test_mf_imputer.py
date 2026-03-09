import numpy as np

from learn2clean.actions import MFImputer
from tests.utils.actions.assert_action_pipeline import assert_action_pipeline


def test_mf_imputation_categorical() -> None:
    """
    Test MF on categorical data.
    The most frequent value (Mode) should be used to fill NaNs.
    """
    input_data = {
        "City": ["Paris", "London", "Paris", np.nan, "London", "Paris"],
        "Code": [1, 2, 1, 1, 2, 1],
    }
    # "Paris" appears 3 times, "London" 2 times. NaN should become "Paris".
    expected_data = {
        "City": ["Paris", "London", "Paris", "Paris", "London", "Paris"],
        "Code": [1, 2, 1, 1, 2, 1],
    }

    assert_action_pipeline(
        MFImputer(),
        transform_data=input_data,
        expected_result=expected_data,
        fit_data=input_data,
    )


def test_mf_imputation_numerical() -> None:
    """
    Test MF on numerical data (behaves like Mode).
    """
    input_data = {"Values": [10.0, 10.0, 20.0, np.nan]}
    expected_data = {"Values": [10.0, 10.0, 20.0, 10.0]}

    assert_action_pipeline(
        MFImputer(),
        transform_data=input_data,
        expected_result=expected_data,
        fit_data=input_data,
        numeric_allclose=True,
    )
