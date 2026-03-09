import numpy as np

from learn2clean.actions import KNNImputer
from tests.utils.actions.assert_action_pipeline import assert_action_pipeline


def test_knn_imputation_spatial_logic() -> None:
    """
    Test KNN logic: Missing value should be imputed based on nearest neighbors.
    Case: 1D clustering.
    Group 1: Values around 0 (0, 1, 0.5)
    Group 2: Values around 100 (100, 101, 99)
    A missing value in row with neighbor 0 should be close to 0, not 50 (mean).
    """
    # A is the feature used for distance. B has missing values.
    input_data = {"A": [0.1, 0.2, 100.1, 100.2], "B": [10.0, np.nan, 500.0, 505.0]}

    # Row 1 (A=0.2) is extremely close to Row 0 (A=0.1).
    # So Row 1's B should be imputed close to Row 0's B (10.0).
    # It is far from Rows 2 and 3.
    expected_data = {
        "A": [0.1, 0.2, 100.1, 100.2],
        "B": [10.0, 10.0, 500.0, 505.0],  # 10.0 is the nearest neighbor value
    }

    assert_action_pipeline(
        KNNImputer(n_neighbors=1),  # Look only at the closest one
        transform_data=input_data,
        expected_result=expected_data,
        fit_data=input_data,
        numeric_allclose=True,
        atol=0.1,
    )


def test_knn_ignores_categorical() -> None:
    """
    Safety test: KNN must strictly ignore text columns to avoid crashing
    on distance calculation.
    """
    input_data = {"A": [1.0, 2.0, np.nan], "Cat": ["A", "B", "C"]}
    # Should only impute A (mean of 1 and 2 is 1.5), and leave Cat alone
    expected_data = {"A": [1.0, 2.0, 1.5], "Cat": ["A", "B", "C"]}

    assert_action_pipeline(
        KNNImputer(n_neighbors=2),
        transform_data=input_data,
        expected_result=expected_data,
        fit_data=input_data,
        numeric_allclose=True,
    )
