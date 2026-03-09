import numpy as np

from learn2clean.distance.correlation_matrix import CorrelationMatrixDistance
from tests.utils.distance.assert_distance_metric import assert_distance_metric


def test_correlation_matrix_identity():
    """
    Test sanity check: Identical datasets imply identical correlation matrices.
    Distance should be 0.0.
    """
    data = {
        "A": [1, 2, 3, 4, 5],
        "B": [2, 4, 6, 8, 10],  # Perfecly correlated with A
    }
    assert_distance_metric(
        CorrelationMatrixDistance(),
        data,
        data,
        0.0,
    )


def test_correlation_matrix_decorrelation():
    """
    Test when a strong correlation is destroyed (becomes random/orthogonal).

    Scenario:
    P (Before): Col A and B are perfectly correlated (Corr = 1.0).
        Matrix P = [[1.0, 1.0],
                    [1.0, 1.0]]

    Q (After): Col A and B have zero correlation (Corr = 0.0).
        Matrix Q = [[1.0, 0.0],
                    [0.0, 1.0]]

    Calculation:
    Diff = Q - P = [[0.0, -1.0],
                    [-1.0, 0.0]]

    Frobenius Norm = sqrt(0^2 + (-1)^2 + (-1)^2 + 0^2)
                   = sqrt(2)
                   = 1.41421...
    """
    assert_distance_metric(
        CorrelationMatrixDistance(),
        {
            # P: Perfect correlation (y = x)
            "A": [1, 2, 3, 4],
            "B": [1, 2, 3, 4],
        },
        {
            # Q: Zero correlation (Orthogonal vectors centered at 0)
            # A: [-1, 1, -1, 1], B: [-1, -1, 1, 1] -> dot product is 0
            "A": [-1, 1, -1, 1],
            "B": [-1, -1, 1, 1],
        },
        1.4142,
    )


def test_correlation_matrix_inversion():
    """
    Test maximal change: Correlation goes from +1 to -1.

    Scenario:
    P (Before): A and B are perfectly correlated (+1).
        Matrix P = [[1.0, 1.0],
                    [1.0, 1.0]]

    Q (After): A and B are perfectly anti-correlated (-1).
        Matrix Q = [[ 1.0, -1.0],
                    [-1.0,  1.0]]

    Calculation:
    Diff = Q - P = [[ 0.0, -2.0],
                    [-2.0,  0.0]]

    Frobenius Norm = sqrt(0^2 + (-2)^2 + (-2)^2 + 0^2)
                   = sqrt(4 + 4)
                   = sqrt(8)
                   = 2.8284...
    """
    assert_distance_metric(
        CorrelationMatrixDistance(),
        {
            # P: Positive correlation
            "A": [1, 2, 3],
            "B": [1, 2, 3],
        },
        {
            # Q: Negative correlation
            "A": [1, 2, 3],
            "B": [3, 2, 1],
        },
        2.8284,
    )


def test_correlation_matrix_subset_columns():
    """
    Tests that the metric correctly handles datasets where columns have been removed.
    It should only calculate distance based on the INTERSECTION of columns.
    """
    # P has 3 columns: A, B (correlated), C (noise)
    data_p = {
        "A": [1, 2, 3],
        "B": [1, 2, 3],  # Corr(A,B) = 1
        "C": [9, 1, 5],
    }

    # Q has dropped C, but A and B are still perfectly correlated
    data_q = {
        "A": [1, 2, 3],
        "B": [1, 2, 3],  # Corr(A,B) still 1
    }

    # The metric should ignore C, compare A&B (unchanged) -> Dist 0.0
    assert_distance_metric(
        CorrelationMatrixDistance(),
        data_p,
        data_q,
        0.0,
    )
