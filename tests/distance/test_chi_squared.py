from learn2clean.distance.chi_squared import ChiSquaredDistance
from tests.utils.distance.assert_distance_metric import assert_distance_metric


def test_chi_squared_statistics_minimal():
    """
    Tests the Symmetric Chi-Squared calculation.

    Formula: 0.5 * sum( (P-Q)^2 / (P+Q) )

    P: [A: 100%, B: 0%] -> [1.0, 0.0]
    Q: [A: 50%, B: 50%] -> [0.5, 0.5]

    Calculation:
    - Term A: (1.0 - 0.5)^2 / (1.0 + 0.5) = 0.25 / 1.5 = 0.1666...
    - Term B: (0.0 - 0.5)^2 / (0.0 + 0.5) = 0.25 / 0.5 = 0.5
    - Sum = 0.6666...
    - Result (x 0.5) = 0.3333...
    """
    assert_distance_metric(
        ChiSquaredDistance(),
        {
            # DataFrame P: 4 "A", 0 "B"
            "data": ["A", "A", "A", "A"],
        },
        {
            # DataFrame Q: 2 "A", 2 "B"
            "data": ["A", "A", "B", "B"],
        },
        0.3333,
    )


def test_chi_squared_statistics_maximal_divergence():
    """
    Tests Maximal Divergence (Mutually Exclusive).

    P: [A: 100%] -> [1.0, 0.0] (on unified vocabulary A, B)
    Q: [B: 100%] -> [0.0, 1.0]

    Calculation:
    - Term A: (1.0 - 0.0)^2 / (1.0 + 0.0) = 1.0 / 1.0 = 1.0
    - Term B: (0.0 - 1.0)^2 / (0.0 + 1.0) = 1.0 / 1.0 = 1.0
    - Sum = 2.0
    - Result (x 0.5) = 1.0
    """
    assert_distance_metric(
        ChiSquaredDistance(),
        {
            "data": ["A", "A", "A", "A"],
        },
        {
            "data": ["B", "B", "B", "B"],
        },
        1.0,
    )


def test_chi_squared_statistics_partial_disparity():
    """
    Tests Partial Disparity.

    P: [1.0, 0.0]
    Q: [0.75, 0.25]

    Calculation:
    - Term A: (1.0 - 0.75)^2 / (1.0 + 0.75) = 0.0625 / 1.75 = 0.03571...
    - Term B: (0.0 - 0.25)^2 / (0.0 + 0.25) = 0.0625 / 0.25 = 0.25
    - Sum = 0.28571...
    - Result (x 0.5) = 0.14285...
    """
    assert_distance_metric(
        ChiSquaredDistance(),
        {
            # DataFrame P: 4 "A", 0 "B" (100% A)
            "data": ["A", "A", "A", "A"],
        },
        {
            # DataFrame Q: 3 "A", 1 "B" (75% A, 25% B)
            "data": ["A", "A", "A", "B"],
        },
        0.1428,
    )
