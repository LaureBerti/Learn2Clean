from learn2clean.distance.histogram_intersection import HistogramIntersectionDistance
from tests.utils.distance.assert_distance_metric import assert_distance_metric


def test_hist_intersection_identity():
    """
    Test 1: Identical datasets.

    Logic:
    - P: [0.5, 0.5] (e.g.)
    - Q: [0.5, 0.5]
    - Intersection (Min) = 0.5 + 0.5 = 1.0
    - Distance = 1.0 - 1.0 = 0.0
    """
    data = {"col1": [1, 2, 3, 4]}

    assert_distance_metric(
        HistogramIntersectionDistance(),
        data,
        data,
        0.0,
    )


def test_hist_intersection_disjoint():
    """
    Test 2: Completely Disjoint Distributions.

    We use bins=2 to separate low values from high values.
    Range: 0 to 10. Bins: [0, 5), [5, 10].

    P: All values in Bin 1 -> Vector P = [1.0, 0.0]
    Q: All values in Bin 2 -> Vector Q = [0.0, 1.0]

    Calculation:
    - Overlap: min(1,0) + min(0,1) = 0 + 0 = 0.0
    - Distance: 1.0 - 0.0 = 1.0
    """
    metric = HistogramIntersectionDistance(bins=2)

    assert_distance_metric(
        metric,
        {
            "data": [1, 1, 2, 2],  # All < 5
        },
        {
            "data": [8, 8, 9, 9],  # All > 5
        },
        1.0,
    )


def test_hist_intersection_partial_overlap():
    """
    Test 3: Partial Overlap (50%).

    Range: 0 to 10. Bins: 2.

    P: Uniform spread -> Vector P = [0.5, 0.5] (One low, one high)
    Q: Only low values -> Vector Q = [1.0, 0.0] (Two low)

    Calculation:
    - Bin 1: min(P=0.5, Q=1.0) = 0.5
    - Bin 2: min(P=0.5, Q=0.0) = 0.0
    - Total Intersection = 0.5
    - Distance = 1.0 - 0.5 = 0.5
    """
    metric = HistogramIntersectionDistance(bins=2)

    assert_distance_metric(
        metric,
        {
            # P: Spread across both bins [0-5] and [5-10]
            "data": [1, 9],
        },
        {
            # Q: Concentrated in first bin [0-5]
            "data": [1, 2],
        },
        0.5,
    )
