import pandas as pd
import numpy as np
from learn2clean.distance.wasserstein import WassersteinDistance
from tests.utils.distance.assert_distance_metric import assert_distance_metric


def test_wasserstein_identity():
    """
    Test 1: Identical distributions.
    Cost to move mass from P to P is 0.
    """
    data = {"col1": [1, 2, 3, 4, 10]}

    assert_distance_metric(WassersteinDistance(), data, data, 0.0)


def test_wasserstein_pure_shift():
    """
    Test 2: Pure Translation (Shift).

    P: All points at 0.
    Q: All points at 10.

    The "Earth Mover's Distance" is logically 10.0 (Amount of work to move
    the pile of data from position 0 to position 10).
    """
    assert_distance_metric(
        WassersteinDistance(), {"A": [0, 0, 0]}, {"A": [10, 10, 10]}, 10.0
    )


def test_wasserstein_different_sizes():
    """
    Test 3: Robustness to different sample sizes.

    Wasserstein works on distributions (CDFs), not row-by-row.

    P: [0] (Single point at 0)
    Q: [0, 0, 0, 0] (Four points at 0)

    The distribution is effectively identical (100% of mass at 0).
    Distance should be 0.0.
    """
    assert_distance_metric(WassersteinDistance(), {"A": [0]}, {"A": [0, 0, 0, 0]}, 0.0)

    # Case B: Different sizes with shift
    # P: [0]
    # Q: [10, 10]
    # Distance should be 10.0
    assert_distance_metric(WassersteinDistance(), {"A": [0]}, {"A": [10, 10]}, 10.0)


def test_wasserstein_nan_handling():
    """
    Test 4: NaN Handling (Explicitly filtered in code).

    P: [0, NaN, 0] -> Cleaned to [0, 0]
    Q: [1, 1, NaN] -> Cleaned to [1, 1]

    Distance between [0, 0] and [1, 1] is 1.0.
    """
    assert_distance_metric(
        WassersteinDistance(), {"A": [0, np.nan, 0]}, {"A": [1, 1, np.nan]}, 1.0
    )


def test_wasserstein_ignore_non_numeric():
    """
    Test 5: Non-numeric columns should be ignored.

    If we have a numeric column (valid) and a string column (invalid),
    the metric should calculate the distance on the numeric one and ignore the string one.

    Col A (Numeric): [0] vs [10] -> Dist 10
    Col B (String):  ['a'] vs ['b'] -> Ignored

    Result: Mean([10]) = 10.0
    """
    data_p = pd.DataFrame({"A": [0], "B": ["a"]})

    data_q = pd.DataFrame({"A": [10], "B": ["b"]})

    assert_distance_metric(WassersteinDistance(), data_p, data_q, 10.0)


def test_wasserstein_no_numeric_data():
    """
    Test 6: Dataset with ONLY non-numeric data.

    Should return 0.0 (safe fallback) instead of crashing.
    """
    data_str = {"B": ["a", "b", "c"]}

    assert_distance_metric(WassersteinDistance(), data_str, data_str, 0.0)
