import pandas as pd
import numpy as np
from learn2clean.distance.variance import (
    MeanVarianceDistance,
    MaxVarianceDistance,
    MedianVarianceDistance,
)
from tests.utils.distance.assert_distance_metric import assert_distance_metric

# --- DATA FIXTURES ---

# P: Raw data with mixed scales
# - 'low_var': small integers (Variance ~ 2.5)
# - 'high_var': multiples of 100 (Variance ~ 25000) -> The dominant feature
# - 'med_var': multiples of 10 (Variance ~ 250)
DATA_P = {
    "low_var": [1, 2, 3, 4, 5],
    "high_var": [100, 200, 300, 400, 500],
    "med_var": [10, 20, 30, 40, 50],
}

# Q: Data after a hypothetical Scaling operation
# - 'low_var': Unchanged (Variance ~ 2.5)
# - 'high_var': Scaled down drastically (Variance becomes ~ 2.5)
# - 'med_var': Scaled down moderately (Variance becomes ~ 2.5)
DATA_Q = {
    "low_var": [1, 2, 3, 4, 5],
    "high_var": [1, 2, 3, 4, 5],  # HUGE drop in variance
    "med_var": [1, 2, 3, 4, 5],  # Moderate drop
}


def get_variance_vectors():
    """
    Helper to calculate the actual variance vectors using Pandas.
    Note: Pandas uses ddof=1 (Sample Variance) by default.
    """
    df_p = pd.DataFrame(DATA_P)
    df_q = pd.DataFrame(DATA_Q)

    # Calculate variance vectors
    vec_p = df_p.var().fillna(0).to_numpy()
    vec_q = df_q.var().fillna(0).to_numpy()

    return vec_p, vec_q


# --- TESTS ---


def test_variance_identity():
    """
    Test 1: Identical datasets should result in 0.0 distance.
    """
    data = {"A": [1, 2, 3, 4, 5]}

    for MetricClass in [
        MeanVarianceDistance,
        MaxVarianceDistance,
        MedianVarianceDistance,
    ]:
        assert_distance_metric(MetricClass(), data, data, 0.0)


def test_mean_variance():
    """
    Test 2: Mean Variance.
    Represents the average reduction in scale across the dataset.

    Formula: |Mean(Var_Q) - Mean(Var_P)|
    """
    vec_p, vec_q = get_variance_vectors()

    expected_dist = abs(vec_q.mean() - vec_p.mean())

    assert_distance_metric(MeanVarianceDistance(), DATA_P, DATA_Q, expected_dist)


def test_max_variance():
    """
    Test 3: Max Variance (Sensitivity to Scaling).

    Logic:
    - In P, 'high_var' has a huge variance (25000).
    - In Q, all variances are small (~2.5).
    - This metric should show a MASSIVE distance (~24997.5), much larger
      than Mean or Median.
    """
    vec_p, vec_q = get_variance_vectors()

    # Compare maximum magnitude (Variance is always positive)
    max_p = np.max(vec_p)
    max_q = np.max(vec_q)

    expected_dist = abs(max_q - max_p)

    assert_distance_metric(MaxVarianceDistance(), DATA_P, DATA_Q, expected_dist)


def test_median_variance():
    """
    Test 4: Median Variance (Robustness).

    Logic:
    - P variances: [Low(~2.5), Med(~250), High(~25000)] -> Median is Med (~250).
    - Q variances: [Low(~2.5), Low(~2.5), Low(~2.5)]     -> Median is Low (~2.5).

    - The distance will be |2.5 - 250| (~247.5).
    - It ignores the change in the 'High' variance column entirely.
    """
    vec_p, vec_q = get_variance_vectors()

    expected_dist = abs(np.median(vec_q) - np.median(vec_p))

    assert_distance_metric(MedianVarianceDistance(), DATA_P, DATA_Q, expected_dist)


def test_variance_nan_handling():
    """
    Test 5: Handling of columns with insufficient data for Variance.

    Variance requires at least 2 data points (ddof=1).
    If N=1, Pandas returns NaN.
    """
    # Column with only 1 value -> Variance is NaN
    data_single = {"A": [10]}

    # BaseStatisticalMomentDistance handles NaNs by removing them.
    # If vector becomes empty, returns 0.0.
    assert_distance_metric(MeanVarianceDistance(), data_single, data_single, 0.0)
