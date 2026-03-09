import pandas as pd
import numpy as np
from learn2clean.distance.skewness import (
    MeanSkewnessDistance,
    MaxSkewnessDistance,
    MedianSkewnessDistance,
)
from tests.utils.distance.assert_distance_metric import assert_distance_metric

# --- DATA FIXTURES ---

# P: Perfectly symmetric distributions (Skewness = 0.0)
DATA_P = {
    "col_stable": [1, 2, 3, 4, 5],  # Skew = 0
    "col_change": [1, 2, 3, 4, 5],  # Skew = 0
    "col_extreme": [10, 20, 30, 40, 50],  # Skew = 0
}

# Q: Modified distributions with varying asymmetry
DATA_Q = {
    "col_stable": [1, 2, 3, 4, 5],  # Identical (Skew = 0)
    "col_change": [1, 1, 1, 2, 5],  # Moderate Positive Skew (~1.5)
    "col_extreme": [1, 1, 1, 1, 100],  # Massive Positive Skew (~5.0)
}


def get_skewness_vectors():
    """
    Helper to calculate the actual skewness vectors dynamically using Pandas.
    This serves as the 'Ground Truth'.
    """
    df_p = pd.DataFrame(DATA_P)
    df_q = pd.DataFrame(DATA_Q)

    # Calculate skewness vectors (handling potential NaNs if any)
    vec_p = df_p.skew().fillna(0).to_numpy()
    vec_q = df_q.skew().fillna(0).to_numpy()

    return vec_p, vec_q


# --- TESTS ---


def test_skewness_identity():
    """
    Test 1: Identical datasets should result in 0.0 distance for all metrics.
    """
    data = {"A": [1, 2, 3, 4, 5]}

    for MetricClass in [
        MeanSkewnessDistance,
        MaxSkewnessDistance,
        MedianSkewnessDistance,
    ]:
        assert_distance_metric(MetricClass(), data, data, 0.0)


def test_mean_skewness():
    """
    Test 2: Mean Skewness.

    Formula: |Mean(Skew_Q) - Mean(Skew_P)|

    Logic:
    - P has 0 skewness everywhere (Mean P = 0).
    - Q has mixed skewness.
    - Result is simply Mean(Q).
    """
    vec_p, vec_q = get_skewness_vectors()

    expected_dist = abs(vec_q.mean() - vec_p.mean())

    assert_distance_metric(MeanSkewnessDistance(), DATA_P, DATA_Q, expected_dist)


def test_max_skewness():
    """
    Test 3: Max Skewness (Sensitivity to Extremes).

    Formula: |Max(|Skew_Q|) - Max(|Skew_P|)|

    Logic:
    - This metric should catch the 'col_extreme' in Q which has high skewness.
    - It ignores the stable columns.
    """
    vec_p, vec_q = get_skewness_vectors()

    # We compare the maximum absolute skewness
    max_abs_p = np.max(np.abs(vec_p))
    max_abs_q = np.max(np.abs(vec_q))

    expected_dist = abs(max_abs_q - max_abs_p)

    assert_distance_metric(MaxSkewnessDistance(), DATA_P, DATA_Q, expected_dist)


def test_median_skewness():
    """
    Test 4: Median Skewness (Robustness).

    Formula: |Median(Skew_Q) - Median(Skew_P)|

    Logic:
    - Q Skew Vector (sorted approx): [0.0 (stable), 1.5 (change), 5.0 (extreme)]
    - Median is 1.5 ('col_change').
    - The huge skew of 'col_extreme' is IGNORED.
    """
    vec_p, vec_q = get_skewness_vectors()

    expected_dist = abs(np.median(vec_q) - np.median(vec_p))

    assert_distance_metric(MedianSkewnessDistance(), DATA_P, DATA_Q, expected_dist)


def test_skewness_nan_handling():
    """
    Test 5: Handling of columns that result in NaN Skewness.

    Pandas skew() returns NaN for constant columns or N < 3.
    The base class must handle this gracefully.
    """
    # Constant column -> Skew is NaN (undefined variance)
    data_const = {"A": [1, 1, 1, 1]}

    # BaseStatisticalMomentDistance handles NaNs by removing them.
    # If all columns are NaN/Constant, it usually returns 0.0 (empty vector case).
    assert_distance_metric(MeanSkewnessDistance(), data_const, data_const, 0.0)
