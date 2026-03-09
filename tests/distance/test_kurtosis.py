import pandas as pd

from learn2clean.distance.kurtosis import (
    MeanKurtosisDistance,
    MaxKurtosisDistance,
    MedianKurtosisDistance,
)
from tests.utils.distance.assert_distance_metric import assert_distance_metric

# --- DATA FIXTURES ---

# P: Normal/Uniform-ish distributions (Low Kurtosis)
DATA_P = {
    "col_stable": [1, 2, 3, 4, 5],  # Kurtosis ~ -1.2
    "col_change": [1, 2, 3, 4, 5],  # Kurtosis ~ -1.2
    "col_extreme": [1, 2, 3, 4, 5],  # Kurtosis ~ -1.2
}

# Q: Modified distributions
DATA_Q = {
    "col_stable": [1, 2, 3, 4, 5],  # Identical (Kurtosis Unchanged)
    "col_change": [1, 1, 1, 1, 5],  # Slight change (Kurtosis ~ 1.5)
    "col_extreme": [0, 0, 0, 0, 100],  # Massive Outlier (Kurtosis ~ 5.0)
}


def get_kurtosis_vectors():
    """Helper to get the actual kurtosis vectors using Pandas."""
    df_p = pd.DataFrame(DATA_P)
    df_q = pd.DataFrame(DATA_Q)

    # Calculate kurtosis vectors
    vec_p = df_p.kurt().to_numpy()
    vec_q = df_q.kurt().to_numpy()

    return vec_p, vec_q


# --- TESTS ---


def test_kurtosis_identity():
    """
    Test 1: Identical datasets should result in 0.0 distance for all metrics.
    """
    data = {"A": [1, 2, 3, 4, 5]}

    # Check all 3 classes
    for MetricClass in [
        MeanKurtosisDistance,
        MaxKurtosisDistance,
        MedianKurtosisDistance,
    ]:
        assert_distance_metric(MetricClass(), data, data, 0.0)


def test_mean_kurtosis():
    """
    Test 2: Mean Kurtosis.

    Logic:
    - Calculates the kurtosis for each column in P and Q.
    - Takes the mean of vector P and mean of vector Q.
    - Result = |Mean(Q) - Mean(P)|
    """
    vec_p, vec_q = get_kurtosis_vectors()

    expected_dist = abs(vec_q.mean() - vec_p.mean())

    assert_distance_metric(MeanKurtosisDistance(), DATA_P, DATA_Q, expected_dist)


def test_max_kurtosis():
    """
    Test 3: Max Kurtosis (Outlier focus).

    Logic:
    - Focuses on the column with the worst tail behavior (Highest absolute Kurtosis).
    - Result = |Max(|Q|) - Max(|P|)|

    In our data: 'col_extreme' in Q has a huge kurtosis. This metric should
    yield the highest distance value compared to Mean or Median.
    """
    vec_p, vec_q = get_kurtosis_vectors()

    # We compare the maximum magnitudes
    max_abs_p = abs(vec_p).max()
    max_abs_q = abs(vec_q).max()

    expected_dist = abs(max_abs_q - max_abs_p)

    assert_distance_metric(MaxKurtosisDistance(), DATA_P, DATA_Q, expected_dist)


def test_median_kurtosis():
    """
    Test 4: Median Kurtosis (Robustness).

    Logic:
    - Calculates the median kurtosis of the columns.
    - 'col_extreme' (the huge outlier) should be ignored by the median logic
      if there are enough columns.

    In our data (3 cols):
    - Sorted Q Kurtosis: [Stable(~-1.2), Change(~1.5), Extreme(~5.0)]
    - Median is 'Change'. The 'Extreme' column doesn't spike the metric.
    """
    vec_p, vec_q = get_kurtosis_vectors()

    import numpy as np

    expected_dist = abs(np.median(vec_q) - np.median(vec_p))

    assert_distance_metric(MedianKurtosisDistance(), DATA_P, DATA_Q, expected_dist)


def test_kurtosis_nan_handling():
    """
    Test 5: Handling of columns that result in NaN Kurtosis.

    Pandas kurtosis returns NaN if N < 4 (usually) or for constant columns.
    The Base class should filter these out.
    """
    # Dataset with too few samples (N=3) -> Kurtosis is NaN
    data_small = {"A": [1, 2, 3]}

    # BaseStatisticalMomentDistance handles NaNs by removing them.
    # If all columns are NaN, it returns 0.0 (empty vector).
    assert_distance_metric(MeanKurtosisDistance(), data_small, data_small, 0.0)
