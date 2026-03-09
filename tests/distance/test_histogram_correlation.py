import pandas as pd

from learn2clean.distance.histogram_correlation import HistogramCorrelationDistance
from tests.utils.distance.assert_distance_metric import assert_distance_metric


def test_hist_correlation_identity():
    """
    Test 1: Identical datasets.

    Logic:
    - Histograms are identical.
    - Pearson Correlation = 1.0.
    - Distance = 1.0 - 1.0 = 0.0.
    """
    data = {"col1": [1, 2, 3, 4, 5, 10]}

    assert_distance_metric(HistogramCorrelationDistance(), data, data, 0.0)


def test_hist_correlation_inverse_shape():
    """
    Test 2: Perfectly Inverse Distributions (Maximal Distance).

    We force 2 bins to create disjoint binary vectors.
    Global Range: 0 to 10. Bins: [0, 5), [5, 10].

    P (Low values): All in Bin 1 -> Vector P = [1.0, 0.0]
    Q (High values): All in Bin 2 -> Vector Q = [0.0, 1.0]

    Correlation Calculation:
    - P centered: [ 0.5, -0.5]
    - Q centered: [-0.5,  0.5]
    - Covariance is negative.
    - Pearson Correlation of [1,0] and [0,1] is exactly -1.0.

    Expected Distance: 1.0 - (-1.0) = 2.0.
    """
    metric = HistogramCorrelationDistance(bins=2)

    assert_distance_metric(
        metric,
        {
            "data": [1, 1, 1, 1],  # All in first bin
        },
        {
            "data": [9, 9, 9, 9],  # All in second bin
        },
        2.0,
    )


def test_hist_correlation_zero_variance_identical():
    """
    Test 3: Zero Variance (Uniform Histograms) - Identical.

    Edge case handling for NaN correlation.

    P: [1, 2] -> 2 bins -> Hist P = [0.5, 0.5] (Variance = 0)
    Q: [1, 2] -> 2 bins -> Hist Q = [0.5, 0.5] (Variance = 0)

    np.corrcoef returns NaN because variance is zero.
    The code checks `np.allclose(p, q)`. Since they are identical, it returns 0.0.
    """
    metric = HistogramCorrelationDistance(bins=2)

    assert_distance_metric(
        metric,
        {
            "data": [1, 10],  # Creates hist [0.5, 0.5]
        },
        {
            "data": [1, 10],  # Creates hist [0.5, 0.5]
        },
        0.0,
    )


def test_hist_correlation_zero_variance_different():
    """
    Test 4: Zero Variance (One Flat) vs Normal - Different.

    P: Flat histogram [0.5, 0.5] (Variance 0).
    Q: Spiked histogram [1.0, 0.0] (Variance > 0).

    Correlation is NaN (division by zero variance of P).
    The code checks `np.allclose`. They are NOT identical.
    Fallback returns 1.0 (Uncorrelated/Distant).
    """
    metric = HistogramCorrelationDistance(bins=2)

    assert_distance_metric(
        metric,
        {
            "data": [1, 10],  # Hist P: [0.5, 0.5] -> Variance 0 -> NaN Corr
        },
        {
            "data": [1, 1],  # Hist Q: [1.0, 0.0]
        },
        1.0,  # Default fallback for NaN when not identical
    )


def test_hist_correlation_partial_similarity():
    """
    Test 5: Partial overlap (Standard usage).

    P: [1.0, 0.0]
    Q: [0.5, 0.5]

    Correlation of [1, 0] and [0.5, 0.5]:
    This is mathematically undefined or NaN if Q has 0 variance?
    No, [0.5, 0.5] has 0 variance.

    Let's try a valid variance case with 3 bins to avoid 0 variance.
    P: [1.0, 0.0, 0.0]
    Q: [0.0, 0.5, 0.5]

    Correlation should be negative (trends opposite).
    """
    metric = HistogramCorrelationDistance(bins=3)

    # We construct data to force histograms:
    # P: All in bin 1
    # Q: Split between bin 2 and 3

    # Range 0-30. Bins: 0-10, 10-20, 20-30.
    p_data = [5, 5]  # Bin 1
    q_data = [15, 25]  # Bin 2, Bin 3

    # Hist P: [1.0, 0.0, 0.0]
    # Hist Q: [0.0, 0.5, 0.5]

    # Correlation calculation:
    # Mean P=0.33, Mean Q=0.33
    # P centered: [ 0.66, -0.33, -0.33]
    # Q centered: [-0.33,  0.16,  0.16]
    # Dot prod: (0.66*-0.33) + ... is negative.

    # Let's trust the metric simply runs and returns something > 1.0
    # (since 1.0 is uncorrelated, >1.0 is negatively correlated)

    df_p = {"A": p_data}
    df_q = {"A": q_data}

    result = metric.calculate(pd.DataFrame(df_p), pd.DataFrame(df_q))

    # We expect distance > 1.0 because shapes are somewhat opposite
    assert result > 1.0
