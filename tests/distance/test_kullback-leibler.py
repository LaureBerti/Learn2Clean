import numpy as np

from learn2clean.distance.kullback_leibler import KullbackLeiblerDistance
from tests.utils.distance.assert_distance_metric import assert_distance_metric


def test_kl_divergence_identity():
    """
    Test 1: Identical distributions.

    KL(P || P) = sum( P(i) * log( P(i) / P(i) ) )
               = sum( P(i) * log(1) )
               = sum( 0 )
               = 0.0
    """
    data = {"col1": [1, 2, 3, 4]}

    assert_distance_metric(KullbackLeiblerDistance(), data, data, 0.0)


def test_kl_divergence_zero_probability_handling():
    """
    Test 2: Infinite Divergence Protection (P > 0 where Q = 0).

    P: [1.0, 0.0] (All mass in bin 1)
    Q: [0.0, 1.0] (All mass in bin 2)

    Mathematically, KL(P || Q) involves log(1.0 / 0.0) -> Infinity.

    The implementation adds epsilon to Q to avoid crash.
    Q_safe = [epsilon, 1.0] (renormalized conceptually, but locally just replaced)

    KL approx = 1.0 * log(1.0 / epsilon) + 0
              = -log(epsilon)

    With epsilon = 1e-10, result should be approx ln(1e10) ≈ 23.02
    """
    metric = KullbackLeiblerDistance(bins=2, epsilon=1e-10)

    # Expected value: -log(1e-10) = 10 * ln(10) ≈ 23.02585
    expected_val = -np.log(1e-10)

    assert_distance_metric(
        metric,
        {
            "data": [1, 1, 1],  # Bin 1
        },
        {
            "data": [10, 10, 10],  # Bin 2
        },
        expected_val,
    )


def test_kl_divergence_standard_case():
    """
    Test 3: Standard divergence between two overlapping distributions.

    P: Uniform [0.5, 0.5]
    Q: Peaked  [0.9, 0.1]

    KL(P || Q) = 0.5 * log(0.5 / 0.9) + 0.5 * log(0.5 / 0.1)
               = 0.5 * log(0.555)      + 0.5 * log(5.0)
               = 0.5 * (-0.5877)       + 0.5 * (1.6094)
               = -0.2938               + 0.8047
               = 0.5108
    """
    metric = KullbackLeiblerDistance(bins=2)

    # P: [0.5, 0.5]
    p_data = [2, 8]  # one low, one high

    # Q: [0.9, 0.1]
    # To simulate this with integer counts is tricky (9 low, 1 high)
    q_data = [2] * 9 + [8] * 1  # 9 times '2' (low), 1 time '8' (high)

    # Manual calculation Check
    p_probs = np.array([0.5, 0.5])
    q_probs = np.array([0.9, 0.1])

    expected_kl = np.sum(p_probs * np.log(p_probs / q_probs))  # approx 0.5108

    assert_distance_metric(metric, {"data": p_data}, {"data": q_data}, expected_kl)
