import numpy as np
from learn2clean.distance.jensen_shannon import JensenShannonDistance
from tests.utils.distance.assert_distance_metric import assert_distance_metric


def test_jensen_shannon_identity():
    """
    Test 1: Identical distributions.

    Jensen-Shannon distance is 0 when P == Q.
    """
    data = {"col1": [1, 2, 3, 4]}

    assert_distance_metric(JensenShannonDistance(), data, data, 0.0)


def test_jensen_shannon_disjoint():
    """
    Test 2: Completely Disjoint Distributions (Maximal Distance).

    P: All in Bin 1 -> [1.0, 0.0]
    Q: All in Bin 2 -> [0.0, 1.0]

    Mathematical Note:
    - Scipy uses natural log (base e) by default.
    - Max JS Divergence = ln(2) ≈ 0.693
    - Max JS Distance = sqrt(ln(2)) ≈ 0.83255
    """
    # Force 2 bins to ensure clean separation
    metric = JensenShannonDistance(bins=2)

    # Expected value: sqrt(ln(2))
    expected_max_dist = np.sqrt(np.log(2))

    assert_distance_metric(
        metric,
        {
            "data": [1, 1, 1],  # Bin 1
        },
        {
            "data": [10, 10, 10],  # Bin 2
        },
        expected_max_dist,  # approx 0.83255
    )


def test_jensen_shannon_mixture():
    """
    Test 3: Partial Overlap (Uniform vs Peaked).

    P: Uniform [0.5, 0.5] (Spread across 2 bins)
    Q: Peaked [1.0, 0.0] (Concentrated in 1st bin)

    Calculation (Base e):
    M = 0.5*(P+Q) = [0.75, 0.25]

    KL(P||M) = 0.5*ln(0.5/0.75) + 0.5*ln(0.5/0.25)
             = 0.5*ln(2/3) + 0.5*ln(2)
             ≈ -0.2027 + 0.3465 = 0.1438

    KL(Q||M) = 1.0*ln(1.0/0.75) + 0
             = ln(1.333) ≈ 0.2876

    JS_Div = 0.5*0.1438 + 0.5*0.2876 ≈ 0.2157
    JS_Dist = sqrt(0.2157) ≈ 0.4644
    """
    metric = JensenShannonDistance(bins=2)

    # P: [1, 10] -> Histo [1, 1] -> Probs [0.5, 0.5]
    # Q: [1, 1]  -> Histo [2, 0] -> Probs [1.0, 0.0]

    # Let's calculate the exact expected float using numpy for precision
    p = np.array([0.5, 0.5])
    q = np.array([1.0, 0.0])
    m = 0.5 * (p + q)

    # KL Divergence manually
    # We use explicit masking/eps logic similar to what Scipy does internally
    def kl(a, b):
        return np.sum(
            [val_a * np.log(val_a / val_b) for val_a, val_b in zip(a, b) if val_a > 0]
        )

    js_div = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    expected_dist = np.sqrt(js_div)

    assert_distance_metric(
        metric,
        {
            "data": [1, 10],
        },
        {
            "data": [1, 1],
        },
        expected_dist,  # approx 0.4644
    )
