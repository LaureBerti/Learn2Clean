import pandas as pd
import pytest

from learn2clean.distance.base_distance import BaseDistance

TOLERANCE = 1e-4


def assert_distance_metric(
    metric: BaseDistance,
    data_p: dict | pd.DataFrame,
    data_q: dict | pd.DataFrame,
    expected_result: float,
) -> None:
    """
    Asserts that the metric correctly calculates the distance between P and Q,
    handling conversions from dict to DataFrame if necessary.
    """
    df_p = pd.DataFrame(data_p) if isinstance(data_p, dict) else data_p
    df_q = pd.DataFrame(data_q) if isinstance(data_q, dict) else data_q

    assert isinstance(
        metric, BaseDistance
    ), f"Metric '{metric.name}' must be an instance of BaseDistance"

    result = metric.calculate(df_p, df_q)

    assert (
        pytest.approx(result, abs=TOLERANCE) == expected_result
    ), f"Metric {metric.name} failed. Expected: {expected_result:.4f}, Got: {result:.4f} (Q - P)"
