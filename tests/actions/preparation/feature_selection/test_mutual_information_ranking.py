import numpy as np

from learn2clean.actions import MutualInformationSelector
from tests.utils.actions.assert_action_pipeline import assert_action_pipeline


def test_mutual_information_selector_defaults() -> None:
    """
    Tests MutualInformationSelector with k=1 using a larger, non-linear dataset
    to ensure it selects Feature_A which has the highest MI score.
    """

    rng = np.random.RandomState(42)
    N = 100
    feature_a = rng.uniform(-5, 5, N)
    feature_b = rng.normal(0, 1, N)
    feature_c = rng.uniform(0, 1, N)
    feature_d = [f"cat_{i}" for i in range(N)]
    target_y = (np.abs(feature_a) > 2).astype(int)

    expected_output_data = {
        "Feature_A": feature_a,
        "Feature_D": feature_d,
    }

    input_data = {
        "Feature_A": feature_a,
        "Feature_B": feature_b,
        "Feature_C": feature_c,
        "Feature_D": feature_d,
    }
    assert_action_pipeline(
        MutualInformationSelector(k=1),
        input_data,
        expected_output_data,
        input_data,
        target_y,
    )
