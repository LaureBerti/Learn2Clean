from learn2clean.actions import VarianceThresholdSelector
from tests.utils.actions.assert_action_pipeline import assert_action_pipeline


def test_variance_threshold_selector_zero_variance() -> None:
    input_data = {
        "Feature_A": [1, 2, 3, 4, 5],
        "Feature_B": [10, 10, 10, 10, 10],
        "Feature_C": ["x", "y", "z", "a", "b"],
        "Feature_D": [1, 1, 0, 0, 1],
    }
    expected_output_data = {
        "Feature_A": [1, 2, 3, 4, 5],
        "Feature_C": ["x", "y", "z", "a", "b"],
        "Feature_D": [1, 1, 0, 0, 1],
    }

    vt_selector = VarianceThresholdSelector()

    assert_action_pipeline(
        vt_selector,
        input_data,
        expected_output_data,
        input_data,
    )
