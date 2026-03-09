from hydra import compose, initialize
from hydra.utils import instantiate

from learn2clean.actions import ZScoreScaler
from tests.utils.actions.assert_action_pipeline import assert_action_pipeline


def test_action_defaults():
    transform_data = {
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature_2": [10.0, 0.0, -10.0, 5.0, -5.0],
        "category": ["A", "B", "A", "C", "B"],
        "constant": [7.0, 7.0, 7.0, 7.0, 7.0],
    }

    assert_action_pipeline(
        action=ZScoreScaler(),
        transform_data=transform_data,
        expected_result={
            "feature_1": [-1.41421, -0.707107, 0, 0.707107, 1.41421],
            "feature_2": [1.41421, 0.0, -1.41421, 0.707107, -0.707107],
            "category": ["A", "B", "A", "C", "B"],
            "constant": [0, 0, 0, 0, 0],
        },
        fit_data=transform_data.copy(),
        numeric_allclose=True,
    )


def test_action_custom_data():
    input_data = {
        "temp_c": [10.0, 15.0, 20.0, 25.0, 30.0],
        "humidity_percent": [60, 50, 40, 50, 60],
        "city": ["A", "B", "C", "B", "A"],
    }

    assert_action_pipeline(
        ZScoreScaler(),
        input_data,
        {
            "temp_c": [-1.41421, -0.707107, 0.000000, 0.707107, 1.41421],
            "humidity_percent": [1.06904, -0.267261, -1.603567, -0.267261, 1.06904],
            "city": ["A", "B", "C", "B", "A"],
        },
        input_data.copy(),
    )


def test_action_transform_without_fit():
    # Test results are not modified if fit is not called
    transform_data = {"a": [1, 2, 3], "b": [10, 20, 30]}
    assert_action_pipeline(
        ZScoreScaler(columns=["a"]),
        transform_data,
        transform_data.copy(),
    )


def test_action_params():
    transform_data = {"a": [1, 2, 3], "b": [10, 20, 30]}
    # Test result is modified after fit is called
    assert_action_pipeline(
        ZScoreScaler(columns=["a"]),
        transform_data,
        {"a": [-1.22474, 0, 1.22474], "b": [10, 20, 30]},
        transform_data.copy(),
    )


def test_action_with_hydra_config():
    assert_action_pipeline(
        instantiate(
            {
                "_target_": "learn2clean.actions.ZScoreScaler",
                "columns": ["a"],
            }
        ),
        {"a": [1, 2, 3], "b": [10, 20, 30]},
        {"a": [-1.224745, 0.000000, 1.224745], "b": [10, 20, 30]},
        {"a": [1, 2, 3], "b": [10, 20, 30]},
    )
