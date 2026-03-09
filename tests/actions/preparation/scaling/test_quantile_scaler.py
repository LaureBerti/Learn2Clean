from hydra import compose, initialize
from hydra.utils import instantiate

from learn2clean.actions import QuantileScaler
from tests.utils.actions.assert_action_pipeline import assert_action_pipeline


def test_action_defaults():
    transform_data = {
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature_2": [10.0, 0.0, -10.0, 5.0, -5.0],
        "category": ["A", "B", "A", "C", "B"],
        "constant": [7.0, 7.0, 7.0, 7.0, 7.0],
    }

    assert_action_pipeline(
        action=QuantileScaler(random_state=42),
        transform_data=transform_data,
        expected_result={
            "feature_1": [0, 0.25, 0.5, 0.75, 1],
            "feature_2": [1, 0.5, 0, 0.75, 0.25],
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
        QuantileScaler(random_state=42),
        input_data,
        {
            "temp_c": [0.0, 0.25, 0.5, 0.75, 1.0],
            "humidity_percent": [1, 0.375, 0.0, 0.375, 1],
            "city": ["A", "B", "C", "B", "A"],
        },
        input_data.copy(),
    )


def test_action_transform_without_fit():
    # Test results are not modified if fit is not called
    transform_data = {"a": [1, 2, 3], "b": [10, 20, 30]}
    assert_action_pipeline(
        QuantileScaler(columns=["a"]),
        transform_data,
        transform_data.copy(),
    )


def test_action_params():
    transform_data = {"a": [1, 2, 3], "b": [10, 20, 30]}
    # Test result is modified after fit is called
    assert_action_pipeline(
        QuantileScaler(columns=["a"]),
        transform_data,
        {"a": [0, 0.5, 1], "b": [10, 20, 30]},
        transform_data.copy(),
    )


def test_action_with_hydra_config():
    assert_action_pipeline(
        instantiate(
            {
                "_target_": "learn2clean.actions.QuantileScaler",
                "columns": ["a"],
            }
        ),
        {"a": [1, 2, 3], "b": [10, 20, 30]},
        {"a": [0, 0.5, 1], "b": [10, 20, 30]},
        {"a": [1, 2, 3], "b": [10, 20, 30]},
    )
