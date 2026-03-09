from hydra import compose, initialize
from hydra.utils import instantiate

from learn2clean.actions import MinMaxScaler
from tests.utils.actions.assert_action_pipeline import assert_action_pipeline


def test_action_defaults():
    assert_action_pipeline(
        MinMaxScaler(),
        {"a": [0.0, 10.0, 35.0], "b": [100, 500, 900], "c_non_num": ["x", "y", "z"]},
        {"a": [-0.5, 0.0, 1.25], "b": [0.0, 2.0, 4.0], "c_non_num": ["x", "y", "z"]},
        {"a": [10.0, 20.0, 30.0], "b": [100, 200, 300], "c_non_num": ["x", "y", "z"]},
    )


def test_action_transform_without_fit():
    # Test results are not modified if fit is not called
    assert_action_pipeline(
        MinMaxScaler(columns=["a"], feature_range=(0, 3)),
        {"a": [0.0, 10.0, 35.0], "b": [100, 500, 900], "c_non_num": ["x", "y", "z"]},
        {"a": [0.0, 10.0, 35.0], "b": [100, 500, 900], "c_non_num": ["x", "y", "z"]},
    )


def test_action_params():
    assert_action_pipeline(
        MinMaxScaler(columns=["a"], feature_range=(0, 3)),
        {"a": [0.0, 10.0, 35.0], "b": [100, 500, 900], "c_non_num": ["x", "y", "z"]},
        {"a": [-1.5, 0.0, 3.75], "b": [100, 500, 900], "c_non_num": ["x", "y", "z"]},
        {"a": [10.0, 20.0, 30.0], "b": [100, 200, 300], "c_non_num": ["x", "y", "z"]},
    )


def test_action_titanic():
    assert_action_pipeline(
        MinMaxScaler(),
        {
            "Age": [5.0, 45.0, 90.0],
            "Fare": [10.0, 120.0, 50.0],
            "Cabin": ["D1", "E2", "F3"],
        },
        {
            "Age": [-0.0714286, 0.5, 1.14286],
            "Fare": [0.1, 1.2, 0.5],
            "Cabin": ["D1", "E2", "F3"],
        },
        {
            "Age": [10.0, 45.0, 80.0],
            "Fare": [0.0, 50.0, 100.0],
            "Cabin": ["C123", "B45", "A1"],
        },
    )


def test_action_with_hydra_config():
    assert_action_pipeline(
        instantiate(
            {
                "_target_": "learn2clean.actions.MinMaxScaler",
                "columns": ["a"],
            }
        ),
        {"a": [0.0, 10.0, 35.0], "b": [100, 500, 900], "c_non_num": ["x", "y", "z"]},
        {"a": [-0.5, 0.0, 1.25], "b": [100, 500, 900], "c_non_num": ["x", "y", "z"]},
        {"a": [10.0, 20.0, 30.0], "b": [100, 200, 300], "c_non_num": ["x", "y", "z"]},
    )
