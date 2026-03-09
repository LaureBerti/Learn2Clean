from hydra import compose, initialize
from hydra.utils import instantiate

from learn2clean.actions.dummy_add import DummyAdd
from tests.utils.actions.assert_action_pipeline import (
    assert_action_pipeline,
)


def test_action_defaults():
    assert_action_pipeline(
        DummyAdd(),
        {
            "age": [20, 25, 30],
            "income": [1000, 1500, 2000],
            "gender": ["M", "F", "F"],
        },
        {
            "age": [21, 26, 31],
            "income": [1001, 1501, 2001],
            "gender": ["M", "F", "F"],
        },
    )


def test_action_params_and_column_selection():
    assert_action_pipeline(
        DummyAdd(increment=5, columns=["age", "income", "gender"]),
        {
            "age": [20, 25, 30],
            "income": [1000, 1500, 2000],
            "gender": ["M", "F", "F"],
            "street": [1, 2, 3],
        },
        {
            "age": [25, 30, 35],
            "income": [1005, 1505, 2005],
            "gender": ["M", "F", "F"],
            "street": [1, 2, 3],
        },
    )


def test_action_exclusion():
    assert_action_pipeline(
        DummyAdd(exclude=["zip_code"]),
        {
            "age": [20, 25, 30],
            "income": [1000, 1500, 2000],
            "gender": ["M", "F", "F"],
            "zip_code": [90210, 10001, 75000],
        },
        {
            "age": [21, 26, 31],
            "income": [1001, 1501, 2001],
            "gender": ["M", "F", "F"],
            "zip_code": [90210, 10001, 75000],
        },
    )


def test_action_leakage_safety():
    assert_action_pipeline(
        DummyAdd(),
        {"A": [4, 5, 6], "B": [40, 50, 60], "C": ["a", "b", "c"]},
        {"A": [5, 6, 7], "B": [41, 51, 61], "C": ["a", "b", "c"]},
        {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["x", "y", "z"]},
    )


def test_action_with_hydra_config():
    assert_action_pipeline(
        instantiate(
            {
                "_target_": "learn2clean.actions.dummy_add.DummyAdd",
                "increment": 2,
                "columns": ["a", "b"],
            }
        ),
        {
            "a": [1, 2, 3],
            "b": [10, 20, 30],
            "c": [100, 200, 300],  # untransformed column
        },
        {
            "a": [3, 4, 5],
            "b": [12, 22, 32],
            "c": [100, 200, 300],
        },
    )
