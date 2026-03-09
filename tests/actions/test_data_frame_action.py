from typing import ClassVar, Any

import pandas as pd
import pytest

from learn2clean.actions.data_frame_action import (
    DataFrameAction,
)


class DummyAction(DataFrameAction):
    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {"default_param": "initial_value"}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if "error_trigger" in self.params and self.params["error_trigger"]:
            raise ValueError("Simulated transformation error.")
        return df.copy()


def test_action_initialization_and_params_merge():
    action = DummyAction(threshold=0.8, exclude=["id"], default_param="overridden")
    assert action.params["threshold"] == 0.8
    assert action.params["default_param"] == "overridden"
    assert action.exclude == ["id"]
    assert action.name == "DummyAction"


def test_action_initialization_no_params():
    action = DummyAction()
    assert action.params["default_param"] == "initial_value"
    assert action.columns is None
    assert action.exclude is None


def test_logical_path_calculation():
    action = DummyAction()
    parts = action.__module__.split(".")
    expected_path = ".".join(parts[2 : len(parts) - 1])
    assert action.logical_path == expected_path


def test_select_columns_numeric_filter():
    df = pd.DataFrame(
        {"A": [1, 2], "B": ["x", "y"], "C": [3.0, 4.0], "D": [True, False]}
    )
    action = DummyAction()
    selected = action.select_columns(df, numeric_only=True)
    assert set(selected) == {"A", "C", "D"}


def test_select_columns_categorical_filter():
    df = pd.DataFrame(
        {"A": [1, 2], "B": ["x", "y"], "C": [3.0, 4.0], "D": [True, False]}
    )
    action = DummyAction()
    selected = action.select_columns(df, categorical_only=True)
    assert set(selected) == {"B"}


def test_select_columns_with_exclusion():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": ["x", "y"]})
    action = DummyAction(exclude=["B"])
    selected = action.select_columns(df, numeric_only=True)
    assert set(selected) == {"A"}


def test_select_columns_with_inclusion_list():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": ["x", "y"]})
    action = DummyAction(columns=["B", "C"])
    selected = action.select_columns(df, numeric_only=True)
    assert set(selected) == {"B"}
    action_empty_cols = DummyAction(columns=[])
    selected_empty = action_empty_cols.select_columns(df)
    assert selected_empty == []


def test_fit_stores_correct_columns():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": ["x", "y"]})
    action = DummyAction(exclude=["C"])
    action.fit(df)
    assert action.select_columns(df) == ["A", "B"]
    assert action._fitted_columns == ["A", "B"]
    assert action._get_fitted_columns(df) == ["A", "B"]


def test_get_fitted_columns_fallback():
    df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    action = DummyAction()
    selected = action._get_fitted_columns(df, numeric_only=True)
    assert selected == ["A"]


def test_call_executes_transform_and_returns_copy():
    df_original = pd.DataFrame({"A": [1, 2]})
    action = DummyAction()
    df_result = action(df_original)
    assert df_result is not df_original
    pd.testing.assert_frame_equal(df_result, df_original)


def test_call_propagates_exception():
    df = pd.DataFrame({"A": [1, 2]})
    action = DummyAction(error_trigger=True)
    with pytest.raises(ValueError, match="Simulated transformation error."):
        action(df)
