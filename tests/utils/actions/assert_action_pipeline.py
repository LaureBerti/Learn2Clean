import numpy as np
import pandas as pd

from learn2clean.actions.data_frame_action import DataFrameAction
from tests.utils.assert_dataframes_allclose import assert_dataframes_allclose
from tests.utils.assert_dataframes_equal import assert_dataframes_equal

DataFrameLike = dict | pd.DataFrame
SeriesLike = list | np.ndarray | dict | pd.Series


def to_dataframe(data: DataFrameLike) -> pd.DataFrame:
    """Helper to convert input dict to DataFrame if necessary."""
    return pd.DataFrame(data) if isinstance(data, dict) else data


def to_series(data: SeriesLike) -> pd.Series:
    """Helper to convert input dict to Series if necessary."""
    if isinstance(data, pd.Series):
        return data
    if isinstance(data, (list, np.ndarray)):
        return pd.Series(data)
    if isinstance(data, dict):
        return pd.Series(data)
    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            return data.iloc[:, 0]
        raise ValueError("Cannot convert multi-column DataFrame to a single Series.")
    raise TypeError(f"Unsupported type for target conversion: {type(data)}")


def assert_action_pipeline(
    action: DataFrameAction,
    transform_data: DataFrameLike,
    expected_result: DataFrameLike,
    fit_data: DataFrameLike | None = None,
    fit_target: SeriesLike | None = None,
    numeric_allclose: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """
    Asserts the correct behavior of a DataFrameAction, supporting both
    fitless execution and the full fit/transform cycle (including a target y).

    Args:
        action: The DataFrameAction instance to test.
        transform_data: Input data used for action.transform().
        expected_result: The expected output DataFrame.
        fit_data: Data used for action.fit(). If None, action is run in fitless mode.
        fit_target: Target Series (y) used for action.fit(). Required if fit_data is not None
                    and the action is supervised (e.g., Feature Selection).
        numeric_allclose: If True, uses tolerance-based comparison for numeric columns.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.
    """
    transform_df = to_dataframe(transform_data)
    expected_df = to_dataframe(expected_result)

    assert isinstance(
        action, DataFrameAction
    ), f"Action '{action.name}' must be an instance of DataFrameAction"

    if fit_data is not None:
        fit_df = to_dataframe(fit_data)

        if fit_target is not None:
            fit_target = to_series(fit_target)
            action.fit(fit_df, fit_target)
        else:
            action.fit(fit_df)
        result_df = action.transform(transform_df)

    else:
        result_df = action(transform_df)

    if numeric_allclose:
        assert_dataframes_allclose(result_df, expected_df, rtol=rtol, atol=atol)
    else:
        assert_dataframes_equal(result_df, expected_df)
