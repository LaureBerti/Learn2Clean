from typing import Any

import numpy as np
import pandas as pd


def assert_dataframes_equal(
    actual: pd.DataFrame | dict[str, list | np.ndarray],
    expected: pd.DataFrame | dict[str, list | np.ndarray],
    check_dtype: bool = False,
    check_index: bool = False,
    **kwargs: Any
) -> None:
    """
    Assert that two DataFrames (or dicts convertible to DataFrames) are equal
    in content, handling common testing prerequisites.

    Args:
        actual: The actual DataFrame or data dictionary result.
        expected: The expected DataFrame or data dictionary.
        check_dtype: Whether to check the column data types. Defaults to False
                     (to match your original logic).
        check_index: Whether to check the row index values. Defaults to False
                     (using reset_index).
        **kwargs: Additional keyword arguments passed to pd.testing.assert_frame_equal.
    """
    # 1. Convert dicts to DataFrames for consistent comparison
    df_actual = pd.DataFrame(actual) if isinstance(actual, dict) else actual
    df_expected = pd.DataFrame(expected) if isinstance(expected, dict) else expected

    # 2. Prepare DataFrames based on parameters (reset index is common for transformations)
    if not check_index:
        df_actual = df_actual.reset_index(drop=True)
        df_expected = df_expected.reset_index(drop=True)

    # 3. Print DataFrames for clear debugging output (crucial for test failures)
    print("\n--- Actual Result ---")
    print(df_actual.to_markdown(index=False))

    print("\n--- Expected Result ---")
    print(df_expected.to_markdown(index=False))

    # 4. Perform the assertion
    pd.testing.assert_frame_equal(
        df_actual,
        df_expected,
        check_dtype=check_dtype,
        # Check column order is usually desired unless explicitly ignored
        check_like=True,
        **kwargs
    )
