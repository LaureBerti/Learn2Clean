import numpy as np
import pandas as pd


def assert_dataframes_allclose(
    actual: pd.DataFrame | dict[str, list | np.ndarray],
    expected: pd.DataFrame | dict[str, list | np.ndarray],
    rtol: float = 1e-5,
    atol: float = 1e-8,
    check_dtype: bool = False,
    check_index: bool = False,
) -> None:
    """
    Assert that two DataFrames (or dicts convertible to DataFrames) are numerically close
    for numeric columns and exactly equal for non-numeric columns.

    Args:
        actual: The actual DataFrame or data dictionary result.
        expected: The expected DataFrame or data dictionary.
        rtol: Relative tolerance for numeric comparison.
        atol: Absolute tolerance for numeric comparison.
        check_dtype: Whether to check column dtypes (default False).
        check_index: Whether to check row index (default False).
    """
    # 1. Convert dicts to DataFrames
    df_actual = pd.DataFrame(actual) if isinstance(actual, dict) else actual.copy()
    df_expected = (
        pd.DataFrame(expected) if isinstance(expected, dict) else expected.copy()
    )

    # 2. Reset index if needed
    if not check_index:
        df_actual = df_actual.reset_index(drop=True)
        df_expected = df_expected.reset_index(drop=True)

    # 3. Print for debug
    print("\n--- Actual Result ---")
    print(df_actual.to_markdown(index=False))
    print("\n--- Expected Result ---")
    print(df_expected.to_markdown(index=False))

    # 4. Check numeric columns with np.allclose
    numeric_cols = df_actual.select_dtypes(include=[np.number]).columns.intersection(
        df_expected.select_dtypes(include=[np.number]).columns
    )

    for col in numeric_cols:
        if not np.allclose(
            df_actual[col].values, df_expected[col].values, rtol=rtol, atol=atol
        ):
            raise AssertionError(f"Numeric column '{col}' not close enough.")

    # 5. Check non-numeric columns exactly
    non_numeric_cols = df_actual.columns.difference(numeric_cols)
    for col in non_numeric_cols:
        if not df_actual[col].equals(df_expected[col]):
            raise AssertionError(f"Non-numeric column '{col}' differs.")

    # 6. Optionally check dtypes
    if check_dtype:
        for col in df_actual.columns:
            if df_actual[col].dtype != df_expected[col].dtype:
                raise AssertionError(
                    f"Dtype mismatch in column '{col}': "
                    f"{df_actual[col].dtype} != {df_expected[col].dtype}"
                )
