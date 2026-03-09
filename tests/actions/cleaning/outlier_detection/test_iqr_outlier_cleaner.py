import pandas as pd

from learn2clean.actions import IQROutlierCleaner
from tests.utils.actions.assert_action_pipeline import assert_action_pipeline


def test_iqr_mask_method():
    """Test replacing outliers with NaN using IQR."""
    # Data: [10, 11, 12, 13, 14, 100]
    # Stats (Pandas linear): Q1=11.25, Q3=13.75, IQR=2.5
    # Upper Bound = 13.75 + (1.5 * 2.5) = 17.5
    # Lower Bound = 11.25 - (1.5 * 2.5) = 7.5
    # 100 is > 17.5, so it is an outlier.

    data = {"A": [10, 11, 12, 13, 14, 100], "B": [1, 1, 1, 1, 1, 1]}

    assert_action_pipeline(
        IQROutlierCleaner(factor=1.5, method="mask"),
        data,
        {"A": [10, 11, 12, 13, 14, None], "B": [1, 1, 1, 1, 1, 1]},
        data,
    )


def test_iqr_drop_method():
    """Test dropping rows containing outliers."""
    data = {"A": [10, 11, 12, 13, 14, 100], "B": [1, 1, 1, 1, 1, 1]}

    assert_action_pipeline(
        IQROutlierCleaner(factor=1.5, method="drop"),
        data,
        {"A": [10, 11, 12, 13, 14], "B": [1, 1, 1, 1, 1]},
        data,
    )


def test_iqr_clip_method():
    """Test clipping outliers to the calculated bounds."""
    # Based on calc above, Upper Bound is 17.5
    data = {"A": [10, 11, 12, 13, 14, 100], "B": [1, 1, 1, 1, 1, 1]}

    assert_action_pipeline(
        IQROutlierCleaner(factor=1.5, method="clip"),
        data,
        {"A": [10, 11, 12, 13, 14, 17.5], "B": [1, 1, 1, 1, 1, 1]},
        data,
    )


def test_iqr_default_masking_multiple_columns() -> None:
    """Test that the cleaner handles multiple columns correctly by masking outliers."""
    # Replaces the old 'removes_rows_above_ratio_threshold' test.
    # Checks that independent outliers in ColA, ColB, ColC are masked.
    assert_action_pipeline(
        IQROutlierCleaner(method="mask"),  # Explicit method for clarity
        {
            "ColA": [1, 2, 5, 8, 9, 100],
            "ColB": [1, 2, 5, 8, 9, 200],
            "ColC": [10, 20, 50, 80, 90, 500],
            "ColD": [5, 5, 5, 5, 5, 5],
            "Category": ["a", "b", "c", "d", "e", "f"],
        },
        {
            "ColA": [1, 2, 5, 8, 9, None],
            "ColB": [1, 2, 5, 8, 9, None],
            "ColC": [10, 20, 50, 80, 90, None],
            "ColD": [5, 5, 5, 5, 5, 5],
            "Category": ["a", "b", "c", "d", "e", "f"],
        },
    )


def test_iqr_high_factor_tolerates_extreme_values():
    """Test that a high 'factor' makes the cleaner permissive."""
    data = {
        "ColA": [1, 2, 5, 8, 9, 100],
        "ColB": [1, 2, 5, 8, 9, 200],
        "ColC": [10, 20, 50, 80, 90, 500],
        "ColD": [5, 5, 5, 5, 5, 5],
        "Category": ["a", "b", "c", "d", "e", "f"],
    }

    assert_action_pipeline(
        IQROutlierCleaner(factor=40.0, method="mask"),
        data,
        data,  # Input equals Output because factor is huge
        fit_data=data,
    )


def test_iqr_handles_no_numeric_columns() -> None:
    """Test robustness when no numeric columns are present."""
    data = {"Text": ["a", "b"], "Date": pd.to_datetime(["2023-01-01", "2023-01-02"])}

    assert_action_pipeline(
        IQROutlierCleaner(),
        data,
        data,
        fit_data=data,
    )
