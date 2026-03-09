from learn2clean.actions import ZScoreOutlierCleaner
from tests.utils.actions.assert_action_pipeline import assert_action_pipeline


def test_z_score_mask_method():
    """Test replacing outliers with NaN."""
    # Create data with obvious outliers (mean=10, std approx small, 100 is > 3 sigma)
    data = {"A": [10, 10, 10, 10, 100], "B": [1, 2, 3, 4, 5]}

    assert_action_pipeline(
        ZScoreOutlierCleaner(threshold=1.5, method="mask"),
        data,
        {"A": [10, 10, 10, 10, None], "B": [1, 2, 3, 4, 5]},
        data,
    )


def test_z_score_drop_method():
    """Test dropping rows with outliers."""
    data = {"A": [0, 0, 0, 1000], "B": ["a", "b", "c", "d"]}  # 1000 is huge outlier

    assert_action_pipeline(
        ZScoreOutlierCleaner(method="drop", threshold=1.4),
        data,
        {
            "A": [0, 0, 0],
            "B": [
                "a",
                "b",
                "c",
            ],
        },
        data,
    )


def test_z_score_removes_rows_with_obvious_outliers() -> None:
    data = {
        "F1": [10, 11, 10, 9, 1000],
        "F2": [10, 11, 10, 9, 1000],
        "Category": ["a", "b", "c", "d", "e"],
    }

    # Note: On small data (N=5), Z-score for 1000 is approx 1.79.
    # We use threshold=1.5 to ensure detection.
    assert_action_pipeline(
        ZScoreOutlierCleaner(threshold=1.5, method="drop"),
        data,
        {
            "F1": [10, 11, 10, 9],
            "F2": [10, 11, 10, 9],
            "Category": ["a", "b", "c", "d"],
        },
        fit_data=data,
    )


def test_z_score_mask_handles_mixed_outliers() -> None:
    data = {
        "A": [10, 10, 10, 10, 1000],  # Outlier
        "B": [10, 10, 10, 10, 1000],  # Outlier
        "C": [10, 10, 10, 10, 1000],  # Outlier
        "D": [5, 6, 7, 8, 9],  # Normal (Z-score < 1.5)
        "Text": ["a", "b", "c", "d", "e"],
    }

    assert_action_pipeline(
        ZScoreOutlierCleaner(threshold=1.5, method="mask"),
        data,
        {
            # A, B, C should have the last value masked (NaN)
            "A": [10.0, 10.0, 10.0, 10.0, None],
            "B": [10.0, 10.0, 10.0, 10.0, None],
            "C": [10.0, 10.0, 10.0, 10.0, None],
            # D is preserved entirely because 9 is not an outlier
            "D": [5, 6, 7, 8, 9],
            "Text": ["a", "b", "c", "d", "e"],
        },
        fit_data=data,
    )


def test_z_score_ignores_low_variance_columns() -> None:
    data = {
        "ColA": [5, 5, 5, 5, 5],  # Variance = 0, std = 0. Should be skipped.
        "ColB": [1, 2, 3, 4, 100],  # Variance > 0. Outlier 100 detected.
        "ID": [1, 2, 3, 4, 5],
    }

    assert_action_pipeline(
        ZScoreOutlierCleaner(threshold=1.5, method="drop"),
        data,
        {
            # Row 5 dropped due to ColB. ColA didn't cause a crash.
            "ColA": [5, 5, 5, 5],
            "ColB": [1, 2, 3, 4],
            "ID": [1, 2, 3, 4],
        },
        fit_data=data,
    )
