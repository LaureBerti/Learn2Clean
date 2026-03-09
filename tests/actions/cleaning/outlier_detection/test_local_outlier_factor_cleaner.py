from learn2clean.actions import LocalOutlierFactorCleaner
from tests.utils.actions.assert_action_pipeline import assert_action_pipeline


def test_lof_stateful_novelty_detection() -> None:
    """
    Crucial test: Verifies that the model learns the distribution from 'fit_data'
    and uses it to evaluate 'transform_data' (Novelty Detection).
    """
    # Train: A very dense cluster around 10
    train_data = {"Val": [10.0, 10.1, 9.9, 10.0, 10.05]}

    # Test: A normal value (10.0) and a distinct outlier (100.0)
    test_data = {"Val": [10.0, 100.0]}

    assert_action_pipeline(
        LocalOutlierFactorCleaner(n_neighbors=4, contamination="auto", method="mask"),
        transform_data=test_data,
        expected_result={
            "Val": [10.0, None]
        },  # 100.0 is masked because it is an anomalous novelty
        fit_data=train_data,  # Learning happens here
    )


def test_lof_ignores_non_numeric_columns() -> None:
    """
    Verifies that non-numeric (text/categorical) columns are left intact
    and do not influence the cleaning process.
    """
    data = {
        "Numeric": [1, 1, 1, 1, 1, 1, 100],  # 100 is the outlier
        "Category": ["A", "B", "C", "D", "E", "F", "G"],  # Must remain intact
    }

    assert_action_pipeline(
        LocalOutlierFactorCleaner(n_neighbors=3, method="mask"),
        transform_data=data,
        expected_result={
            "Numeric": [1, 1, 1, 1, 1, 1, None],  # 100 is the outlier
            "Category": ["A", "B", "C", "D", "E", "F", "G"],  # Must remain intact
        },
        # No fit_data = Stateless mode (Standard Outlier Detection)
    )


def test_lof_sensitivity_to_neighbors() -> None:
    """
    Verifies that detection results depend on the neighborhood size (n_neighbors).

    Here, 20.0 is a local outlier relative to a dense cluster (1, 1, 1),
    which should be detected when the neighborhood is sufficient.
    """
    data = {"A": [1, 1.1, 0.9, 1.0, 1, 1.1, 0.9, 20.0]}

    # Case 1: Sufficient neighborhood (4) -> 20.0 is compared to all 1.x values -> Detected
    assert_action_pipeline(
        LocalOutlierFactorCleaner(n_neighbors=4, method="mask"),
        transform_data=data,
        expected_result={"A": [1, 1.1, 0.9, 1.0, 1, 1.1, 0.9, None]},
    )


def test_lof_stateless_high_contamination() -> None:
    """
    Verifies aggressive cleaning in stateless mode (fitless) with high contamination.

    Forcing contamination to 50% (0.5) means the top 50% most distant values
    (2 out of 5) should be removed.
    """
    data = {"A": [10, 10, 10, 10, 10, 50, 100]}  # 50 and 100 are the most distant

    assert_action_pipeline(
        LocalOutlierFactorCleaner(contamination=0.5, n_neighbors=3, method="drop"),
        transform_data=data,
        expected_result={"A": [10, 10, 10, 10, 10]},  # Only the main cluster remains
    )


def test_lof_removes_top_percentage_outliers() -> None:
    """
    Verifies that setting a specific contamination level (e.g., 0.1) correctly
    removes the top 10% of outliers using the 'drop' method.
    """
    assert_action_pipeline(
        LocalOutlierFactorCleaner(contamination=0.1, n_neighbors=3, method="drop"),
        {
            "Feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
            "Feature2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
            "ID": list(range(10)),
        },
        {
            "Feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "Feature2": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "ID": list(range(9)),
        },
    )


def test_lof_keeps_clean_data_auto_contamination() -> None:
    """
    Verifies that LOF with 'auto' contamination preserves data that is
    already clean and uniform.
    """
    assert_action_pipeline(
        LocalOutlierFactorCleaner(contamination="auto", method="drop"),
        {
            "F1": [10, 11, 12, 13, 14],
            "F2": [10, 11, 12, 13, 14],
            "Text": ["a", "b", "c", "d", "e"],
        },
        {
            "F1": [10, 11, 12, 13, 14],
            "F2": [10, 11, 12, 13, 14],
            "Text": ["a", "b", "c", "d", "e"],
        },
    )


def test_lof_removes_high_contamination() -> None:
    """
    Verifies behavior with very high contamination (0.4) on multiple columns,
    ensuring consistent row dropping.
    """
    assert_action_pipeline(
        LocalOutlierFactorCleaner(contamination=0.4, n_neighbors=5, method="drop"),
        {
            "X": [1, 2, 3, 4, 5, 6, 10, 50, 100, 200],
            "Y": [1, 2, 3, 4, 5, 6, 10, 50, 100, 200],
        },
        {
            "X": [1, 2, 3, 4, 5, 6],
            "Y": [1, 2, 3, 4, 5, 6],
        },
    )
