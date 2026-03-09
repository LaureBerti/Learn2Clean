from hydra.utils import instantiate

from learn2clean.actions import LinearCorrelationSelector
from tests.utils.actions.assert_action_pipeline import assert_action_pipeline


def test_action_selection_defaults() -> None:
    assert_action_pipeline(
        LinearCorrelationSelector(k=2),
        {
            "Feature_A": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Feature_B": [1.0, 2.5, 3.0, 4.5, 5.0],
            "Feature_C": [50.0, 49.0, 51.0, 50.0, 52.0],
            "Feature_D": ["x", "y", "z", "a", "b"],
        },
        {
            "Feature_A": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Feature_B": [1.0, 2.5, 3.0, 4.5, 5.0],
            "Feature_D": ["x", "y", "z", "a", "b"],
        },
        fit_data={
            "Feature_A": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Feature_B": [1.0, 2.5, 3.0, 4.5, 5.0],
            "Feature_C": [50.0, 49.0, 51.0, 50.0, 52.0],
            "Feature_D": ["x", "y", "z", "a", "b"],
        },
        fit_target=[1.0, 2.0, 3.0, 4.0, 5.0],
    )


def test_linear_correlation_selector_selection() -> None:
    target_y = [10, 20, 30, 40, 50, 60]

    input_data = {
        "Feature_A": [1, 2, 3, 4, 5, 6],
        "Feature_B": [10, 10, 10, 10, 10, 10],
        "Feature_C": [6, 5, 4, 3, 2, 1],
        "Feature_D": ["a", "b", "c", "d", "e", "f"],
    }

    expected_output_data = {
        "Feature_A": [1, 2, 3, 4, 5, 6],
        "Feature_C": [6, 5, 4, 3, 2, 1],
        "Feature_D": ["a", "b", "c", "d", "e", "f"],
    }

    assert_action_pipeline(
        LinearCorrelationSelector(k=2),
        input_data,
        expected_output_data,
        input_data,
        target_y,
    )


def test_action_params_k_limit() -> None:
    """Tests feature selection with a specific k value (k=1)."""

    assert_action_pipeline(
        LinearCorrelationSelector(k=1),
        {
            "Feature_A": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Feature_B": [1.0, 2.5, 3.0, 4.5, 5.0],
            "Feature_C": [50.0, 49.0, 51.0, 50.0, 52.0],
            "Feature_D": ["x", "y", "z", "a", "b"],
        },
        {
            "Feature_A": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Feature_D": ["x", "y", "z", "a", "b"],
        },
        {
            "Feature_A": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Feature_B": [1.0, 2.5, 3.0, 4.5, 5.0],
            "Feature_C": [50.0, 49.0, 51.0, 50.0, 52.0],
            "Feature_D": ["x", "y", "z", "a", "b"],
        },
        [1, 2, 3, 4, 5],
    )


def test_action_params_columns_selection() -> None:
    """
    Tests selection when the feature search space is restricted
    by the base class `columns` parameter.
    """
    assert_action_pipeline(
        LinearCorrelationSelector(
            k=1,
            columns=[
                "Feature_B",
                "Feature_C",
                "Target",
                "Feature_D",
            ],  # A is excluded from search
        ),
        {
            "Feature_A": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Feature_B": [1.0, 2.5, 3.0, 4.5, 5.0],
            "Feature_C": [50.0, 49.0, 51.0, 50.0, 52.0],
            "Feature_D": ["x", "y", "z", "a", "b"],
        },
        {
            "Feature_A": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Feature_B": [1.0, 2.5, 3.0, 4.5, 5.0],
            "Feature_D": ["x", "y", "z", "a", "b"],
        },
        {
            "Feature_A": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Feature_B": [1.0, 2.5, 3.0, 4.5, 5.0],
            "Feature_C": [50.0, 49.0, 51.0, 50.0, 52.0],
            "Feature_D": ["x", "y", "z", "a", "b"],
        },
        [1, 2, 3, 4, 5],
    )


def test_action_params_exclude_feature() -> None:
    """
    Tests feature selection when a highly correlated feature is explicitly
    excluded using the base class `exclude` parameter.
    """
    assert_action_pipeline(
        LinearCorrelationSelector(k=1, exclude=["Feature_A", "Feature_D"]),
        {
            "Feature_A": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Feature_B": [1.0, 2.5, 3.0, 4.5, 5.0],
            "Feature_C": [50.0, 49.0, 51.0, 50.0, 52.0],
            "Feature_D": ["x", "y", "z", "a", "b"],
        },
        {
            "Feature_A": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Feature_B": [1.0, 2.5, 3.0, 4.5, 5.0],
            "Feature_D": ["x", "y", "z", "a", "b"],
        },
        {
            "Feature_A": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Feature_B": [1.0, 2.5, 3.0, 4.5, 5.0],
            "Feature_C": [50.0, 49.0, 51.0, 50.0, 52.0],
            "Feature_D": ["x", "y", "z", "a", "b"],
        },
        [1, 2, 3, 4, 5],
    )


def test_action_with_hydra_config() -> None:
    """Tests instantiation via Hydra dict configuration, ensuring proper parameter passing."""
    action: LinearCorrelationSelector = instantiate(
        {
            "_target_": "learn2clean.actions.LinearCorrelationSelector",
            "k": 1,
        }
    )
    assert_action_pipeline(
        action,
        {
            "Feature_A": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Feature_B": [1.0, 2.5, 3.0, 4.5, 5.0],
            "Feature_C": [50.0, 49.0, 51.0, 50.0, 52.0],
            "Feature_D": ["x", "y", "z", "a", "b"],
        },
        {
            "Feature_A": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Feature_D": ["x", "y", "z", "a", "b"],
        },
        {
            "Feature_A": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Feature_B": [1.0, 2.5, 3.0, 4.5, 5.0],
            "Feature_C": [50.0, 49.0, 51.0, 50.0, 52.0],
            "Feature_D": ["x", "y", "z", "a", "b"],
        },
        [1, 2, 3, 4, 5],
    )
