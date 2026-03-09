from hydra.utils import instantiate

from learn2clean.actions import ExactDeduplicator
from tests.utils.actions.assert_action_pipeline import assert_action_pipeline


def test_action_defaults() -> None:
    """
    Tests exact duplicate removal across all columns with keep='first' (default).
    Rows 3 and 5 are duplicates of rows 0 and 2, respectively, and are dropped.
    """

    assert_action_pipeline(
        ExactDeduplicator(),
        {
            "ID": [1, 2, 3, 1, 4, 3],
            "Value": ["A", "B", "C", "A", "D", "C"],
            "Flag": [
                True,
                False,
                True,
                True,
                False,
                True,
            ],
        },
        {
            "ID": [1, 2, 3, 4],
            "Value": ["A", "B", "C", "D"],
            "Flag": [True, False, True, False],
        },
    )


def test_action_params_keep_last() -> None:
    """
    Tests duplicate removal using the parameter keep='last'.
    Rows 0 and 2 are dropped in favor of rows 3 and 5.
    """

    assert_action_pipeline(
        ExactDeduplicator(keep="last"),
        {
            "ID": [1, 2, 3, 1, 4, 3],
            "Value": ["A", "B", "C", "A", "D", "C"],
            "Flag": [
                True,
                False,
                True,
                True,
                False,
                True,
            ],  # Index 5 Flag can change for subset test
        },
        {
            "ID": [2, 1, 4, 3],
            "Value": ["B", "A", "D", "C"],
            "Flag": [False, True, False, True],
        },
    )


def test_action_params_subset_columns() -> None:
    """
    Tests duplicate removal based only on a subset of columns ('ID' and 'Value'),
    ignoring the 'Flag' column for the check.
    """
    # Use data where Flag differs at index 2 vs 5, but ID/Value is the same.
    # This proves the subset logic works independently of ignored columns.

    assert_action_pipeline(
        ExactDeduplicator(columns=["ID", "Value"]),
        {
            "ID": [1, 2, 3, 1, 4, 3],
            "Value": ["A", "B", "C", "A", "D", "C"],
            "Flag": [
                True,
                False,
                False,
                True,
                False,
                True,
            ],
        },
        {
            "ID": [1, 2, 3, 4],
            "Value": ["A", "B", "C", "D"],
            "Flag": [True, False, False, False],
        },
    )


def test_action_with_hydra_config() -> None:
    """
    Tests instantiation via Hydra dict configuration, passing the 'keep' parameter.
    """

    # Arrange: Instantiation via dict config (setting keep='last')
    action: ExactDeduplicator = instantiate(
        {
            "_target_": "learn2clean.actions.ExactDeduplicator",
            "keep": "last",
            "columns": ["ID", "Value"],  # Also set subset columns
        }
    )

    assert_action_pipeline(
        action,
        {
            "ID": [1, 2, 3, 1, 4, 3],
            "Value": ["A", "B", "C", "A", "D", "C"],
            "Flag": [
                True,
                False,
                True,
                True,
                False,
                True,
            ],
        },
        {
            "ID": [2, 1, 4, 3],
            "Value": ["B", "A", "D", "C"],
            "Flag": [False, True, False, True],
        },
    )
