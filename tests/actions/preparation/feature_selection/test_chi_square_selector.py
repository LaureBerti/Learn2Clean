import pytest
from hydra.utils import instantiate
from learn2clean.actions import ChiSquareSelector
from tests.utils.actions.assert_action_pipeline import assert_action_pipeline

# --- Fixtures ---


@pytest.fixture
def input_data() -> dict:
    """
    Returns a dataset designed for Chi-Square testing.
    - Feature_A: Perfect correlation with target [1, 0, 1, 0...]. High Score.
    - Feature_B: Constant (Variance=0). Low Score.
    - Feature_C: Some correlation. Medium Score.
    - Feature_D: String/Categorical (Ignored by numeric selector). Must be preserved.
    """
    return {
        "Feature_A": [10, 0, 10, 0, 10, 0],
        "Feature_B": [1, 1, 1, 1, 1, 1],
        "Feature_C": [5, 5, 1, 1, 5, 1],
        "Feature_D": ["a", "b", "c", "d", "e", "f"],
    }


@pytest.fixture
def target_y() -> list[int]:
    """Target variable perfectly correlated with Feature_A."""
    return [1, 0, 1, 0, 1, 0]


# --- Tests ---


def test_chi_square_selector_k2(input_data: dict, target_y: list[int]) -> None:
    """
    Tests ChiSquareSelector with k=2.
    Expectation:
    - Feature_A (Best) -> Kept
    - Feature_C (2nd Best) -> Kept
    - Feature_B (Worst) -> Dropped
    - Feature_D (Non-numeric) -> Kept (Pass-through behavior)
    """
    assert_action_pipeline(
        action=ChiSquareSelector(k=2),
        transform_data=input_data,
        expected_result={
            "Feature_A": input_data["Feature_A"],
            "Feature_C": input_data["Feature_C"],
            "Feature_D": input_data["Feature_D"],  # CRUCIAL: Must be preserved
        },
        fit_data=input_data,
        fit_target=target_y,
    )


def test_chi_square_selector_k1_limit(input_data: dict, target_y: list[int]) -> None:
    """
    Tests ChiSquareSelector with k=1.
    Expectation:
    - Feature_A (Best) -> Kept
    - Feature_C -> Dropped
    - Feature_B -> Dropped
    - Feature_D -> Kept
    """
    assert_action_pipeline(
        action=ChiSquareSelector(k=1),
        transform_data=input_data,
        expected_result={
            "Feature_A": input_data["Feature_A"],
            "Feature_D": input_data["Feature_D"],  # CRUCIAL: Must be preserved
        },
        fit_data=input_data,
        fit_target=target_y,
    )


def test_chi_square_hydra_instantiation(input_data: dict, target_y: list[int]) -> None:
    """Tests instantiation via Hydra dict config works as expected."""
    action: ChiSquareSelector = instantiate(
        {
            "_target_": "learn2clean.actions.ChiSquareSelector",
            "k": 1,
        }
    )

    assert_action_pipeline(
        action=action,
        transform_data=input_data,
        expected_result={
            "Feature_A": input_data["Feature_A"],
            "Feature_D": input_data["Feature_D"],
        },
        fit_data=input_data,
        fit_target=target_y,
    )
