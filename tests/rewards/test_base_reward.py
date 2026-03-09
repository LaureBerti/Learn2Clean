import pandas as pd
import pytest

from learn2clean.rewards.base_reward import BaseReward
from learn2clean.types import Features, OptionalTarget


class ConcreteReward(BaseReward):
    """
    A concrete and minimal implementation of BaseReward
    to allow instantiation.
    """

    def reset(self):
        pass

    def __call__(self, X: Features, y: OptionalTarget) -> float:
        return 0.0


class IncompleteReward(BaseReward):
    """
    A class that fails to implement abstract methods.
    """

    pass  # Defines neither reset nor __call__


# --- 2. Unit Tests ---


def test_cannot_instantiate_abstract_class():
    """Verifies that BaseReward cannot be instantiated directly."""
    with pytest.raises(TypeError) as excinfo:
        BaseReward(None, None)

    # Check that the error mentions the missing abstract methods
    assert "Can't instantiate abstract class" in str(excinfo.value)


def test_cannot_instantiate_incomplete_subclass():
    """Verifies that a subclass must implement ALL abstract methods."""
    with pytest.raises(TypeError):
        IncompleteReward(None, None)


def test_init_copies_dataframe():
    """
    CRITICAL TEST: Verifies that __init__ correctly COPIES the data.
    If the source is modified after init, the Reward object should not change.
    """
    # 1. Source data
    original_X = pd.DataFrame({"col1": [1, 2, 3]})
    original_y = pd.Series([0, 0, 1])

    # 2. Instantiation
    reward = ConcreteReward(original_X, original_y)

    # 3. Sneaky modification of original data
    original_X.iloc[0, 0] = 999
    original_y.iloc[0] = 999

    # 4. Verification: Internal Reward data must remain intact
    assert reward.initial_X.iloc[0, 0] == 1  # Still 1, not 999
    assert reward.initial_y.iloc[0] == 0  # Still 0, not 999

    # Memory identity check (they are not the same objects)
    assert reward.initial_X is not original_X
    assert reward.initial_y is not original_y


def test_init_handles_none():
    """Verifies that the constructor handles None correctly (no crash)."""
    reward = ConcreteReward(None, None)

    assert reward.initial_X is None
    assert reward.initial_y is None


def test_init_handles_objects_without_copy():
    """
    Verifies behavior when passing an object without a .copy() method
    (Code should fallback and keep the reference as is).
    """

    class SimpleObject:
        pass

    obj = SimpleObject()

    # Pass this object (which has no .copy())
    reward = ConcreteReward(obj, None)

    # Since there is no copy method, it stores the reference
    assert reward.initial_X is obj


def test_concrete_implementation_works():
    """Verifies that the concrete class works as expected."""
    df = pd.DataFrame({"A": [1]})
    reward = ConcreteReward(df, None)

    # Can call reset without error
    reward.reset()

    # Can call the instance
    score = reward(df, None)
    assert score == 0.0
