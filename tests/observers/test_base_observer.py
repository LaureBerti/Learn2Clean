import numpy as np
import pandas as pd
import pytest
from gymnasium import spaces

from learn2clean.observers.base_observer import BaseObserver
from learn2clean.types import Features, OptionalTarget


# --- 1. Mocks for Testing ---


class ValidObserver(BaseObserver):
    """
    A valid concrete implementation of BaseObserver.
    It implements all abstract methods.
    """

    def get_observation_space(self) -> spaces.Space:
        return spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def observe(
        self,
        X: Features,
        y: OptionalTarget,
        action_history: np.ndarray | None = None,
    ) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)


class IncompleteObserver(BaseObserver):
    """
    An invalid implementation that misses the 'observe' method.
    """

    def get_observation_space(self) -> spaces.Space:
        return spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    # Missing: observe()


# --- 2. Unit Tests ---


def test_cannot_instantiate_abstract_base_class():
    """
    Verifies that BaseObserver cannot be instantiated directly
    because it contains abstract methods.
    """
    with pytest.raises(TypeError) as excinfo:
        BaseObserver(n_actions=5)

    # Ensure the error message mentions the abstract methods
    msg = str(excinfo.value)
    assert "Can't instantiate abstract class" in msg
    assert "get_observation_space" in msg
    assert "observe" in msg


def test_cannot_instantiate_incomplete_subclass():
    """
    Verifies that a subclass that does not implement all abstract methods
    raises a TypeError upon instantiation.
    """
    with pytest.raises(TypeError) as excinfo:
        IncompleteObserver(n_actions=5)

    assert "Can't instantiate abstract class" in str(excinfo.value)
    assert "observe" in str(excinfo.value)


def test_initialization_attributes():
    """
    Verifies that the __init__ method correctly sets the attributes,
    specifically 'n_actions'.
    """
    # Case 1: Default value
    obs_default = ValidObserver()
    assert obs_default.n_actions == 0

    # Case 2: Custom value
    obs_custom = ValidObserver(n_actions=10)
    assert obs_custom.n_actions == 10


def test_valid_implementation_workflow():
    """
    Verifies that a correctly implemented subclass works as expected.
    """
    observer = ValidObserver(n_actions=3)

    # Test 1: get_observation_space
    space = observer.get_observation_space()
    assert isinstance(space, spaces.Box)
    assert space.shape == (1,)

    # Test 2: observe
    # Mock data
    df = pd.DataFrame({"A": [1, 2]})
    y = pd.Series([0, 1])
    history = np.array([1, 0, 0])

    observation = observer.observe(df, y, action_history=history)

    assert isinstance(observation, np.ndarray)
    assert observation.shape == (1,)
    assert np.all(observation == 0.0)
