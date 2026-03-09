import numpy as np
import pandas as pd
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from learn2clean.actions.data_frame_action import DataFrameAction
from learn2clean.envs.sequential_cleaning_env import SequentialCleaningEnv
from learn2clean.observers.data_stats_observer import DataStatsObserver
from learn2clean.rewards.base_reward import BaseReward
from learn2clean.types import Features, Target, OptionalTarget


# --- 1. Mocks & Fixtures for Isolation ---


class MockReward(BaseReward):
    """A simple deterministic reward: returns the mean of the dataframe."""

    def __init__(self, initial_X: Features, initial_y: Target):
        super().__init__(initial_X, initial_y)
        self.reset_called_count = 0

    def reset(self) -> None:
        self.reset_called_count += 1

    def __call__(self, X: Features, y: Target) -> float:
        # Simple metric: mean of all values.
        # If X is all 0s -> 0.0. If X is all 1s -> 1.0
        if X.empty:
            return 0.0
        # Check if numeric to avoid errors
        if hasattr(X, "select_dtypes"):
            nums = X.select_dtypes(include=[np.number])
            if nums.empty:
                return 0.0
            return float(nums.values.mean())
        return 0.0


class AddOneAction(DataFrameAction):
    """Action that successfully adds 1 to the dataframe."""

    def __init__(self):
        super().__init__(name="Add One")

    def fit(self, X: Features, y: OptionalTarget = None) -> "DataFrameAction":
        return self

    def transform(self, X: Features) -> Features:
        return X + 1.0


class SubtractAction(DataFrameAction):
    """Action that successfully subtract 0.5 to the dataframe."""

    def __init__(self):
        super().__init__(name="Subtract One")

    def fit(self, X: Features, y: OptionalTarget = None) -> "DataFrameAction":
        return self

    def transform(self, X: Features) -> Features:
        return X - 0.5


class FailAction(DataFrameAction):
    """Action that always raises an exception."""

    def __init__(self):
        super().__init__(name="Fail Action")

    def fit(self, X: Features, y: OptionalTarget = None) -> "DataFrameAction":
        raise ValueError("Simulated Failure")

    def transform(self, X: Features) -> Features:
        return X


@pytest.fixture
def simple_data():
    """Returns a simple DataFrame and Target."""
    X = pd.DataFrame({"A": [0.0, 0.0], "B": [0.0, 0.0]})
    y = pd.Series([0, 1])
    return X, y


@pytest.fixture
def env(simple_data):
    """Instantiates the environment with mock actions, reward, and OBSERVER."""
    X, y = simple_data
    actions = [AddOneAction(), FailAction(), SubtractAction()]
    reward_calc = MockReward(X, y)

    # NEW: Instantiate the Observer
    observer = DataStatsObserver()

    return SequentialCleaningEnv(
        X=X,
        y=y,
        actions=actions,
        reward_calculator=reward_calc,
        observer=observer,  # Inject Dependency
        max_steps=5,
        render_mode="ansi",
        penalty_error=-0.5,
        penalty_repetition=-1.0,
    )


# --- 2. Unit Tests ---


def test_initialization(env):
    """Test if spaces and state are correctly initialized."""
    # 1. Check Action Space
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n == 3

    # 2. Check Observation Space (Now a Dict!)
    assert isinstance(env.observation_space, spaces.Dict)
    assert "dataset_stats" in env.observation_space.spaces
    assert "action_history" in env.observation_space.spaces

    # Check Action History shape inside the Dict
    assert env.observation_space["action_history"].shape == (3,)

    # 3. Check Internal State
    assert np.all(env.action_history == 0.0)
    assert env.current_step == 0
    assert env.last_score == 0.0


def test_gymnasium_compliance(env):
    """
    Ensure the environment complies with Gymnasium API standards.
    We skip render check as we instantiate the env directly without 'spec'.
    """
    check_env(env, skip_render_check=True)


def test_reset(env):
    """Test if reset restores the environment state."""
    # 1. Modify state manually
    env.current_step = 3
    env.action_history[0] = 1.0
    env.last_score = 99.9

    # 2. Call Reset
    obs, info = env.reset()

    # 3. Assertions
    assert env.current_step == 0
    assert np.all(env.action_history == 0.0)

    # NEW: Check observation structure
    assert isinstance(obs, dict)
    assert np.array_equal(obs["action_history"], env.action_history)

    assert info["score"] == 0.0
    assert env.reward_calculator.reset_called_count >= 1


def test_step_success_logic(env):
    """
    Test a successful step:
    - Data should change.
    - Reward should be calculated.
    """
    # Apply Action 0 (AddOneAction)
    obs, reward, terminated, truncated, info = env.step(0)

    # 1. Check Data update
    assert env.current_X.mean().mean() == 1.0

    # 2. Check Reward
    assert reward == 1.0
    assert env.last_score == 1.0

    # 3. Check State update
    assert env.current_step == 1
    assert env.action_history[0] == 1.0

    # NEW: Check observation content
    assert obs["action_history"][0] == 1.0

    assert info["msg"] == "Success"


def test_step_failure_logic(env):
    """
    Test a failed step:
    - Data should NOT change.
    - Penalty applied.
    """
    initial_mean = env.current_X.mean().mean()

    # Apply Action 1 (FailAction)
    obs, reward, terminated, truncated, info = env.step(1)

    # 1. Check Data integrity
    assert env.current_X.mean().mean() == initial_mean

    # 2. Check Reward
    assert reward == -0.5

    # 3. Check State update (History is updated even on failure)
    assert env.current_step == 1
    assert env.action_history[1] == 1.0
    assert obs["action_history"][1] == 1.0

    assert "Error" in info["msg"]


def test_step_repetition_logic(env):
    """
    Test repeating an action:
    - Penalty applied.
    - Data NOT changed.
    """
    # 1. First execution
    env.step(0)

    # 2. Second execution (Repetition)
    obs, reward, terminated, truncated, info = env.step(0)

    # Check Penalty
    assert reward == -1.0
    assert "Invalid: Repeated" in info["msg"]

    # Check Data integrity
    assert env.current_X.mean().mean() == 1.0


def test_custom_penalty_values(simple_data):
    """Test initializing with custom penalties."""
    X, y = simple_data
    actions = [FailAction()]
    reward_calc = MockReward(X, y)
    observer = DataStatsObserver()  # Need to inject observer here too

    custom_env = SequentialCleaningEnv(
        X=X,
        y=y,
        actions=actions,
        reward_calculator=reward_calc,
        observer=observer,
        penalty_error=-10.0,
        penalty_repetition=-20.0,
    )

    # Test Error Penalty
    _, reward, _, _, _ = custom_env.step(0)
    assert reward == -10.0


def test_sequential_rewards(env):
    """Test reward accumulation."""
    # Step 1: Add 1 (+1.0)
    env.step(0)
    # Step 2: Subtract 0.5 (New 0.5 - Old 1.0 = -0.5)
    _, reward2, _, _, _ = env.step(2)

    assert reward2 == -0.5
    assert env.last_score == 0.5


def test_truncation(env):
    """Test max_steps logic."""
    env.max_steps = 2
    env.reset()

    _, _, _, truncated, _ = env.step(0)
    assert not truncated

    _, _, _, truncated, _ = env.step(1)
    assert truncated


def test_render_ansi(env):
    """Test ansi render."""
    env.reset()
    env.step(0)
    output = env.render()

    assert isinstance(output, str)
    assert "Environment State" in output
    assert "Add One" in output
    assert "Utility Score" in output


def test_render_human(env):
    """
    Test render in 'human' mode.
    Uses pytest's capsys fixture to capture stdout.
    """
    env.render_mode = "human"
    env.reset()

    # Call render, which should print to stdout
    env.render()
    env.step(0)
    env.render()
    env.step(1)
    env.render()
    env.step(2)
    env.render()
    env.step(2)
    env.render()


def test_render_none(env):
    """Test render returns None if render_mode is None."""
    env.render_mode = None
    env.step(0)
    output = env.render()
    assert output is None
