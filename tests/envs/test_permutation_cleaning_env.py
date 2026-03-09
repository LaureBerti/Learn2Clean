import numpy as np
import pandas as pd
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from learn2clean.actions.data_frame_action import DataFrameAction
from learn2clean.envs.permutations_cleaning_env import PermutationsCleaningEnv
from learn2clean.observers.data_stats_observer import DataStatsObserver
from learn2clean.rewards.base_reward import BaseReward
from learn2clean.types import Features, OptionalTarget, Target


# --- 1. Mocks & Fixtures ---


class MockReward(BaseReward):
    """A deterministic reward: returns the mean of the dataframe."""

    def __init__(self, initial_X: Features, initial_y: Target):
        super().__init__(initial_X, initial_y)
        self.reset_called_count = 0

    def reset(self) -> None:
        self.reset_called_count += 1

    def __call__(self, X: Features, y: Target) -> float:
        if X.empty:
            return 0.0
        # Safe mean calculation
        if hasattr(X, "select_dtypes"):
            nums = X.select_dtypes(include=[np.number])
            if nums.empty:
                return 0.0
            return float(nums.values.mean())
        return 0.0


class AddAction(DataFrameAction):
    """Adds 1.0 to data."""

    def __init__(self):
        super().__init__(name="Add")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X + 1.0


class MultAction(DataFrameAction):
    """Multiplies data by 2.0."""

    def __init__(self):
        super().__init__(name="Mult")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X * 2.0


class FailAction(DataFrameAction):
    """Always crashes."""

    def __init__(self):
        super().__init__(name="Fail")

    def fit(self, X, y=None):
        raise ValueError("Crash!")

    def transform(self, X):
        return X


@pytest.fixture
def simple_data():
    X = pd.DataFrame({"A": [1.0, 1.0]})
    y = pd.Series([0, 0])
    return X, y


@pytest.fixture
def env(simple_data):
    X, y = simple_data
    # Actions: 0->Add, 1->Mult, 2->Fail
    actions = [AddAction(), MultAction(), FailAction()]

    return PermutationsCleaningEnv(
        X=X,
        y=y,
        actions=actions,
        reward_calculator=MockReward(X, y),
        observer=DataStatsObserver(),
        penalty_error=-5.0,
        render_mode="ansi",
    )


# --- 2. Unit Tests ---


def test_initialization(env):
    """Verifies that spaces and internal state are correctly set up."""
    # Action Space should be Discrete (but implemented as PermutationSpace)
    assert isinstance(env.action_space, spaces.Discrete)
    # 3 items -> Total permutations = P(3,1)+P(3,2)+P(3,3) = 3+6+6 = 15
    assert env.action_space.n == 15

    # Observation Space (Dict via DataStatsObserver)
    assert isinstance(env.observation_space, spaces.Dict)
    assert "dataset_stats" in env.observation_space.spaces

    # Internal state
    assert env.baseline_score == 0.0
    assert env.last_status == "Start"

    env.reset()
    assert env.baseline_score == 1.0  # Mean of [1.0, 1.0] is 1.0
    assert env.last_status == "Start"


def test_gymnasium_compliance(env):
    """Ensures compliance with standard Gym API."""
    # We skip render check because we are not using a standard render mode
    check_env(env, skip_render_check=True)


def test_reset(env):
    """Verifies that reset restores the initial data."""
    # 1. Corrupt the state
    env.current_X = pd.DataFrame({"A": [999.9]})
    env.last_score = -100.0

    # 2. Reset
    obs, info = env.reset()

    # 3. Check restoration
    assert env.current_X.mean().mean() == 1.0
    assert info["initial_score"] == 1.0
    assert env.reward_calculator.reset_called_count >= 1


def test_step_pipeline_execution(env):
    """
    Test a valid pipeline execution.
    Target Pipeline: Add -> Mult
    Logic: (1.0 + 1.0) * 2.0 = 4.0
    """
    # We need to find the index for (Add, Mult).
    # Actions list: [0:Add, 1:Mult, 2:Fail]
    # In PermutationSpace logic, let's assume index 7 maps to (Add, Mult).
    # To be safe, we search for it dynamically in the test.

    target_idx = -1
    for i in range(env.action_space.n):
        perm = env.action_space.idx_to_permutation(i)
        if len(perm) == 2 and perm[0].name == "Add" and perm[1].name == "Mult":
            target_idx = i
            break

    assert target_idx != -1, "Could not find (Add, Mult) permutation index"

    # Execute Step
    obs, reward, terminated, truncated, info = env.step(target_idx)

    # 1. Check Data: Should be 4.0
    assert env.current_X.mean().mean() == 4.0

    # 2. Check Reward
    assert reward == 4.0

    # 3. Check Termination (One-Shot -> Always True)
    assert terminated
    assert not truncated

    # 4. Check Info
    assert info["msg"] == "Success"
    assert "Add" in info["pipeline"][0]


def test_step_failure_handling(env):
    """Test behavior when an action in the pipeline fails."""
    # Find index for single action: FailAction
    fail_idx = -1
    for i in range(env.action_space.n):
        perm = env.action_space.idx_to_permutation(i)
        if len(perm) == 1 and perm[0].name == "Fail":
            fail_idx = i
            break

    obs, reward, terminated, truncated, info = env.step(fail_idx)

    # 1. Reward should be penalty
    assert reward == -5.0

    # 2. Status
    assert info["error"] is True
    assert "Crash!" in info["msg"]
    assert terminated


def test_invalid_action_index(env):
    """Test out of bounds index."""
    invalid_idx = env.action_space.n + 100

    obs, reward, terminated, truncated, info = env.step(invalid_idx)

    assert reward == -5.0  # Penalty
    assert info["error"] == "Invalid Action Index"
    assert terminated


def test_render_ansi(env):
    """Test render output."""
    env.reset()
    output = env.render()

    assert isinstance(output, str)
    assert "One-Shot Pipeline Result" in output
    assert "Baseline" in output
