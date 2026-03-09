from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from learn2clean.actions.data_frame_action import DataFrameAction
from learn2clean.observers.base_observer import BaseObserver  # <--- NOUVEAU
from learn2clean.rewards.base_reward import BaseReward
from learn2clean.types import Features, Target


class SequentialCleaningEnv(gym.Env):
    """
    A simplified Gymnasium environment designed for the Learn2Clean tutorial.

    This environment models the data cleaning process as a Markov Decision Process (MDP).
    The Agent observes the history of actions performed so far, selects a new
    transformation action (e.g., Imputation, Scaling), and receives a reward based on
    the improvement of a specific utility metric (e.g., Model Accuracy, Completeness).

    Attributes:
        action_space (spaces.Discrete): The discrete set of available cleaning actions.
        observation_space (spaces.Box): A binary vector indicating the history of
                                        applied actions (1.0 if visited, 0.0 otherwise).
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        X: Features,
        y: Target,
        actions: list[DataFrameAction],
        reward_calculator: BaseReward,
        observer: BaseObserver,
        max_steps: int = 10,
        render_mode: Literal["human", "ansi"] | None = None,
        penalty_error: float = -0.5,
        penalty_repetition: float = -1.0,
    ) -> None:
        """
        Initializes the Learn2Clean environment.

        Args:
            X: The initial feature set (DataFrame).
            y: The target variable (Series or Array).
            actions: A list of available DataFrameAction objects.
            reward_calculator: An instance of BaseReward to compute data utility.
            max_steps: The maximum number of steps allowed per episode.
            render_mode: The mode for rendering ('human', 'ansi', or None).
            penalty_error: The penalty applied when an action raises an exception.
            penalty_repetition: The penalty applied when an action is repeated.
        """
        super().__init__()

        # --- Data Management ---
        # Store the initial dataset to allow the environment to reset later.
        self.initial_X = X
        self.initial_y = y

        # These attributes track the dataset state as it evolves during the episode.
        self.current_X = X.copy()
        self.current_y = y.copy() if hasattr(y, "copy") else y

        # --- Configuration ---
        self.actions = actions
        self.reward_calculator = reward_calculator

        # --- Observer Setup ---
        self.observer = observer
        # The observer needs to know the action space size (e.g., for One-Hot encoding history)
        self.observer.n_actions = len(actions)

        self.penalty_error = penalty_error
        self.penalty_repetition = penalty_repetition
        self.max_steps = max_steps
        self.render_mode = render_mode

        # --- Spaces Definition ---

        # 1. Action Space: Discrete selection.
        # The agent chooses an index 'i' corresponding to actions[i].
        self.action_space = spaces.Discrete(len(actions))

        self.observation_space = self.observer.get_observation_space()

        # --- Internal State ---
        self.current_step = 0
        self.action_history = np.zeros(len(actions), dtype=np.float32)
        self.last_score = 0.0
        self.last_action_name: str = "Start"
        # Tracks the outcome of the last step for display purposes
        self.last_action_status: Literal["Start", "Success", "Failed", "Repeated"] = (
            "Start"
        )

    def _get_observation(self) -> Any:
        """Helper to delegate observation creation."""
        return self.observer.observe(
            X=self.current_X, y=self.current_y, action_history=self.action_history
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Resets the environment to its initial state.

        This is called at the beginning of every new episode. It restores the
        original dataset, clears the action history, and re-calculates the baseline score.

        Args:
            seed: Random seed for reproducibility (Gymnasium standard).
            options: Additional options (unused here).

        Returns:
            observation (np.ndarray): The initial observation (all zeros).
            info (dict): Dictionary containing the initial 'score'.
        """
        super().reset(seed=seed)

        # 1. Reset Data: Restore the original features and target
        self.current_X = self.initial_X.copy()
        # Note: If y is immutable or not transformed, .copy() is technically optional
        # but kept here for safety.
        self.current_y = (
            self.initial_y.copy() if hasattr(self.initial_y, "copy") else self.initial_y
        )

        # 2. Reset Internal Counters
        self.current_step = 0
        self.action_history.fill(0.0)

        # 3. Reset Reward State (Crucial for stateful rewards!)
        self.reward_calculator.reset()

        # 4. Calculate Initial Baseline
        # We compute the score on the dirty data to establish the baseline.
        self.last_score = self.reward_calculator(self.current_X, self.current_y)

        # 5. Reset Render State
        self.last_action_name = "Start"
        self.last_action_status = "Start"

        return self._get_observation(), {"score": self.last_score}

    def step(
        self, action_idx: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Executes one time step within the environment.

        The environment attempts to apply the selected action.
        - If successful: The state is updated, and reward = new_score - old_score.
        - If failed: The state remains unchanged (data-wise), and a penalty is applied.

        Args:
            action_idx (int): The index of the selected action to perform.

        Returns:
            observation (np.ndarray): The updated action history.
            reward (float): The improvement in score (or penalty).
            terminated (bool): Whether the goal is reached (False in this simple version).
            truncated (bool): Whether the episode exceeded max_steps.
            info (dict): Auxiliary info (current score, error messages).
        """
        # 1. Retrieve the Action
        action = self.actions[action_idx]
        self.last_action_name = action.name
        info_msg = "Success"

        # --- CHECK: Action already executed? ---
        if self.action_history[action_idx] > 0:
            self.last_action_status = "Repeated"
            self.current_step += 1

            return (
                self._get_observation(),
                self.penalty_repetition,
                True,
                False,
                {"score": self.last_score, "msg": "Invalid: Repeated"},
            )

        # --- EXECUTION ---
        try:
            # 1. Attempt transformation (Atomic Transaction)
            # Use a temporary variable to prevent corrupting self.current_X if fit/transform fails.
            X_temp = self.current_X.copy()

            action.fit(X_temp, self.current_y)
            X_transformed = action.transform(X_temp)

            # 2. Calculate New Absolute Score
            current_score = self.reward_calculator(X_transformed, self.current_y)

            # 3. Calculate Reward (Improvement Delta)
            # The agent is rewarded for the DIFFERENCE it creates.
            step_reward = current_score - self.last_score

            # 4. Commit Changes (Only if everything succeeded)
            self.current_X = X_transformed
            self.last_score = current_score
            self.last_action_status = "Success"

        except Exception as e:
            # CASE: Action Failed (e.g., Nan values in input, invalid column type)

            # 1. Apply Negative Reward (Penalty)
            # This teaches the agent to avoid invalid actions for the current state.
            step_reward = self.penalty_error
            info_msg = f"Error: {str(e)}"
            self.last_action_status = "Failed"

            # Note: We do NOT update self.current_X or self.last_score.
            # The agent stays in the same data state.

        # --- COMMON UPDATES ---

        # Mark the action as visited in the history vector (even if it failed)
        self.action_history[action_idx] = 1.0
        self.current_step += 1

        # --- TERMINATION CONDITIONS ---
        terminated = False
        truncated = self.current_step >= self.max_steps

        return (
            self._get_observation(),  # Return new state (dataset might have changed)
            step_reward,
            terminated,
            truncated,
            {"score": self.last_score, "msg": info_msg},
        )

    def render(self) -> str | None:
        """
        Renders the environment state using standard Python strings and ANSI codes.
        No external libraries (Rich/Tabulate) required.
        """
        if self.render_mode is None:
            return None

        # 1. Calculate Metrics
        n_rows, n_cols = self.current_X.shape
        missing_count = self.current_X.isna().sum().sum()
        missing_ratio = missing_count / self.current_X.size

        # Determine colors for visual feedback
        RESET = "\033[0m"
        BOLD = "\033[1m"
        RED = "\033[91m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        BLUE = "\033[94m"
        CYAN = "\033[96m"

        # 3. Dynamic Coloring
        score_color = GREEN if self.last_score > 0.7 else YELLOW
        if self.last_score < 0.3:
            score_color = RED

        missing_color = GREEN if missing_count == 0 else RED

        # 4. Handle Status Display
        status = self.last_action_status
        status_color = (
            GREEN
            if status == "Success"
            else RED if status == "Failed" else YELLOW if status == "Repeated" else CYAN
        )

        # 5. Table Construction
        # :<20 means "align left, padding on 20 characters"
        title = f"{BOLD}Environment State (Step {self.current_step}/{self.max_steps}){RESET}"
        separator = "-" * 45

        lines = [
            separator,
            title,
            separator,
            f"{CYAN}{'Metric':<20}{RESET} | {BOLD}{'Value'}{RESET}",
            separator,
            f"{'Last Action':<20} | {status_color}{self.last_action_name} ({status}){RESET}",
            f"{'Dataset Shape':<20} | ({n_rows}, {n_cols})",
            f"{'Missing Cells':<20} | {missing_color}{missing_count} ({missing_ratio:.2%}){RESET}",
            f"{'Utility Score':<20} | {score_color}{self.last_score:.4f}{RESET}",
            separator,
        ]

        final_output = "\n".join(lines)

        if self.render_mode == "human":
            print(final_output)
            return None
        return final_output
