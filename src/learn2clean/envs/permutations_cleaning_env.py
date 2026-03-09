from typing import Any, Literal

import gymnasium as gym
import numpy as np

from learn2clean.actions.data_frame_action import DataFrameAction
from learn2clean.observers.base_observer import BaseObserver
from learn2clean.rewards.base_reward import BaseReward
from learn2clean.spaces.permutation_space import PermutationSpace
from learn2clean.types import Features, Target


class PermutationsCleaningEnv(gym.Env):
    """
    Gymnasium Environment for 'Learn2Clean' using a Combinatorial Action Space.

    Unlike a sequential environment where the agent builds the pipeline step-by-step,
    this environment operates in a 'One-Shot' fashion:
    1. The agent selects a single integer index.
    2. The environment decodes this index into a full ordered sequence of actions (Pipeline).
    3. The pipeline is executed entirely.
    4. The reward is computed based on the final result.
    5. The episode terminates immediately.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self,
        X: Features,
        y: Target,
        actions: list[DataFrameAction],
        reward_calculator: BaseReward,
        observer: BaseObserver,
        render_mode: Literal["human", "ansi"] | None = None,
        penalty_error: float = -1.0,
    ) -> None:
        """
        Initializes the One-Shot Permutation Environment.

        Args:
            X: The initial feature set (DataFrame).
            y: The target variable.
            actions: List of available atomic DataFrameAction objects.
            reward_calculator: An instance of BaseReward to compute data utility.
            render_mode: The mode for rendering ('human', 'ansi', or None).
            penalty_error: The penalty applied when an action raises an exception.
        """
        super().__init__()

        # --- Data Management ---
        # Store the initial dataset to allow the environment to reset later.
        self.initial_X = X
        self.initial_y = y

        # Working copies (reset at each episode)
        self.current_X = X.copy()
        self.current_y = y.copy() if hasattr(y, "copy") else y

        self.actions = actions
        self.reward_calculator = reward_calculator

        # --- Observer Setup ---
        self.observer = observer
        # For permutation space, 'n_actions' technically refers to atomic actions available
        self.observer.n_actions = len(actions)

        self.render_mode = render_mode
        self.penalty_error = penalty_error

        # --- Spaces Definition ---

        # 1. Action Space: The Combinatorial Space
        # The agent selects one index representing a full pipeline.
        # Note: We enforce a hard limit of 20 actions inside PermutationSpace for safety.
        self.action_space: PermutationSpace = PermutationSpace(actions)

        # 2. Observation Space:
        self.observation_space = self.observer.get_observation_space()

        # --- Internal State ---
        self.baseline_score: float = 0.0
        self.last_pipeline_str: str = "Start"
        self.last_score: float = 0.0
        self.last_status: Literal["Start", "Success", "Failed"] = "Start"

    def _get_observation(self) -> Any:
        """Delegates observation. No history needed for One-Shot."""
        return self.observer.observe(self.current_X, self.current_y)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Resets the environment.
        """
        super().reset(seed=seed)

        # Propagate seed to the custom space (Important!)
        if seed is not None:
            self.action_space.seed(seed)

        # Restore Data
        self.current_X = self.initial_X.copy()
        self.current_y = (
            self.initial_y.copy() if hasattr(self.initial_y, "copy") else self.initial_y
        )

        # Reset Reward State (Crucial for stateful rewards!)
        self.reward_calculator.reset()

        # Calculate initial baseline (score on dirty data)
        self.baseline_score = self.reward_calculator(self.current_X, self.current_y)
        self.last_score = self.baseline_score
        self.last_pipeline_str = "Start"
        self.last_status = "Start"

        return self._get_observation(), {"initial_score": self.baseline_score}

    def step(
        self, action_idx: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Executes the full pipeline corresponding to action_idx.
        """
        # 1. Decode the integer index into a sequence of action objects
        # This uses the O(k) Unranking algorithm from PermutationSpace
        try:
            pipeline = self.action_space.idx_to_permutation(action_idx)
            self.last_pipeline_str = " -> ".join([a.name for a in pipeline])
        except IndexError:
            # Should not happen if the agent respects the space, but good for safety
            return (
                self._get_observation(),
                self.penalty_error,
                True,
                False,
                {"error": "Invalid Action Index"},
            )

        info_msg = "Success"
        error_encountered = False

        # --- EXECUTION LOOP ---
        # We apply the actions one by one on the working copy
        X_temp = self.current_X.copy()

        try:
            for action in pipeline:
                action.fit(X_temp, self.current_y)
                X_temp = action.transform(X_temp)
                if X_temp.empty:
                    raise ValueError("Pipeline resulted in empty DataFrame")

            # --- SCORING ---
            # If pipeline succeeded, we evaluate the final result
            final_score = self.reward_calculator(X_temp, self.current_y)

            # Option A: Absolute Reward (Simple for Bandits)
            reward = final_score

            # Option B: Delta Reward (Better for RL)
            # reward = final_score - self.baseline_score

            # Update state (technically useless for One-Shot, but consistent)
            self.current_X = X_temp
            self.last_score = final_score
            self.last_status = "Success"

        except Exception as e:
            reward = self.penalty_error
            info_msg = str(e)
            error_encountered = True
            self.last_status = "Failed"

        # --- TERMINATION ---
        # In a Permutation/One-Shot environment, the episode ends immediately after the sequence.
        terminated = True
        truncated = False

        info = {
            "pipeline": [str(a) for a in pipeline],
            "pipeline_length": len(pipeline),
            "msg": info_msg,
            "error": error_encountered,
            "baseline_score": self.baseline_score,
        }

        return self._get_observation(), reward, terminated, truncated, info

    def render(self) -> str | None:
        """Renders the One-Shot result using the same visual style as SequentialEnv."""
        if self.render_mode is None:
            return None

        # Colors
        RESET, BOLD, RED, GREEN, YELLOW, BLUE, CYAN = (
            "\033[0m",
            "\033[1m",
            "\033[91m",
            "\033[92m",
            "\033[93m",
            "\033[94m",
            "\033[96m",
        )

        # Logic for colors
        status = self.last_status
        status_color = (
            GREEN if status == "Success" else RED if status == "Failed" else CYAN
        )
        score_color = GREEN if self.last_score > 0.7 else YELLOW
        if self.last_score < 0.3:
            score_color = RED

        title = f"{BOLD}One-Shot Pipeline Result{RESET}"
        separator = "-" * 60
        pipe_str = self.last_pipeline_str

        lines = [
            separator,
            title,
            separator,
            f"{CYAN}{'Metric':<20}{RESET} | {BOLD}{'Value'}{RESET}",
            separator,
            f"{'Pipeline':<20} | {BLUE}{pipe_str}{RESET}",
            f"{'Status':<20} | {status_color}{status}{RESET}",
            f"{'Final Score':<20} | {score_color}{self.last_score:.4f}{RESET}",
            f"{'Baseline':<20} | {self.baseline_score:.4f}",
            separator,
        ]

        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output)
            return None
        return output
