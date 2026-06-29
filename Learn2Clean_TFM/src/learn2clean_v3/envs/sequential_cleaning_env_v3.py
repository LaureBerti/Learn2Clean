"""
SequentialCleaningEnvV3 — improved Gymnasium environment.

V3 improvements over V2
-----------------------
1. DataQualityObserver wired in by default (richer state).
2. ExplainableReward integration — action name is passed before each step.
3. Pandera schema validation after every action (optional).
4. ParameterizedAction support — env can optionally pass param dicts.
5. Configurable invalid-action penalty.
6. ``render()`` shows per-objective reward breakdown when MultiObjectiveReward is used.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import pandera as pa

from learn2clean_v3.actions.data_frame_action import DataFrameAction, DataValidationError
from learn2clean_v3.observers.base_observer import BaseObserver
from learn2clean_v3.observers.data_quality_observer import DataQualityObserver
from learn2clean_v3.rewards.base_reward import BaseReward
from learn2clean_v3.rewards.completeness_retention_reward import CompletenessRetentionReward
from learn2clean_v3.rewards.explainable_reward import ExplainableReward
from learn2clean_v3.types import ActionHistory, Features, OptionalTarget

logger = logging.getLogger(__name__)


class SequentialCleaningEnvV3(gym.Env):
    """
    Parameters
    ----------
    X : Features
        Initial DataFrame (will be copied, never mutated).
    y : OptionalTarget
        Labels (may be None for unsupervised tasks).
    actions : list[DataFrameAction]
        Ordered list of available cleaning actions.
    reward_fn : BaseReward | None
        Reward function.  Defaults to CompletenessRetentionReward.
    observer : BaseObserver | None
        State observer.  Defaults to DataQualityObserver.
    max_steps : int
        Episode length cap.
    invalid_action_penalty : float
        Reward given when an action raises an exception.
    pandera_schema : pa.DataFrameSchema | None
        Global schema applied after every action.
    allow_repeated_actions : bool
        If False, re-applying the same action in an episode gives the penalty.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        X: Features,
        y: OptionalTarget,
        actions: List[DataFrameAction],
        reward_fn: Optional[BaseReward] = None,
        observer: Optional[BaseObserver] = None,
        max_steps: int = 10,
        invalid_action_penalty: float = -0.05,
        pandera_schema: Optional[pa.DataFrameSchema] = None,
        allow_repeated_actions: bool = False,
    ) -> None:
        super().__init__()

        if len(actions) == 0:
            raise ValueError("At least one action is required.")

        self._X_orig = X.copy()
        self._y_orig = copy.deepcopy(y)
        self._actions = actions
        self._reward_fn = reward_fn or CompletenessRetentionReward()
        self._observer = observer or DataQualityObserver()
        self._max_steps = max_steps
        self._penalty = invalid_action_penalty
        self._schema = pandera_schema
        self._allow_repeated = allow_repeated_actions

        # Runtime state
        self._current_X: Features = X.copy()
        self._current_y: OptionalTarget = copy.deepcopy(y)
        self._action_history: ActionHistory = []
        self._step_count: int = 0

        # Set reference distribution in observer once
        if isinstance(self._observer, DataQualityObserver):
            self._observer.set_reference(self._X_orig)

        # Gymnasium spaces
        n = len(self._actions)
        self.action_space = gym.spaces.Discrete(n)
        self.observation_space = self._observer.observation_space(n)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self._current_X = self._X_orig.copy()
        self._current_y = copy.deepcopy(self._y_orig)
        self._action_history = []
        self._step_count = 0

        for action in self._actions:
            action.reset()

        self._reward_fn.reset(self._current_X, self._current_y)

        # Re-register reference distribution
        if isinstance(self._observer, DataQualityObserver):
            self._observer.set_reference(self._X_orig)

        obs = self._get_obs()
        return obs, {}

    def step(
        self, action_idx: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = self._actions[action_idx]
        info: Dict[str, Any] = {"action": action.name, "step": self._step_count}

        # Repeated-action guard
        if not self._allow_repeated and action_idx in self._action_history:
            self._step_count += 1
            obs = self._get_obs()
            terminated = self._step_count >= self._max_steps
            return obs, self._penalty, terminated, False, {**info, "reason": "repeated_action"}

        # Apply action
        try:
            # Inform ExplainableReward of upcoming action name
            if isinstance(self._reward_fn, ExplainableReward):
                self._reward_fn.set_action_name(action.name)

            new_X = action(self._current_X, self._current_y)

            # Global pandera schema guard
            if self._schema is not None:
                try:
                    self._schema.validate(new_X, lazy=True)
                except pa.errors.SchemaErrors as exc:
                    raise DataValidationError(str(exc)) from exc

            self._current_X = new_X
            # Sync y when row-dropping actions (e.g. outlier removal) shrink X.
            # Requires actions to preserve the original index (no reset_index).
            if (
                self._current_y is not None
                and hasattr(self._current_y, "loc")
                and len(new_X) < len(self._current_y)
            ):
                self._current_y = self._current_y.loc[new_X.index]
            self._action_history.append(action_idx)
            self._step_count += 1

            reward = self._reward_fn(self._current_X, self._current_y)

            # Attach reward components if available
            if hasattr(self._reward_fn, "last_components") and self._reward_fn.last_components:
                info["components"] = self._reward_fn.last_components

        except DataValidationError as exc:
            logger.debug("Schema violation after %s: %s", action.name, exc)
            reward = self._penalty
            info["error"] = str(exc)
        except Exception as exc:
            logger.debug("Action %s raised %s: %s", action.name, type(exc).__name__, exc)
            reward = self._penalty
            info["error"] = str(exc)

        obs = self._get_obs()
        terminated = self._step_count >= self._max_steps
        truncated = False
        return obs, float(reward), terminated, truncated, info

    def render(self) -> Optional[str]:
        n_rows, n_cols = self._current_X.shape
        n_missing = int(self._current_X.isna().sum().sum())
        history_names = [self._actions[i].name for i in self._action_history]
        lines = [
            "─" * 60,
            f" Step {self._step_count}/{self._max_steps}",
            f" Shape  : {n_rows} × {n_cols}",
            f" Missing: {n_missing}",
            f" History: {history_names or '(none)'}",
        ]
        if hasattr(self._reward_fn, "last_components") and self._reward_fn.last_components:
            c = self._reward_fn.last_components
            lines.append(
                f" Reward : {c.total:.4f}  "
                f"(acc={c.accuracy:.3f} ret={c.retention:.3f} "
                f"qual={c.quality:.3f} drift_pen={c.drift_penalty:.3f})"
            )
        lines.append("─" * 60)
        output = "\n".join(lines)
        print(output)
        return output

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_X(self) -> Features:
        return self._current_X.copy()

    @property
    def current_y(self) -> OptionalTarget:
        return copy.deepcopy(self._current_y)

    @property
    def action_history(self) -> ActionHistory:
        return list(self._action_history)

    @property
    def action_names(self) -> List[str]:
        return [a.name for a in self._actions]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        return self._observer.observe(
            self._current_X,
            self._current_y,
            self._action_history,
            len(self._actions),
        )
