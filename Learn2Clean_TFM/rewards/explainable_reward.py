
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from learn2clean_v3.rewards.base_reward import BaseReward
from learn2clean_v3.types import Features, OptionalTarget, RewardDelta

logger = logging.getLogger(__name__)


class ExplainableReward(BaseReward):

    def __init__(self, base_reward: BaseReward) -> None:
        self._base = base_reward
        self._history: List[RewardDelta] = []
        self._episode_history: List[RewardDelta] = []
        self._prev_reward: Optional[float] = None
        self._step: int = 0
        self._pending_action_name: str = "unknown"


    def reset(self, X_initial: Features, y_initial: OptionalTarget) -> None:
        self._base.reset(X_initial, y_initial)
        self._prev_reward = self._base(X_initial, y_initial)
        self._step = 0
        self._episode_history = []
        self._pending_action_name = "unknown"

    def __call__(self, X: Features, y: OptionalTarget) -> float:
        reward = self._base(X, y)

        if self._prev_reward is not None:
            delta = reward - self._prev_reward
            cumulative = sum(e.delta for e in self._episode_history) + delta
            entry = RewardDelta(
                step=self._step,
                action_name=self._pending_action_name,
                reward_before=self._prev_reward,
                reward_after=reward,
                delta=delta,
                cumulative_reward=cumulative,
            )
            self._episode_history.append(entry)
            self._history.append(entry)

        self._prev_reward = reward
        self._step += 1
        return reward

    def set_action_name(self, name: str) -> None:
        self._pending_action_name = name


    @property
    def name(self) -> str:
        return f"Explainable({self._base.name})"

    def episode_history(self) -> List[RewardDelta]:
        return list(self._episode_history)

    def full_history(self) -> List[RewardDelta]:
        return list(self._history)

    def history_as_dataframe(self, episode_only: bool = False) -> pd.DataFrame:
        data = self._episode_history if episode_only else self._history
        if not data:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "step": e.step,
                "action": e.action_name,
                "reward_before": round(e.reward_before, 5),
                "reward_after": round(e.reward_after, 5),
                "delta": round(e.delta, 5),
                "cumulative": round(e.cumulative_reward, 5),
            }
            for e in data
        ])

    def top_actions(self, n: int = 5) -> pd.DataFrame:
        df = self.history_as_dataframe()
        if df.empty:
            return df
        return (
            df.groupby("action")["delta"]
            .agg(["mean", "sum", "count"])
            .rename(columns={"mean": "mean_delta", "sum": "total_gain", "count": "times_used"})
            .sort_values("mean_delta", ascending=False)
            .head(n)
        )

    def worst_actions(self, n: int = 5) -> pd.DataFrame:
        df = self.history_as_dataframe()
        if df.empty:
            return df
        return (
            df.groupby("action")["delta"]
            .agg(["mean", "sum", "count"])
            .rename(columns={"mean": "mean_delta", "sum": "total_gain", "count": "times_used"})
            .sort_values("mean_delta", ascending=True)
            .head(n)
        )

    def plot(self, episode_only: bool = True, title: str = "") -> None:
        try:
            import plotly.express as px
        except ImportError:
            logger.warning("plotly not installed — cannot plot ExplainableReward history.")
            return

        df = self.history_as_dataframe(episode_only=episode_only)
        if df.empty:
            logger.warning("No reward history to plot.")
            return
        fig = px.bar(
            df,
            x="step",
            y="delta",
            color="action",
            title=title or f"Per-Step Reward Deltas ({self.name})",
            labels={"delta": "Reward Δ", "step": "Step"},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.show()
