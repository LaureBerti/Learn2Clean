"""
RewardBenchmark — V3 improvement #9 (new module).

Systematically compares multiple reward functions on the same dataset and
action space.  For each reward function it:

  1. Creates a fresh SequentialCleaningEnvV3 instance.
  2. Wraps with an ExplainableReward so per-action deltas are captured.
  3. Trains a PPO agent for ``n_episodes`` * ``max_steps`` total steps.
  4. Evaluates the learned policy for ``n_eval_episodes``.
  5. Records learning curve, final reward, convergence step, and
     the discovered action sequence.

Results are returned as a BenchmarkResults object and can be plotted.

Reward functions included by default
-------------------------------------
  CompletenessRetentionReward  — V2 baseline
  AccuracyReward               — pure ML performance
  MultiObjectiveReward         — weighted accuracy + retention + quality
  DriftPenaltyReward           — accuracy penalised by distribution drift
  IncrementalGainReward        — reward step-deltas, not absolute score

Usage
-----
    from learn2clean_v3.benchmark.reward_benchmark import RewardBenchmark, default_reward_functions

    bench = RewardBenchmark(
        X=df,
        y=labels,
        actions=my_actions,
        reward_functions=default_reward_functions(),
        n_episodes=200,
        n_eval_episodes=20,
    )
    results = bench.run()
    print(results.as_dataframe())
    bench.plot(results)
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from learn2clean_v3.actions.data_frame_action import DataFrameAction
from learn2clean_v3.envs.sequential_cleaning_env_v3 import SequentialCleaningEnvV3
from learn2clean_v3.observers.data_quality_observer import DataQualityObserver
from learn2clean_v3.rewards.base_reward import BaseReward
from learn2clean_v3.rewards.completeness_retention_reward import CompletenessRetentionReward
from learn2clean_v3.rewards.data_distortion_reward import DataDistortionPenaltyReward
from learn2clean_v3.rewards.explainable_reward import ExplainableReward
from learn2clean_v3.rewards.multi_objective_reward import (
    AccuracyReward,
    DriftPenaltyReward,
    IncrementalGainReward,
    MultiObjectiveReward,
)
from learn2clean_v3.types import (
    BenchmarkEntry,
    BenchmarkResults,
    Features,
    OptionalTarget,
    RewardComponents,
)

logger = logging.getLogger(__name__)


def default_reward_functions() -> List[BaseReward]:
    """Return the standard suite of reward functions for benchmarking."""
    return [
        CompletenessRetentionReward(),
        AccuracyReward(eval_model="random_forest"),
        MultiObjectiveReward(
            weight_accuracy=0.5,
            weight_retention=0.3,
            weight_quality=0.2,
            drift_penalty_coeff=0.1,
        ),
        DriftPenaltyReward(drift_coeff=0.4),
        IncrementalGainReward(base_reward=MultiObjectiveReward()),
        DataDistortionPenaltyReward(),
        DataDistortionPenaltyReward(
            weight_accuracy=0.3,
            eval_cv_folds=1,
        ),
    ]


class RewardBenchmark:
    """
    Parameters
    ----------
    X : Features
    y : OptionalTarget
    actions : list[DataFrameAction]
    reward_functions : list[BaseReward]
    n_episodes : int
        Training budget per reward function (in full episodes).
    n_eval_episodes : int
        Deterministic evaluation episodes after training.
    max_steps : int
        Steps per episode.
    seed : int
    output_dir : str | None
        If set, saves per-reward learning curves as CSV.
    verbose : int
        SB3 verbosity level.
    """

    def __init__(
        self,
        X: Features,
        y: OptionalTarget,
        actions: List[DataFrameAction],
        reward_functions: Optional[List[BaseReward]] = None,
        n_episodes: int = 100,
        n_eval_episodes: int = 20,
        max_steps: int = 10,
        seed: int = 42,
        output_dir: Optional[str] = None,
        verbose: int = 0,
    ) -> None:
        self._X = X
        self._y = y
        self._actions = actions
        self._reward_fns = reward_functions or default_reward_functions()
        self._n_episodes = n_episodes
        self._n_eval = n_eval_episodes
        self._max_steps = max_steps
        self._seed = seed
        self._output_dir = Path(output_dir) if output_dir else None
        self._verbose = verbose

    # ------------------------------------------------------------------

    def run(self) -> BenchmarkResults:
        """Run the full benchmark; returns BenchmarkResults."""
        entries: List[BenchmarkEntry] = []

        for reward_fn in self._reward_fns:
            logger.info("Benchmarking: %s", reward_fn.name)
            entry = self._run_one(reward_fn)
            entries.append(entry)

            if self._output_dir:
                self._output_dir.mkdir(parents=True, exist_ok=True)
                df = pd.DataFrame({"episode_reward": entry.episode_rewards})
                df.to_csv(self._output_dir / f"{entry.reward_fn_name}.csv", index=False)

        results = BenchmarkResults(
            entries=entries,
            dataset_name=f"DataFrame({len(self._X)}×{len(self._X.columns)})",
            n_actions=len(self._actions),
            n_episodes=self._n_episodes,
        )
        self._print_summary(results)
        return results

    # ------------------------------------------------------------------

    def _run_one(self, reward_fn: BaseReward) -> BenchmarkEntry:
        """Train and evaluate one reward function."""
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import BaseCallback
            from stable_baselines3.common.monitor import Monitor
            from stable_baselines3.common.vec_env import DummyVecEnv
        except ImportError as exc:
            raise ImportError("stable-baselines3 required for benchmarking.") from exc

        # Wrap in ExplainableReward so we capture per-step deltas
        explainable = ExplainableReward(reward_fn)

        def make_env():
            env = SequentialCleaningEnvV3(
                X=self._X,
                y=self._y,
                actions=self._actions,
                reward_fn=explainable,
                observer=DataQualityObserver(),
                max_steps=self._max_steps,
            )
            return Monitor(env)

        vec_env = DummyVecEnv([make_env])

        # Training callback to record episode rewards
        episode_rewards: List[float] = []

        class _RecordCallback(BaseCallback):
            def __init__(self) -> None:
                super().__init__(verbose=0)

            def _on_step(self) -> bool:
                infos = self.locals.get("infos", [])
                for info in infos:
                    if "episode" in info:
                        episode_rewards.append(info["episode"]["r"])
                return True

        total_timesteps = self._n_episodes * self._max_steps

        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=self._verbose,
            seed=self._seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.learn(total_timesteps=total_timesteps, callback=_RecordCallback())

        # Evaluate deterministically
        final_rewards: List[float] = []
        final_sequences: List[List[int]] = []

        eval_env = SequentialCleaningEnvV3(
            X=self._X,
            y=self._y,
            actions=self._actions,
            reward_fn=reward_fn,
            observer=DataQualityObserver(),
            max_steps=self._max_steps,
        )
        for _ in range(self._n_eval):
            obs, _ = eval_env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(int(action))
                ep_reward += reward
                done = terminated or truncated
            final_rewards.append(ep_reward)
            final_sequences.append(list(eval_env.action_history))

        final_reward = float(np.mean(final_rewards))
        convergence_step = self._find_convergence(episode_rewards)

        # Best action sequence (from highest-reward eval episode)
        best_idx = int(np.argmax(final_rewards))
        best_seq = final_sequences[best_idx]

        # Last reward components if available
        last_components: Optional[RewardComponents] = None
        if isinstance(reward_fn, MultiObjectiveReward):
            last_components = reward_fn.last_components

        vec_env.close()

        return BenchmarkEntry(
            reward_fn_name=reward_fn.name,
            episode_rewards=episode_rewards,
            final_reward=final_reward,
            convergence_step=convergence_step,
            action_sequence=best_seq,
            reward_components=last_components,
        )

    # ------------------------------------------------------------------

    def plot(self, results: BenchmarkResults, show: bool = True) -> None:
        """Plot learning curves for all reward functions."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("plotly not installed — cannot plot benchmark results.")
            return

        fig = go.Figure()
        for entry in results.entries:
            smoothed = self._smooth(entry.episode_rewards)
            fig.add_trace(go.Scatter(
                y=smoothed,
                mode="lines",
                name=entry.reward_fn_name,
            ))
        fig.update_layout(
            title=f"Reward Benchmark — {results.dataset_name}",
            xaxis_title="Episode",
            yaxis_title="Episode Reward (smoothed)",
            legend_title="Reward Function",
        )
        if show:
            fig.show()

    def plot_bar(self, results: BenchmarkResults, show: bool = True) -> None:
        """Bar chart of final performance per reward function."""
        try:
            import plotly.express as px
        except ImportError:
            return
        df = results.as_dataframe().sort_values("final_reward", ascending=True)
        fig = px.bar(
            df,
            x="final_reward",
            y="reward_fn",
            orientation="h",
            title="Final Reward by Reward Function",
            labels={"final_reward": "Mean Final Reward", "reward_fn": ""},
            color="final_reward",
            color_continuous_scale="Blues",
        )
        if show:
            fig.show()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _smooth(values: List[float], window: int = 10) -> List[float]:
        if len(values) < window:
            return values
        kernel = np.ones(window) / window
        return list(np.convolve(values, kernel, mode="valid"))

    @staticmethod
    def _find_convergence(rewards: List[float], patience: int = 20, tol: float = 0.01) -> Optional[int]:
        """Return the episode at which the smoothed reward stops improving."""
        if len(rewards) < patience * 2:
            return None
        smoothed = RewardBenchmark._smooth(rewards, window=patience)
        for i in range(len(smoothed) - patience):
            window = smoothed[i: i + patience]
            if max(window) - min(window) < tol:
                return i
        return None

    @staticmethod
    def _print_summary(results: BenchmarkResults) -> None:
        df = results.as_dataframe().sort_values("final_reward", ascending=False)
        try:
            from rich.console import Console
            from rich.table import Table
            console = Console()
            table = Table(title=f"Benchmark Results — {results.dataset_name}")
            for col in df.columns:
                table.add_column(col, style="cyan" if col == "reward_fn" else "white")
            for _, row in df.iterrows():
                table.add_row(*[str(round(v, 4)) if isinstance(v, float) else str(v) for v in row])
            console.print(table)
        except ImportError:
            print(df.to_string(index=False))
