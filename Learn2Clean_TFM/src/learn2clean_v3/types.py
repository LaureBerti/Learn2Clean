"""Core type aliases for Learn2Clean V3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Base data types (unchanged from V2)
# ---------------------------------------------------------------------------

Features = pd.DataFrame
OptionalFeatures = Optional[Features]

Target = Union[pd.Series, np.ndarray]
OptionalTarget = Optional[Target]

RewardFunction = Callable[[Features, OptionalTarget], float]
MetricFunction = Callable[[Target, Target], float]
MetricType = Union[str, MetricFunction]

# ---------------------------------------------------------------------------
# V3 additions
# ---------------------------------------------------------------------------

ActionIndex = int
ActionHistory = List[ActionIndex]
ObservationVector = np.ndarray

# Reward delta produced by ExplainableReward
@dataclass
class RewardDelta:
    step: int
    action_name: str
    reward_before: float
    reward_after: float
    delta: float
    cumulative_reward: float
    meta: Dict[str, Any] = field(default_factory=dict)


# Multi-objective reward components
@dataclass
class RewardComponents:
    accuracy: float = 0.0
    retention: float = 0.0
    quality: float = 0.0
    drift_penalty: float = 0.0
    total: float = 0.0


# Parameterized action spec
@dataclass
class ParamSpec:
    """Specification for a single hyperparameter of a ParameterizedAction."""
    name: str
    dtype: str            # "float", "int", or "categorical"
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    default: Any = None


# Offline RL transition
@dataclass
class Transition:
    obs: ObservationVector
    action: ActionIndex
    reward: float
    next_obs: ObservationVector
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


# Benchmark results
@dataclass
class BenchmarkEntry:
    reward_fn_name: str
    episode_rewards: List[float]
    final_reward: float
    convergence_step: Optional[int]
    action_sequence: ActionHistory
    reward_components: Optional[RewardComponents]


@dataclass
class BenchmarkResults:
    entries: List[BenchmarkEntry]
    dataset_name: str
    n_actions: int
    n_episodes: int

    def best(self) -> BenchmarkEntry:
        return max(self.entries, key=lambda e: e.final_reward)

    def as_dataframe(self) -> pd.DataFrame:
        rows = []
        for e in self.entries:
            rows.append({
                "reward_fn": e.reward_fn_name,
                "final_reward": e.final_reward,
                "convergence_step": e.convergence_step,
                "n_episodes": len(e.episode_rewards),
            })
        return pd.DataFrame(rows)
