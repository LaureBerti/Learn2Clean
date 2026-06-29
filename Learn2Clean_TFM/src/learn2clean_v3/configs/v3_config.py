"""
Learn2Clean V3 — Hydra configuration dataclasses.

All tunable parameters are exposed here.  Write defaults to conf/ with:
    python experiments/tutorials/01_v3_demo.py --write-config
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    _target_: str = MISSING
    target_col: str = "label"
    test_size: float = 0.25
    seed: int = 42


@dataclass
class CSVDatasetConfig(DatasetConfig):
    _target_: str = "learn2clean_v3.loaders.CSVLoader"
    path: str = MISSING


@dataclass
class OpenMLDatasetConfig(DatasetConfig):
    _target_: str = "learn2clean_v3.loaders.OpenMLLoader"
    task_id: int = MISSING


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------

@dataclass
class ObserverConfig:
    _target_: str = "learn2clean_v3.observers.DataQualityObserver"
    include_drift: bool = True
    include_skewness: bool = True
    include_kurtosis: bool = True
    include_missing_per_column: bool = True
    n_bins: int = 50
    epsilon: float = 1e-10


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    _target_: str = "learn2clean_v3.rewards.MultiObjectiveReward"
    # Weights for each objective (must sum to 1 or will be normalised)
    weight_accuracy: float = 0.5
    weight_retention: float = 0.3
    weight_quality: float = 0.2
    # Optional drift penalty coefficient (subtracted from total)
    drift_penalty_coeff: float = 0.1
    # Evaluation model used to compute accuracy inside reward
    eval_model: str = "random_forest"    # "random_forest" | "logistic" | "gradient_boosting"
    eval_metric: str = "accuracy"        # "accuracy" | "f1" | "roc_auc"
    eval_cv_folds: int = 3
    # ExplainableReward wrapper
    explainable: bool = True


@dataclass
class BaselineRewardConfig(RewardConfig):
    """V2-compatible single-objective reward (completeness × sqrt(retention))."""
    _target_: str = "learn2clean_v3.rewards.CompletenessRetentionReward"
    weight_accuracy: float = 0.0
    weight_retention: float = 1.0
    weight_quality: float = 0.0
    drift_penalty_coeff: float = 0.0


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@dataclass
class ActionConfig:
    """Single action entry in the action list."""
    _target_: str = MISSING
    # ParameterizedAction params (ignored for plain actions)
    strategy: Optional[str] = None
    n_neighbors: Optional[int] = None
    threshold: Optional[float] = None
    # Pandera validation after apply
    validate_schema: bool = True
    validation_strict: bool = False


@dataclass
class ActionsConfig:
    use_mean_imputer: bool = True
    use_median_imputer: bool = True
    use_knn_imputer: bool = True
    use_iqr_outlier: bool = True
    use_zscore_outlier: bool = True
    use_exact_dedup: bool = True
    use_minmax_scaler: bool = True
    use_zscore_scaler: bool = True


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    max_steps: int = 10
    invalid_action_penalty: float = -0.05
    # Offline RL pre-computation
    offline_mode: bool = False
    offline_max_sequences: int = 1000
    offline_max_length: int = 5


# ---------------------------------------------------------------------------
# Agent (SB3)
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    algorithm: str = "PPO"          # "PPO" | "DQN" | "A2C"
    total_timesteps: int = 50_000
    learning_rate: float = 3e-4
    n_steps: int = 2048             # PPO only
    batch_size: int = 64
    n_epochs: int = 10              # PPO only
    gamma: float = 0.99
    verbose: int = 1
    # Transfer learning
    pretrained_checkpoint: Optional[str] = None
    freeze_policy_layers: int = 0   # 0 = no freezing


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    enabled: bool = False
    n_episodes: int = 100
    n_eval_episodes: int = 20
    reward_functions: List[str] = field(default_factory=lambda: [
        "CompletenessRetentionReward",
        "AccuracyReward",
        "MultiObjectiveReward",
        "DriftPenaltyReward",
        "IncrementalGainReward",
    ])
    output_dir: str = "outputs/benchmark"
    plot: bool = True


# ---------------------------------------------------------------------------
# Experiment (top-level)
# ---------------------------------------------------------------------------

@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "learn2clean-v3"
    entity: Optional[str] = None    # set via WANDB_ENTITY env var


@dataclass
class V3Config:
    """Root Hydra config for Learn2Clean V3."""
    dataset: DatasetConfig = MISSING
    observer: ObserverConfig = field(default_factory=ObserverConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    actions: ActionsConfig = field(default_factory=ActionsConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    output_dir: str = "outputs"
    seed: int = 42


# ---------------------------------------------------------------------------
# Register with Hydra ConfigStore
# ---------------------------------------------------------------------------

def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="v3_config", node=V3Config)
    cs.store(group="dataset", name="csv", node=CSVDatasetConfig)
    cs.store(group="dataset", name="openml", node=OpenMLDatasetConfig)
    cs.store(group="reward", name="multi_objective", node=RewardConfig)
    cs.store(group="reward", name="baseline", node=BaselineRewardConfig)
