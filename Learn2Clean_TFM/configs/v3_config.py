
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING



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



@dataclass
class ObserverConfig:
    _target_: str = "learn2clean_v3.observers.DataQualityObserver"
    include_drift: bool = True
    include_skewness: bool = True
    include_kurtosis: bool = True
    include_missing_per_column: bool = True
    n_bins: int = 50
    epsilon: float = 1e-10



@dataclass
class RewardConfig:
    _target_: str = "learn2clean_v3.rewards.MultiObjectiveReward"
    weight_accuracy: float = 0.5
    weight_retention: float = 0.3
    weight_quality: float = 0.2
    drift_penalty_coeff: float = 0.1
    eval_model: str = "random_forest"
    eval_metric: str = "accuracy"
    eval_cv_folds: int = 3
    explainable: bool = True


@dataclass
class BaselineRewardConfig(RewardConfig):
    _target_: str = "learn2clean_v3.rewards.CompletenessRetentionReward"
    weight_accuracy: float = 0.0
    weight_retention: float = 1.0
    weight_quality: float = 0.0
    drift_penalty_coeff: float = 0.0



@dataclass
class ActionConfig:
    _target_: str = MISSING
    strategy: Optional[str] = None
    n_neighbors: Optional[int] = None
    threshold: Optional[float] = None
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



@dataclass
class EnvConfig:
    max_steps: int = 10
    invalid_action_penalty: float = -0.05
    offline_mode: bool = False
    offline_max_sequences: int = 1000
    offline_max_length: int = 5



@dataclass
class AgentConfig:
    algorithm: str = "PPO"
    total_timesteps: int = 50_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    verbose: int = 1
    pretrained_checkpoint: Optional[str] = None
    freeze_policy_layers: int = 0



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



@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "learn2clean-v3"
    entity: Optional[str] = None


@dataclass
class V3Config:
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



def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="v3_config", node=V3Config)
    cs.store(group="dataset", name="csv", node=CSVDatasetConfig)
    cs.store(group="dataset", name="openml", node=OpenMLDatasetConfig)
    cs.store(group="reward", name="multi_objective", node=RewardConfig)
    cs.store(group="reward", name="baseline", node=BaselineRewardConfig)
