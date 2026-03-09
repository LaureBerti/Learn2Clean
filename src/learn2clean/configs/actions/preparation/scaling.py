from dataclasses import dataclass
from typing import Any, Optional

from hydra.core.config_store import ConfigStore

from ..base import ActionConfig


@dataclass
class DecimalScalerConfig(ActionConfig):
    name: str = "DecimalScaler"
    _target_: str = "learn2clean.actions.DecimalScaler"


@dataclass
class Log10ScalerConfig(ActionConfig):
    name: str = "Log10Scaler"
    _target_: str = "learn2clean.actions.Log10Scaler"
    shift_epsilon: float = 1e-6


@dataclass
class MinMaxScalerConfig(ActionConfig):
    name: str = "MinMaxScaler"
    _target_: str = "learn2clean.actions.MinMaxScaler"


@dataclass
class QuantileScalerConfig(ActionConfig):
    name: str = "QuantileScaler"
    _target_: str = "learn2clean.actions.QuantileScaler"
    n_quantiles: int = 1000
    output_distribution: str = "uniform"
    ignore_implicit_zeros: bool = False
    subsample: int = 100000
    random_state: Optional[int] = None


@dataclass
class ZScoreScalerConfig(ActionConfig):
    name: str = "ZScoreScaler"
    _target_: str = "learn2clean.actions.ZScoreScaler"
    with_mean: bool = True
    with_std: bool = True


def register_scaling_configs(cs: ConfigStore) -> None:
    group = "action/scaling"
    cs.store(group=group, name="decimal", node=DecimalScalerConfig)
    cs.store(group=group, name="log10", node=Log10ScalerConfig)
    cs.store(group=group, name="min_max", node=MinMaxScalerConfig)
    cs.store(group=group, name="quantile", node=QuantileScalerConfig)
    cs.store(group=group, name="z_score", node=ZScoreScalerConfig)
