from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore

from ..base import ActionConfig


@dataclass
class IQROutlierCleanerConfig(ActionConfig):
    name: str = "IQROutlierCleaner"
    _target_: str = "learn2clean.actions.IQROutlierCleaner"
    factor: float = 1.5
    method: str = "mask"  # 'drop', 'mask', 'clip'
    numeric_only: bool = True


@dataclass
class LocalOutlierFactorCleanerConfig(ActionConfig):
    name: str = "LocalOutlierFactorCleaner"
    _target_: str = "learn2clean.actions.LocalOutlierFactorCleaner"
    n_neighbors: int = 20
    contamination: Any = "auto"
    method: str = "mask"  # 'drop', 'mask'


@dataclass
class ZScoreOutlierCleanerConfig(ActionConfig):
    name: str = "ZScoreOutlierCleaner"
    _target_: str = "learn2clean.actions.ZScoreOutlierCleaner"
    threshold: float = 3.0
    method: str = "mask"  # 'drop', 'mask', 'clip'
    numeric_only: bool = True


def register_outlier_configs(cs: ConfigStore) -> None:
    group = "action/outlier_detection"
    cs.store(group=group, name="iqr", node=IQROutlierCleanerConfig)
    cs.store(group=group, name="lof", node=LocalOutlierFactorCleanerConfig)
    cs.store(group=group, name="z_score", node=ZScoreOutlierCleanerConfig)
