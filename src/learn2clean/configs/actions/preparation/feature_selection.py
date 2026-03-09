from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from ..base import ActionConfig


@dataclass
class ChiSquareSelectorConfig(ActionConfig):
    name: str = "ChiSquareSelector"
    _target_: str = "learn2clean.actions.ChiSquareSelector"
    k: int = 10


@dataclass
class LinearCorrelationSelectorConfig(ActionConfig):
    name: str = "LinearCorrelationSelector"
    _target_: str = "learn2clean.actions.LinearCorrelationSelector"
    k: int = 10


@dataclass
class MutualInformationSelectorConfig(ActionConfig):
    name: str = "MutualInformationSelector"
    _target_: str = "learn2clean.actions.MutualInformationSelector"
    k: int = 10
    random_state: int | None = 42
    n_neighbors: int = 3


@dataclass
class RandomForestSelectorConfig(ActionConfig):
    _target_: str = "learn2clean.actions.RandomForestSelector"
    name: str = "RandomForestSelector"
    n_estimators: int = 100
    threshold: str | float = "median"
    max_features: int | None = None
    random_state: int | None = 42


@dataclass
class VarianceThresholdSelectorConfig(ActionConfig):
    _target_: str = "learn2clean.actions.VarianceThresholdSelector"
    name: str = "VarianceThresholdSelector"
    threshold: float = 0.0


def register_feature_selection_configs(cs: ConfigStore) -> None:
    grp = "action/feature_selection"
    cs.store(group=grp, name="chi_square", node=ChiSquareSelectorConfig)
    cs.store(group=grp, name="linear_correlation", node=LinearCorrelationSelectorConfig)
    cs.store(group=grp, name="mutual_info", node=MutualInformationSelectorConfig)
    cs.store(group=grp, name="random_forest", node=RandomForestSelectorConfig)
    cs.store(group=grp, name="variance_threshold", node=VarianceThresholdSelectorConfig)
