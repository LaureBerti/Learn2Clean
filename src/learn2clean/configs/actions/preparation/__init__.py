from hydra.core.config_store import ConfigStore

from .feature_selection import register_feature_selection_configs
from .scaling import register_scaling_configs


def register_preparation_configs(cs: ConfigStore) -> None:
    register_feature_selection_configs(cs)
    register_scaling_configs(cs)
