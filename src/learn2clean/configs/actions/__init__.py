from hydra.core.config_store import ConfigStore

from .base import ActionListConfig
from .cleaning import register_cleaning_configs
from .dummy import register_dummy_configs
from .preparation import register_preparation_configs


def register_action_configs(cs: ConfigStore) -> None:
    register_dummy_configs(cs)
    register_cleaning_configs(cs)
    register_preparation_configs(cs)
    cs.store(name="actions", node=ActionListConfig)
