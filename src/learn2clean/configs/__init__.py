from hydra.core.config_store import ConfigStore

from .actions import register_action_configs
from .dataset import register_dataset_configs
from .distances import register_distance_configs
from .experiment import register_experiment_configs


def register_all_configs(cs=ConfigStore.instance()) -> None:
    if cs is None:
        cs = ConfigStore.instance()
    register_action_configs(cs)
    register_dataset_configs(cs)
    register_distance_configs(cs)
    register_experiment_configs(cs)
