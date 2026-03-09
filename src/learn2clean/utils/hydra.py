from typing import List

from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

from learn2clean.actions.data_frame_action import DataFrameAction


def extract_and_instantiate_actions(actions_cfg: DictConfig) -> List[DataFrameAction]:
    """
    Recursively traverse a Hydra DictConfig and instantiate all actions
    (i.e., sub-dictionaries containing a `_target_` key).

    Returns a list of instantiated DataFrameAction objects.
    """
    instances: List[DataFrameAction] = []

    def _recurse_actions(node: DictConfig):
        for key, value in node.items():
            if OmegaConf.is_dict(value) and "_target_" in value:
                instance = instantiate(value)
                instances.append(instance)
            elif OmegaConf.is_dict(value):
                _recurse_actions(value)

    _recurse_actions(actions_cfg)
    return instances
