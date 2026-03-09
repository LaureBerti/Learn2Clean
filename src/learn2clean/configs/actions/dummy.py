from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from .base import ActionConfig


@dataclass
class DummyAddConfig(ActionConfig):
    _target_: str = "learn2clean.actions.DummyAdd"
    increment: int = 1
    name: str = "DummyAdd"


def register_dummy_configs(cs: ConfigStore) -> None:
    cs.store(group="action", name="dummy", node=DummyAddConfig)
