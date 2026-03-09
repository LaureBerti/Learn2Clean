from dataclasses import dataclass, field
from typing import Any, Dict

from hydra.core.config_store import ConfigStore

from ..base import ActionConfig


@dataclass
class PanderaSchemaValidatorConfig(ActionConfig):
    name: str = "PanderaSchemaValidator"
    _target_: str = "learn2clean.actions.PanderaSchemaValidator"
    remove_inconsistent_rows: bool = True
    schema_config: Dict[str, Any] = field(default_factory=lambda: {"columns": {}})


def register_inconsistency_detection_configs(cs: ConfigStore) -> None:
    group = "action/inconsistency_detection"
    cs.store(group=group, name="pandera", node=PanderaSchemaValidatorConfig)
