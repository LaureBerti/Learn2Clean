from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class ActionConfig:
    _target_: str = MISSING
    name: str | None = None
    columns: list[str] | None = None
    exclude: list[str] | None = None


@dataclass
class ActionListConfig:
    actions: list[ActionConfig] = field(default_factory=list)
