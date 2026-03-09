from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class ExperimentConfig:
    name: str = MISSING
    seed: int = 42
    test_size: float = 0.25


def register_experiment_configs(cs: ConfigStore) -> None:
    cs.store(name="experiment", node=ExperimentConfig)
