from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


# ==========================================
# 1. BASE CONFIGURATION
# ==========================================


@dataclass
class DatasetConfig:
    _target_: str = MISSING
    target_col: str | None = None


@dataclass
class CSVDatasetConfig(DatasetConfig):
    """
    Configuration for loading local CSV files.
    """

    _target_: str = "learn2clean.loaders.CSVLoader"
    file_path: str | None = None
    dataset: str | None = None
    read_csv_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class HuggingFaceDatasetConfig(DatasetConfig):
    """
    Configuration for loading datasets from Hugging Face Hub.
    """

    _target_: str = "learn2clean.loaders.HuggingFaceLoader"
    path: str = MISSING
    name: str | None = None
    split: str = "train"
    revision: str | None = None
    load_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class KaggleDatasetConfig(DatasetConfig):
    """
    Configuration for loading datasets from Kaggle.
    """

    _target_: str = "learn2clean.loaders.KaggleLoader"
    dataset_id: str = MISSING  # Ex: 'zillow/zecon'
    filename: str = MISSING  # Ex: 'State_time_series.csv'
    download_dir: str = "${hydra:runtime.cwd}/data/raw/kaggle"
    read_csv_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenMLDatasetConfig(DatasetConfig):
    """
    Configuration for loading datasets from OpenML.
    """

    _target_: str = "learn2clean.loaders.OpenMLLoader"
    name: str | None = None
    version: str | int = "active"
    data_id: int | None = None
    parser: str = "auto"
    fetch_kwargs: dict[str, Any] = field(default_factory=dict)


def register_dataset_configs(cs: ConfigStore) -> None:
    """
    Registers the dataset configurations in the Hydra ConfigStore.
    This allows selecting them via CLI: python main.py dataset=kaggle
    """
    # 'dataset' group: user can choose dataset=csv, dataset=hf, etc.
    cs.store(group="dataset", name="csv", node=CSVDatasetConfig)
    cs.store(group="dataset", name="hf", node=HuggingFaceDatasetConfig)
    cs.store(group="dataset", name="kaggle", node=KaggleDatasetConfig)
    cs.store(group="dataset", name="openml", node=OpenMLDatasetConfig)
