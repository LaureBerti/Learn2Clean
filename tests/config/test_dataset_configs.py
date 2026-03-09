from pathlib import Path
from typing import Any

import pytest
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf, MissingMandatoryValue

from learn2clean.configs.dataset import (
    CSVDatasetConfig,
    HuggingFaceDatasetConfig,
    KaggleDatasetConfig,
    OpenMLDatasetConfig,
    register_dataset_configs,
)


# --- 1. Dataclass Structure & Defaults Tests ---


def test_csv_loader_config_defaults() -> None:
    """
    Test CSVLoaderConfig defaults and mandatory fields.
    """
    # Initialize with mandatory field
    cfg = CSVDatasetConfig(file_path="data.csv")

    assert cfg._target_ == "learn2clean.loaders.CSVLoader"
    assert cfg.file_path == "data.csv"
    # Verify default mutable field factory
    assert cfg.read_csv_kwargs == {}


def test_hugging_face_loader_config_defaults() -> None:
    """
    Test HuggingFaceLoaderConfig defaults.
    """
    # Initialize with mandatory field
    cfg = HuggingFaceDatasetConfig(path="julien-c/titanic-survival")

    assert cfg._target_ == "learn2clean.loaders.HuggingFaceLoader"
    assert cfg.path == "julien-c/titanic-survival"

    # Verify defaults
    assert cfg.split == "train"
    assert cfg.name is None
    assert cfg.revision is None
    assert cfg.load_kwargs == {}


def test_kaggle_loader_config_defaults() -> None:
    """
    Test KaggleLoaderConfig defaults and interpolation strings.
    """
    # Initialize with mandatory fields
    cfg = KaggleDatasetConfig(dataset_id="user/dataset", filename="data.csv")

    assert cfg._target_ == "learn2clean.loaders.KaggleLoader"
    assert cfg.dataset_id == "user/dataset"
    assert cfg.filename == "data.csv"

    # Verify Hydra interpolation string matches exactly
    assert cfg.download_dir == "${hydra:runtime.cwd}/data/raw/kaggle"
    assert cfg.read_csv_kwargs == {}


def test_open_ml_loader_config_defaults() -> None:
    """
    Test OpenMLLoaderConfig defaults.
    """
    # OpenMLLoaderConfig has no strictly mandatory fields for instantiation
    # (defaults are None or specific values), so we can instantiate it empty.
    cfg = OpenMLDatasetConfig()

    assert cfg._target_ == "learn2clean.loaders.OpenMLLoader"

    # Verify defaults
    assert cfg.name is None
    assert cfg.data_id is None
    assert cfg.version == "active"
    assert cfg.parser == "auto"
    assert cfg.fetch_kwargs == {}


def test_mandatory_fields_are_missing() -> None:
    """
    Ensure that fields set to '???' behave as missing mandatory values
    when processed by OmegaConf.
    """
    # Create the raw dataclass
    raw_cfg = HuggingFaceDatasetConfig()

    # Convert to OmegaConf DictConfig to enable '???' behavior logic
    conf = OmegaConf.structured(raw_cfg)

    # Accessing a '???' field should raise an error
    with pytest.raises(MissingMandatoryValue):
        _ = conf.path


# --- 2. ConfigStore Registration Tests ---


def test_register_data_configs() -> None:
    """
    Verify that register_data_configs actually places the items
    into the Hydra ConfigStore.
    """
    # 1. Clear the store to ensure isolation (optional but safe)
    store = ConfigStore.instance()

    # 2. Call the registration function
    register_dataset_configs(store)

    # 3. Verify presence in the repo
    # Note: Accessing private attributes is necessary for unit testing the store content
    # strictly speaking, but logically we check if we can retrieve them.
    repo = store.repo
    assert "dataset" in repo


# --- 3. Instantiation Tests (Mocking) ---


def test_instantiate_csv_loader(mocker: Any) -> None:
    """
    Test that hydra.utils.instantiate correctly uses the config
    to call the target class.

    We mock the actual CSVLoader class to avoid filesystem operations.
    """
    # 1. Mock the class defined in _target_
    mock_cls = mocker.patch("learn2clean.loaders.CSVLoader")

    # 2. Create a valid config
    cfg = CSVDatasetConfig(
        file_path="dummy.csv", read_csv_kwargs={"sep": ";"}, target_col="Survived"
    )

    # 3. Instantiate via Hydra
    # We must convert to DictConfig or pass the object; Hydra handles both.
    loader = instantiate(cfg)

    # 4. Assertions
    # Did Hydra call our mock?
    mock_cls.assert_called_once_with(
        dataset=None,
        file_path="dummy.csv",
        read_csv_kwargs={"sep": ";"},
        target_col="Survived",
    )
    # Did it return the instance of our mock?
    assert loader == mock_cls.return_value


def test_instantiate_hugging_face_loader(mocker: Any) -> None:
    """
    Test instantiation for HuggingFaceDatasetConfig.
    """
    mock_cls = mocker.patch("learn2clean.loaders.HuggingFaceLoader")

    cfg = HuggingFaceDatasetConfig(
        path="my/dataset", split="test", load_kwargs={"trust_remote_code": True}
    )

    instantiate(cfg)

    mock_cls.assert_called_once_with(
        path="my/dataset",
        name=None,
        split="test",
        revision=None,
        load_kwargs={"trust_remote_code": True},
        target_col=None,
    )


def test_instantiate_kaggle_loader(
    mocker: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Test instantiation for KaggleDatasetConfig.
    """
    monkeypatch.setenv("KAGGLE_USERNAME", "dummy_user")
    monkeypatch.setenv("KAGGLE_KEY", "dummy_key")

    mock_cls = mocker.patch("learn2clean.loaders.KaggleLoader")

    cfg = KaggleDatasetConfig(
        dataset_id="sapal6/the-testcase-dataset",
        filename="Test_cases.csv",
        download_dir=str(tmp_path),
    )

    instantiate(cfg)

    mock_cls.assert_called_once_with(
        dataset_id="sapal6/the-testcase-dataset",
        filename="Test_cases.csv",
        download_dir=str(tmp_path),
        read_csv_kwargs={},
        target_col=None,
    )


def test_instantiate_open_ml_loader(mocker: Any) -> None:
    """
    Test instantiation for OpenMLLoaderConfig.
    Verifies that parameters from the config dataclass are correctly passed
    to the OpenMLLoader __init__ method.
    """
    # 1. Mock the class defined in _target_ to avoid real initialization/network
    mock_cls = mocker.patch("learn2clean.loaders.OpenMLLoader")

    # 2. Create a specific configuration
    cfg = OpenMLDatasetConfig(
        name="titanic",
        version=1,
        # data_id stays None by default
        parser="liac-arff",
        fetch_kwargs={"cache": True},
    )

    # 3. Instantiate via Hydra
    loader = instantiate(cfg)

    # 4. Assertions
    # Verify that the mocked class was called with the exact arguments from config
    mock_cls.assert_called_once_with(
        name="titanic",
        version=1,
        data_id=None,  # Default value in the Dataclass
        parser="liac-arff",
        fetch_kwargs={"cache": True},
        target_col=None,
    )

    # Verify the return value is indeed our mock instance
    assert loader == mock_cls.return_value
