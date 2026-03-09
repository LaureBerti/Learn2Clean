from typing import cast

import pandas as pd
import pytest

from learn2clean.loaders import DatasetLoader, HuggingFaceLoader


@pytest.fixture
def hf_loader() -> HuggingFaceLoader:
    """
    Provides a configured HuggingFaceLoader instance for the Titanic dataset.

    Returns:
        HuggingFaceLoader: An initialized loader instance.
    """
    return HuggingFaceLoader(
        path="julien-c/titanic-survival",
        split="train",
        revision="main",
        load_kwargs={},
    )


def test_huggingface_loader_implements_interface(
    hf_loader: HuggingFaceLoader,
) -> None:
    """
    Ensures the concrete loader correctly implements the abstract DatasetLoader interface.
    """
    # Use 'cast' to assert the type conformance for mypy compliance
    loader_interface: DatasetLoader = cast(DatasetLoader, hf_loader)
    assert isinstance(loader_interface, DatasetLoader)


def test_huggingface_loader_loads_data_correctly(
    hf_loader: HuggingFaceLoader,
) -> None:
    """
    Tests if the loader loads the correct dataset split into a non-empty pandas DataFrame.
    """

    df = hf_loader.load()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape[0] > 0
    assert "Survived" in df.columns


def test_huggingface_loader_source_info(hf_loader: HuggingFaceLoader) -> None:
    """
    Tests get_source_info returns the correct metadata matching the configuration.
    """
    info = hf_loader.get_source_info()

    # Assert that the essential configuration parameters are correctly present in the metadata
    assert info["source_type"] == "hugging_face"
    assert info["path"] == "julien-c/titanic-survival"
    assert info["config_name"] is None
    assert info["split"] == "train"
    assert info["revision"] == "main"
