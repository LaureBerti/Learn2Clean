from pathlib import Path

import pandas as pd
import pytest

from learn2clean.loaders import DatasetLoader, CSVLoader

DUMMY_CSV_CONTENT = "col1,col2,col3\n1,a,True\n2,b,False\n3,c,True\n"
DUMMY_FILENAME = "test_titanic.csv"


@pytest.fixture
def temp_csv_file(tmp_path: Path) -> Path:
    """
    Pytest fixture that creates a temporary CSV file for testing.
    The file is automatically cleaned up after the test.
    """
    file_path = tmp_path / DUMMY_FILENAME

    file_path.write_text(DUMMY_CSV_CONTENT)

    return file_path


def test_csv_loader_inherits_from_dataset_loader():
    """Verify that CSVLoader respects the inheritance contract."""
    loader = CSVLoader("/dummy/path/file.csv", dataset="titanic.csv")
    assert isinstance(loader, DatasetLoader)


def test_csv_loader_initializes_path_correctly(temp_csv_file: Path):
    """Verify that the path is stored as a pathlib.Path object."""
    path_str = str(temp_csv_file)
    loader = CSVLoader(path_str, file_path=path_str)

    assert isinstance(loader.path, Path)
    assert loader.path == temp_csv_file


def test_csv_loader_initializes_correctly():
    loader = CSVLoader(dataset="titanic.csv")
    assert isinstance(loader.path, Path)
    assert loader.path.name == "titanic.csv"
    assert loader.path.parent.name == "data"
    assert loader.path.is_absolute()


def test_csv_loader_loads_data_into_dataframe(temp_csv_file: Path):
    """Verify that the load method correctly reads the CSV content."""
    loader = CSVLoader(file_path=str(temp_csv_file))
    df = loader.load()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert df.columns.tolist() == ["col1", "col2", "col3"]


def test_csv_loader_raises_file_not_found():
    """Verify that FileNotFoundError is raised when the file is missing."""
    missing_path = "/non/existent/path/missing_file.csv"
    loader = CSVLoader(file_path=missing_path)

    with pytest.raises(FileNotFoundError) as excinfo:
        loader.load()

    assert "CSV file missing at:" in str(excinfo.value)


def test_init_raises_if_both_provided():
    """Ensure we can't provide both arguments."""
    with pytest.raises(ValueError, match="provide exactly one"):
        CSVLoader(file_path="a.csv", dataset="b.csv")


def test_csv_loader_returns_correct_source_info(temp_csv_file: Path):
    """Verify that get_source_info returns accurate metadata."""
    path_str = str(temp_csv_file)
    loader = CSVLoader(file_path=path_str)

    info = loader.get_source_info()
    assert isinstance(info, dict)
    assert info["source_type"] == "csv_file"
    assert Path(info["path"]).name == "test_titanic.csv"
