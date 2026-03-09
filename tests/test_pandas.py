import pandas as pd


def test_create_dataframe() -> None:
    """Check that a DataFrame can be created and has correct shape and columns."""
    # Create a simple DataFrame
    df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]})

    # Check the shape
    assert df.shape == (3, 2)

    # Check the column names
    assert list(df.columns) == ["name", "age"]

    # Check a value
    assert df.loc[0, "name"] == "Alice"


def test_add_column() -> None:
    """Check that a new column can be added correctly."""
    df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})

    # Add a new column
    df["sum"] = df["x"] + df["y"]

    # Check the new column
    assert "sum" in df.columns
    assert df["sum"].tolist() == [11, 22, 33]
