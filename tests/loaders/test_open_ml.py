from pathlib import Path

from sklearn.datasets import fetch_openml

from learn2clean.loaders import OpenMLLoader


def describe_dataframe(df):
    """Utility function to print dataset info (numeric vs categorical features)."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    print(f"Shape: {df.shape}")
    print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols[:5]}...")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols[:5]}...")
    return numeric_cols, categorical_cols


def test_titanic_dataset(tmp_path: Path) -> None:
    """Titanic dataset: contains missing values and categorical features, great for testing imputation and feature encoding."""
    loader = OpenMLLoader(name="titanic", version=1)
    df = loader.load()
    print("\n=== Titanic Dataset ===")
    describe_dataframe(df)
    print(df.head())
    assert not df.empty


def test_adult_dataset(tmp_path: Path) -> None:
    """Adult dataset: census data with categorical and numerical features, contains missing values, good for preprocessing experiments."""
    loader = OpenMLLoader(name="adult", version=2)
    df = loader.load()
    print("\n=== Adult Dataset ===")
    describe_dataframe(df)
    print(df.head())
    assert not df.empty


def test_german_credit_dataset(tmp_path: Path) -> None:
    """German Credit dataset: mixed numerical and categorical features, useful for testing normalization and encoding."""
    loader = OpenMLLoader(
        data_id=31
    )  # Name is ignored if data_id is passed to fetch_openml
    df = loader.load()
    print("\n=== German Credit Dataset ===")
    describe_dataframe(df)
    print(df.head())
    assert not df.empty


def test_blood_transfusion_dataset(tmp_path: Path) -> None:
    """Blood Transfusion dataset: small numeric dataset, can test outlier handling and scaling."""
    loader = OpenMLLoader(data_id=1464)
    df = loader.load()
    print("\n=== Blood Transfusion Dataset ===")
    describe_dataframe(df)
    print(df.head())
    assert not df.empty


def test_iris_dataset(tmp_path: Path) -> None:
    """Iris dataset: classic small dataset, useful for basic sanity checks."""
    loader = OpenMLLoader(name="iris", version=1)
    df = loader.load()
    print("\n=== Iris Dataset ===")
    describe_dataframe(df)
    print(df.head())
    assert not df.empty


def test_wine_dataset(tmp_path: Path) -> None:
    """Wine dataset: small dataset with some categorical columns, easy for quick tests."""
    loader = OpenMLLoader(name="wine", version=1)
    df = loader.load()
    print("\n=== Wine Dataset ===")
    describe_dataframe(df)
    print(df.head())
    assert not df.empty


def test_boston_housing_dataset(tmp_path: Path) -> None:
    """Boston Housing dataset: numeric features, good for testing normalization and outlier transformations."""
    df = fetch_openml(data_id=531).frame
    print("\n=== Boston Housing Dataset ===")
    describe_dataframe(df)
    print(df.head())
    assert not df.empty


def test_diabetes_dataset(tmp_path: Path) -> None:
    """Diabetes dataset: numeric medical data with missing values, suitable for imputation and regression testing."""
    loader = OpenMLLoader(data_id=37)
    df = loader.load()
    print("\n=== Diabetes Dataset ===")
    describe_dataframe(df)
    print(df.head())
    assert not df.empty


def test_breast_cancer_dataset(tmp_path: Path) -> None:
    """Breast Cancer Wisconsin dataset: numeric and categorical features, commonly used for testing pipelines."""
    loader = OpenMLLoader(name="breast-w", version=1)
    df = loader.load()
    print("\n=== Breast Cancer Dataset ===")
    describe_dataframe(df)
    print(df.head())
    assert not df.empty


def test_balance_scale_dataset(tmp_path: Path) -> None:
    """Balance Scale dataset: small dataset, useful for testing feature transformations and model training."""
    loader = OpenMLLoader(name="balance-scale", version=1)
    df = loader.load()
    print("\n=== Balance Scale Dataset ===")
    describe_dataframe(df)
    print(df.head())
    assert not df.empty
