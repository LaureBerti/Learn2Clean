from pathlib import Path

from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml


def test_titanic_dataset(tmp_path: Path) -> None:
    """Titanic dataset: contains missing values and categorical features, great for testing imputation and feature encoding."""
    df = fetch_openml(name="titanic", version=1, as_frame=True).frame
    plt.figure(figsize=(8, 5))
    plt.hist(df["age"].dropna(), bins=30, color="skyblue", edgecolor="black")
    plt.title("Passengers age distribution")
    plt.xlabel("Age")
    plt.ylabel("Passengers number")
    plt.show()
