from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, fetch_openml
from ydata_profiling import ProfileReport


def test_profiling(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "age": [25, 30, np.nan, 45],
            "name": ["Alice", "Bob", "Charlie", "David"],
            "score": [88.5, 92.3, 85.0, None],
            "is_active": [True, False, True, True],
        }
    )

    report = ProfileReport(df, title="Test Profiling", minimal=True)

    output_file = tmp_path / "profile_test.html"
    report.to_file(output_file)
    print(output_file)
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_profile_titanic_dataset(tmp_path: Path) -> None:
    df = fetch_openml(name="titanic", version=1, as_frame=True).frame
    report = ProfileReport(df, title="Test Profiling", minimal=True)
    output_file = tmp_path / "titanic_profile_test.html"
    report.to_file(output_file)
    print(output_file)
    assert output_file.exists()


def test_profile_load_iris(tmp_path: Path):
    iris = load_iris(as_frame=True)
    df = iris.frame
    report = ProfileReport(df, title="Iris Profiling", minimal=True)
    output_file = tmp_path / "iris_profile.html"
    report.to_file(output_file)
    print(output_file)
    assert output_file.exists()


def test_compare(tmp_path: Path) -> None:
    df1 = pd.DataFrame(
        {
            "age": [25, None, 25, 30, None],
            "gender": ["M", "F", None, "F", "F"],
            "income": [50000, 50000, None, 50000, None],
            "city": ["Paris", "Paris", "Lyon", None, "Paris"],
        }
    )
    pr1 = ProfileReport(df1, title="Before")
    print(pr1.to_json())
    df2 = pd.DataFrame(
        {
            "age": [25, 25, 25, 30, 25],
            "gender": ["M", "F", "F", "F", "F"],
            "income": [50000, 50000, 50000, 50000, 50000],
            "city": ["Paris", "Paris", "Lyon", "Paris", "Paris"],
        }
    )
    pr2 = ProfileReport(df2, title="After")
    comparison_report = pr1.compare(pr2)
    report = comparison_report.to_json()
    print(report)
