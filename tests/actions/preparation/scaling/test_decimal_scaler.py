import pandas as pd

from learn2clean.actions import DecimalScaler
from tests.utils.actions.assert_action_pipeline import assert_action_pipeline


def test_action_defaults():
    transform_data = {
        "a": [1, 10, 100],
        "b": [0.5, 5, 50],
        "c": [1000, 2000, 3000],
        "label": ["x", "y", "z"],
    }
    assert_action_pipeline(
        DecimalScaler(),
        transform_data,
        {
            "a": [0.001, 0.01, 0.1],
            "b": [0.005, 0.05, 0.5],
            "c": [0.1, 0.2, 0.3],
            "label": ["x", "y", "z"],
        },
        transform_data.copy(),
    )


def test_decimal_scaling_fit():
    sample_data = pd.DataFrame(
        {
            "Age": [80.0, 10.0, 45.5, 0.0],
            "Fare": [512.3292, 12.0, 0.5, 200.0],
            "Name": ["A", "B", "C", "D"],  # Non-numeric column
            "PowerOf10": [100.0, 10.0, 1.0, 0.0],
            "ZeroMax": [0.0, 0.0, 0.0, 0.0],
        }
    )
    scaler = DecimalScaler()
    scaler.fit(sample_data)

    # Max values: Age=80.0, Fare=512.3292, PowerOf10=100.0
    # Age: max=80.0 -> j=2 -> Divisor=100
    assert scaler._divisors.get("Age") == 100.0
    # Fare: max=512.3292 -> j=3 -> Divisor=1000
    assert scaler._divisors.get("Fare") == 1000.0
    # PowerOf10: max=100.0 -> log10=2.0 -> j=3 (correction appliquée) -> Divisor=1000
    assert scaler._divisors.get("PowerOf10") == 1000.0
    # Non-numeric column 'Name' should be ignored
    assert "Name" not in scaler._divisors
    # Column 'ZeroMax' should be ignored/skipped due to max|x| <= 0
    assert "ZeroMax" not in scaler._divisors

    assert_action_pipeline(
        scaler,
        sample_data,
        {
            "Age": [0.8, 0.1, 0.455, 0],
            "Fare": [0.5123292, 0.012, 0.0005, 0.2],
            "Name": ["A", "B", "C", "D"],
            "PowerOf10": [0.1, 0.01, 0.001, 0],
            "ZeroMax": [0.0, 0.0, 0.0, 0.0],
        },
    )


def test_action_params():
    assert_action_pipeline(
        DecimalScaler(columns=["a", "c"]),
        {
            "a": [1, 10, 100],
            "b": [0.5, 5, 50],
            "c": [1000, 2000, 3000],
        },
        {
            "a": [0.001, 0.01, 0.1],
            "b": [0.5, 5, 50],  # unchanged
            "c": [0.1, 0.2, 0.3],
        },
        {
            "a": [1, 10, 100],
            "b": [0.5, 5, 50],
            "c": [1000, 2000, 3000],
        },
    )


def test_decimal_scaling_leakage_safety():

    assert_action_pipeline(
        DecimalScaler(),
        {
            "value": [5000, 100, 1],
            "other": [50, 100, 20],
        },
        {
            "value": [0.5, 0.01, 0.0001],
            "other": [0.5, 1.0, 0.2],
        },
        {
            "value": [1, 10, 1000],
            "other": [5, 10, 8],
        },
        numeric_allclose=True,
    )


def test_action_with_exclude_param():
    assert_action_pipeline(
        DecimalScaler(exclude=["a", "c"]),
        {
            "a": [10, 100],
            "b": [5.0, 50.0],
            "c": [2000, 3000],
        },
        {
            "a": [10.0, 100.0],
            "b": [0.05, 0.5],
            "c": [2000.0, 3000.0],
        },
        {
            "a": [10, 100],
            "b": [5.0, 50.0],
            "c": [2000, 3000],
        },
    )
