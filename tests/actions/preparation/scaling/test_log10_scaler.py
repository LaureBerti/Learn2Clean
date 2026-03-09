from hydra import compose, initialize
from hydra.utils import instantiate

from learn2clean.actions import Log10Scaler
from tests.utils.actions.assert_action_pipeline import assert_action_pipeline


def test_log10_scaler_positive_data():
    transform_data = {
        "x1": [1.0, 10.0, 100.0, 1000.0],
        "x2": [5.0, 50.0, 500.0, 5000.0],
    }

    assert_action_pipeline(
        action=Log10Scaler(shift_epsilon=1e-6),
        transform_data=transform_data,
        expected_result={
            "x1": [0.0, 1.0, 2.0, 3.0],
            "x2": [0.69897, 1.69897, 2.69897, 3.69897],
        },
        fit_data=transform_data.copy(),
        numeric_allclose=True,
    )


def test_log10_scaler_zero_data():
    epsilon = 1e-6
    transform_data = {
        "y1": [0.0, 1.0, 10.0],
        "category": ["A", "B", "C"],
    }

    # log10(0 + 1e-6) = -6.0
    # log10(1 + 1e-6) ≈ 0.0
    # log10(10 + 1e-6) ≈ 1.0

    assert_action_pipeline(
        action=Log10Scaler(shift_epsilon=epsilon),
        transform_data=transform_data,
        expected_result={
            "y1": [-6.0, 0.000000434294, 1.000000043429],
            "category": ["A", "B", "C"],
        },
        fit_data=transform_data.copy(),
        numeric_allclose=True,
        atol=1e-8,
    )


def test_log10_scaler_negative_data():
    epsilon = 1e-6
    shift = 10.0 + epsilon

    transform_data = {
        "z1": [-10.0, -9.0, 0.0, 90.0],
        "z2": [10.0, 10.0, 10.0, 10.0],
    }

    # -10.0 -> log10(0.000001) = -6.0
    # -9.0 -> log10(1.000001) ≈ 0.0
    # 0.0 -> log10(10.000001) ≈ 1.0
    # 90.0 -> log10(100.000001) ≈ 2.0

    assert_action_pipeline(
        action=Log10Scaler(shift_epsilon=epsilon),
        transform_data=transform_data,
        expected_result={
            "z1": [-6.0, 0.000000434294, 1.000000043429, 2.000000004342],
            "z2": [1.30103, 1.30103, 1.30103, 1.30103],  # log10(10 + shift)
        },
        fit_data=transform_data.copy(),
        numeric_allclose=True,
    )


def test_log10_scaler_transform_without_fit():
    transform_data = {"a": [1, 2, 3], "b": [10, 20, 30]}

    assert_action_pipeline(
        Log10Scaler(columns=["a"]),
        transform_data,
        transform_data.copy(),
    )


def test_log10_scaler_column_selection():
    input_data = {
        "to_scale": [1.0, 10.0],
        "ignore": [5, 50],
        "category": ["X", "Y"],
    }

    assert_action_pipeline(
        action=Log10Scaler(columns=["to_scale"], shift_epsilon=1e-6),
        transform_data=input_data,
        expected_result={
            "to_scale": [0.0, 1.0],
            "ignore": [5.0, 50.0],
            "category": ["X", "Y"],
        },
        fit_data=input_data.copy(),
        numeric_allclose=True,
    )
