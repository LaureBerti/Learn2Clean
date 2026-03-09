import pandas as pd
import numpy as np
from learn2clean.distance.missingness import MissingnessRatioDistance
from tests.utils.distance.assert_distance_metric import assert_distance_metric


def test_missingness_ratio_identity():
    """
    Test 1: Identical missingness ratio.

    P: 50% missing (2 NaNs / 4 cells)
    Q: 50% missing (same structure)

    Expected Distance: |0.5 - 0.5| = 0.0
    """
    # 2 rows, 2 cols = 4 cells
    data_p = {"A": [1, np.nan], "B": [np.nan, 4]}
    data_q = {
        "A": [10, np.nan],
        "B": [np.nan, 40],
    }  # Values differ, but NaN ratio is same

    assert_distance_metric(MissingnessRatioDistance(), data_p, data_q, 0.0)


def test_missingness_ratio_full_cleanup():
    """
    Test 2: Complete Imputation (Dirty -> Clean).

    P: 2 NaNs out of 4 cells (Ratio 0.5)
    Q: 0 NaNs out of 4 cells (Ratio 0.0)

    Expected Distance: |0.0 - 0.5| = 0.5
    """
    data_p = {"A": [1, np.nan], "B": [np.nan, 4]}  # Ratio 0.5
    data_q = {"A": [1, 2], "B": [3, 4]}  # Ratio 0.0

    assert_distance_metric(MissingnessRatioDistance(), data_p, data_q, 0.5)


def test_missingness_ratio_degradation():
    """
    Test 3: Introduction of NaNs (Clean -> Dirty).

    P: Clean (Ratio 0.0)
    Q: 1 NaN out of 4 cells (Ratio 0.25)

    Expected Distance: |0.25 - 0.0| = 0.25
    """
    data_p = {"A": [1, 2], "B": [3, 4]}  # Ratio 0.0
    data_q = {"A": [1, 2], "B": [3, np.nan]}  # Ratio 0.25

    assert_distance_metric(MissingnessRatioDistance(), data_p, data_q, 0.25)


def test_missingness_ratio_empty_dataframe():
    """
    Test 4: Empty DataFrame (Guard Clause check).

    If P or Q is empty (size=0), the code handles division by zero and returns ratio 0.0.
    """
    df_empty = pd.DataFrame()

    # Case A: Both empty
    assert_distance_metric(MissingnessRatioDistance(), df_empty, df_empty, 0.0)

    # Case B: P empty (Ratio 0) vs Q full NaNs (Ratio 1)
    # We need at least 1 cell to have 100% NaNs
    df_nan = pd.DataFrame({"A": [np.nan]})  # 1 cell, 1 NaN -> Ratio 1.0

    assert_distance_metric(
        MissingnessRatioDistance(),
        df_empty,  # size=0 -> ratio=0.0
        df_nan,  # size=1 -> ratio=1.0
        1.0,
    )
