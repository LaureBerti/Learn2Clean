import pandas as pd
from learn2clean.distance.row_count import RowCountDistance
from tests.utils.distance.assert_distance_metric import assert_distance_metric


def test_row_count_identity():
    """
    Test 1: Identical row count.

    P: 5 rows
    Q: 5 rows

    Formula: |5 - 5| / 5 = 0.0
    """
    data_p = pd.DataFrame({"A": range(5)})
    data_q = pd.DataFrame({"A": range(5)})

    assert_distance_metric(RowCountDistance(), data_p, data_q, 0.0)


def test_row_count_loss():
    """
    Test 2: Data Loss (e.g. DropNA).

    P: 10 rows
    Q: 8 rows (2 dropped)

    Formula: |8 - 10| / 10 = 2 / 10 = 0.2 (20% change)
    """
    data_p = pd.DataFrame({"A": range(10)})
    data_q = pd.DataFrame({"A": range(8)})

    assert_distance_metric(RowCountDistance(), data_p, data_q, 0.2)


def test_row_count_total_deletion():
    """
    Test 3: Total Deletion (Catastrophic cleaning).

    P: 10 rows
    Q: 0 rows (Empty)

    Formula: |0 - 10| / 10 = 10 / 10 = 1.0 (100% change)
    """
    data_p = pd.DataFrame({"A": range(10)})
    data_q = pd.DataFrame({"A": []})  # Empty

    assert_distance_metric(RowCountDistance(), data_p, data_q, 1.0)


def test_row_count_increase():
    """
    Test 4: Row Increase (e.g. Oversampling).

    P: 100 rows
    Q: 150 rows

    Formula: |150 - 100| / 100 = 50 / 100 = 0.5 (50% change)
    """
    data_p = pd.DataFrame({"A": range(100)})
    data_q = pd.DataFrame({"A": range(150)})

    assert_distance_metric(RowCountDistance(), data_p, data_q, 0.5)


def test_row_count_initial_empty():
    """
    Test 5: Edge case where original dataset is empty.

    P: 0 rows
    Q: 5 rows

    Guard clause should return 0.0 to avoid DivisionByZero.
    (Mathematically undefined relative change, but 0.0 is safe for RL).
    """
    data_p = pd.DataFrame({"A": []})
    data_q = pd.DataFrame({"A": range(5)})

    assert_distance_metric(RowCountDistance(), data_p, data_q, 0.0)
