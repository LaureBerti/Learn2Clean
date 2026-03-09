import pandas as pd


def prepare_numeric_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts numeric columns, drop NaN rows,
    ensures minimal sample size.
    """
    X = df.select_dtypes(include="number").dropna()

    if X.shape[0] < 5 or X.shape[1] < 1:
        return None

    return X
