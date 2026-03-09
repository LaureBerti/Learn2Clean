from typing import NamedTuple

import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from learn2clean.loaders import DatasetLoader


class SplitData(NamedTuple):
    """Container for data splits."""

    X: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y: pd.Series | None
    y_train: pd.Series | None
    y_test: pd.Series | None


def load_and_split_data(cfg: DictConfig) -> SplitData:
    """
    Loads the dataset via Hydra config and splits it into Train/Test sets.
    Handles both Supervised (with target) and Unsupervised (no target) cases.

    Args:
        cfg: The Hydra configuration containing 'dataset' and 'experiment' blocks.

    Returns:
        SplitData: A named tuple containing (X, X_train, X_test, y, y_train, y_test).
    """
    loader: DatasetLoader = instantiate(cfg.dataset)
    df_raw = loader.load()

    target_col = loader.target_col

    if target_col:
        if target_col not in df_raw.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in loaded dataframe."
            )

        X = df_raw.drop(columns=[target_col])
        y = df_raw[target_col]
    else:
        # Unsupervised Case
        X = df_raw
        y = None

    # Retrieve parameters safely
    test_size = cfg.get("experiment", {}).get("test_size", None)
    seed = cfg.get("experiment", {}).get("seed", None)

    # Split Data: Train / Test
    # Note: We use the retrieved test_size/seed or defaults if config is missing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    return SplitData(X, X_train, X_test, y, y_train, y_test)
