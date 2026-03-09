import logging
import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from learn2clean.actions.data_frame_action import DataFrameAction
from learn2clean.configs import register_all_configs
from learn2clean.loaders.base import DatasetLoader

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="tutorials/01_titanic_csv_dummy",
)
def main(cfg: DictConfig) -> None:
    """
    Tutorial 01: Basics - Loading Data & Applying a Single Action.

    This script demonstrates the core mechanism of Learn2Clean without any RL agent:
    1. Loading a dataset via Hydra configuration.
    2. Instantiating a specific cleaning action (e.g., Drop, Impute) via config.
    3. Applying the Fit/Transform pattern to modify the data.
    """
    log.info(f"STARTING TUTORIAL: {cfg.experiment.name}")

    # --- 1. Load Data ---
    log.info("Loading Dataset...")
    # Hydra 'instantiate' reads the 'dataset' section in the yaml and creates the object
    loader: DatasetLoader = instantiate(cfg.dataset)

    df_raw = loader.load()
    log.info(f"Data loaded successfully! Shape: {df_raw.shape}")

    print("\n--- RAW DATA (Head) ---")
    print(df_raw.head(3))

    # --- 2. Instantiate Action ---
    log.info("Initializing Action...")
    # Similarly, we instantiate the action defined in the config (e.g. 'drop_rows')
    action: DataFrameAction = instantiate(cfg.action)
    log.info(f"Action ready: '{action.name}'")

    # --- 3. Execute Action (Fit & Transform) ---
    # We work on a copy to preserve the original dataframe for comparison
    df_input = df_raw.copy()

    log.info("Applying Action (Fit)...")
    # 'fit' prepares the action (e.g. calculating mean for imputation)
    # Note: Target 'y' is optional for many unsupervised cleaning actions
    action.fit(df_input)

    log.info("Applying Action (Transform)...")
    # 'transform' applies the change and returns a new DataFrame
    df_processed = action.transform(df_input)

    log.info(f"Action applied! New Shape: {df_processed.shape}")

    print(f"\n--- PROCESSED DATA (Head) - After {action.name} ---")
    print(df_processed.head(3))

    # --- 4. Validation & Comparison ---
    print("\n--- SUMMARY ---")
    rows_diff = len(df_raw) - len(df_processed)
    cols_diff = len(df_raw.columns) - len(df_processed.columns)

    if df_raw.equals(df_processed):
        log.warning("No changes detected. (Did you use a Dummy action?)")
    else:
        log.info("SUCCESS: Data modification detected.")
        print(f"Rows removed: {rows_diff}")
        print(f"Columns removed: {cols_diff}")

    log.info("TUTORIAL COMPLETED")


if __name__ == "__main__":
    os.environ["PROJECT_ROOT"] = str(Path(__file__).resolve().parents[2])
    register_all_configs()
    main()
