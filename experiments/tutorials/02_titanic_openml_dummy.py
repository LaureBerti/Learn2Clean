import logging
import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from experiments.tools.data_profiler import DataProfiler
from learn2clean.actions.data_frame_action import DataFrameAction
from learn2clean.configs import register_all_configs
from learn2clean.loaders.base import DatasetLoader

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="tutorials/02_titanic_openml_dummy",
)
def main(cfg: DictConfig) -> None:
    """
    Example experiment: Loads the Titanic OpenML dataset and applies a Dummy Action.

    Demonstrates:
    1. Instantiation of a DatasetLoader from config.
    2. Instantiation of a DataFrameAction from config.
    3. Comparison of data before and after the action.
    """

    log.info(f"STARTING EXPERIMENT: {cfg.experiment.name}")
    profiler = DataProfiler(cfg.profiler, root_dir=Path(cfg.paths.run_output_dir))

    # 1. Instantiate and Load Data
    log.info("[1] Loading Dataset...")
    loader: DatasetLoader = instantiate(cfg.dataset)
    df_raw = loader.load()
    log.info(f"Data loaded successfully!")
    log.debug(f"Source: {loader.get_source_info()}")
    log.debug(f"Shape: {df_raw.shape}")
    log.debug(df_raw.head(3))
    profiler.run(df_raw, step_name="raw_features")

    # 2. Instantiate Action
    log.info("[2] Initializing Action...")
    action: DataFrameAction = instantiate(cfg.action)
    log.info(f"Action instantiated: {action.name}")

    # 3. Fit Action
    log.info("[3] Applying Action fit...")
    action.fit(df_raw)

    # 4. Apply Action transform
    log.info("[4] Applying Action transform...")
    df_processed = action.transform(df_raw)

    log.info(f"Action applied successfully!")
    log.debug(f"New Shape: {df_processed.shape}")
    log.debug(df_processed.head(3))

    # 5. Sanity Check (Simple Validation)
    log.info("[5] Validation Check...")
    if df_raw.equals(df_processed):
        log.warning("The data matches exactly (No changes detected).")
        log.warning("Check if your action is supposed to modify the data.")
    else:
        log.info("SUCCESS: Data has been modified by the action.")

    log.info("EXPERIMENT COMPLETED")


if __name__ == "__main__":
    os.environ["PROJECT_ROOT"] = str(Path(__file__).resolve().parents[2])
    register_all_configs()
    main()
