import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from experiments.tools.instantiate_list import instantiate_list
from experiments.tools.load_and_split_data import load_and_split_data
from learn2clean.actions.data_frame_action import DataFrameAction
from learn2clean.configs import register_all_configs
from learn2clean.distance.wasserstein import WassersteinDistance

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="tutorials/03_titanic_benchmark",
)
def main(cfg: DictConfig) -> None:
    """
    Tutorial 03: Benchmark of all available actions on the Titanic dataset.

    This script serves as an integration test and a demonstration of the
    'Action Space'. It iterates through every action defined in the configuration,
    applies it to the dataset, and measures the statistical distance (Data Drift)
    induced by the transformation.

    Key Concepts:
    - **Hydra Composition**: Loading a list of actions from a config group.
    - **Fit/Transform**: Following the Scikit-Learn API standard to prevent data leakage.
    - **Wasserstein Distance**: Quantifying the magnitude of change applied to the data.

    Args:
        cfg (DictConfig): The Hydra configuration object containing dataset,
                          experiment settings, and the action space.
    """
    log.info(f"STARTING EXPERIMENT: {cfg.experiment.name}")

    # --------------------------------------------------------------------------
    # 1. DATA LOADING & PREPARATION
    # --------------------------------------------------------------------------
    log.info("[1] Loading Dataset...")
    data = load_and_split_data(cfg)

    # Initialize the metric to measure data distortion (Earth Mover's Distance)
    distance_metric = WassersteinDistance()

    # --------------------------------------------------------------------------
    # 2. ACTION SPACE INSTANTIATION
    # --------------------------------------------------------------------------
    log.info("[2] Instantiate all Actions...")

    # Hydra composition returns a Dictionary (index -> Config) for lists.
    # We resolve the config, convert values to a list, and instantiate them.
    actions: list[DataFrameAction] = instantiate_list(cfg.actions)

    log.info(f"Actions initialized successfully: {len(actions)} actions found.")

    # --------------------------------------------------------------------------
    # 3. BENCHMARK LOOP
    # --------------------------------------------------------------------------
    for action in actions:
        log.info(f"--- Processing: {action.name} ---")
        try:
            # A. Fit: Learn statistics from the Training set
            # (e.g., Calculate the mean of 'Age' from X_train)
            action.fit(data.X_train, data.y_train)

            # B. Transform: Apply the logic to the Test set
            X_test_after = action.transform(data.X_test)

            # C. Measure Impact (Distance)
            # We calculate how much the distribution shifted between
            # the original X_test and the transformed version.
            try:
                # Wasserstein distance gives a magnitude of change.
                # ~0.0: Minimal change (e.g. Imputation of few rows)
                # >1.0: Major change (e.g. Standardization/Normalization)
                dist = distance_metric.calculate(data.X_test, X_test_after)
                log.info(f"Wasserstein Distance: {dist:.4f}")

            except Exception as e:
                # Distance calculation might fail if shapes differ (Feature Selection)
                # or if data types became incompatible.
                log.warning(f"Distance Calculation Error: {e}")

        except Exception as e:
            # If an action is incompatible with the dataset (e.g. OneHotEncoding
            # on a dataset with only numbers), we catch it and continue.
            log.warning(f"Action failed: {e}")
            pass

    log.info("EXPERIMENT COMPLETED")


if __name__ == "__main__":
    # Set the project root environment variable for Hydra relative paths
    os.environ["PROJECT_ROOT"] = str(Path(__file__).resolve().parents[2])
    register_all_configs()
    main()
