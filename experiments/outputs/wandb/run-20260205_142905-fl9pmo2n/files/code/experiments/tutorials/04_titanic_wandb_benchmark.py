import logging
import os
from pathlib import Path

import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from experiments.tools.instantiate_list import instantiate_list
from experiments.tools.load_and_split_data import load_and_split_data
from learn2clean.actions.data_frame_action import DataFrameAction
from learn2clean.configs import register_all_configs
from learn2clean.distance.base_distance import BaseDistance

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="tutorials/04_titanic_wandb_benchmark",
)
def main(cfg: DictConfig) -> None:
    """
    Tutorial 04: Visual Benchmark with WandB.

    Runs all actions defined in the 'Action Space' against the Titanic dataset.
    Logs the Execution Status, the Data Drift (Wasserstein Distance), and a
    visual preview of the data to a Weights & Biases dashboard.
    """
    distance: BaseDistance = instantiate(cfg.distance)

    # 1. INIT WANDB
    run = wandb.init(
        name=cfg.experiment.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        **cfg.wandb,
    )

    log.info(f"WANDB RUN: {run.name}")

    # 2. DATA LOADING
    data = load_and_split_data(cfg)

    # 3. ACTIONS INSTANTIATION
    log.info("Instantiate all Actions...")

    # Hydra composition returns a Dictionary (index -> Config) for lists.
    # We resolve the config, convert values to a list, and instantiate them.
    actions: list[DataFrameAction] = instantiate_list(cfg.actions)

    # --- TABLEAU WANDB ---
    columns = [
        "Action Name",
        "Category",
        "Status",
        f"Distance ({distance.name})",
        "Error Msg",
        "Preview (Head)",
    ]
    benchmark_table = wandb.Table(columns=columns)

    log.info(f"Benchmarking {len(actions)} actions...")

    for action in actions:
        log.info(f"--- Processing: {action.name} ---")
        action_name = action.name
        category = action.logical_path
        status = "Success"
        dist = -1.0
        error_msg = ""
        preview_html = ""

        try:
            # A. Fit: Learn statistics from the Training set
            # (e.g., Calculate the mean of 'Age' from X_train)
            action.fit(data.X_train, data.y_train)

            # B. Transform: Apply the logic to the Test set
            X_test_transformed = action.transform(data.X_test)

            preview_html = X_test_transformed.head(3).to_html()

            # C. Measure Impact (Distance)
            # We calculate how much the distribution shifted between
            # the original X_test and the transformed version.
            try:
                # Wasserstein distance gives a magnitude of change.
                # ~0.0: Minimal change (e.g. Imputation of few rows)
                # >1.0: Major change (e.g. Standardization/Normalization)
                dist = distance.calculate(data.X_test, X_test_transformed)
                log.info(f"Distance ({distance.name}): {dist:.4f}")

            except Exception as e:
                log.warning(f"Distance Calculation Error: {e}")
                error_msg = f"Distance Error: {str(e)}"

        except Exception as e:
            status = "Failed"
            error_msg = str(e)
            log.warning(f"Action failed: {e}")

        benchmark_table.add_data(
            action_name,
            category,
            status,
            dist,
            error_msg,
            (
                wandb.Html(preview_html)
                if status == "Success"
                else wandb.Html("<i>No Preview (Failed)</i>")
            ),
        )

    log.info("EXPERIMENT COMPLETED")
    wandb.log({"benchmark_results": benchmark_table})
    wandb.log(
        {
            f"{distance.name}_comparison": wandb.plot.bar(
                benchmark_table,
                "Action Name",
                f"Distance ({distance.name})",
                title=f"Data Drift by Action ({distance.name})",
            )
        }
    )

    wandb.finish()
    log.info("Benchmark sent to WandB. Check your dashboard!")


if __name__ == "__main__":
    # Set the project root environment variable for Hydra relative paths
    os.environ["PROJECT_ROOT"] = str(Path(__file__).resolve().parents[2])
    register_all_configs()
    main()
