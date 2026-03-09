import logging
import os
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import plotly.express as px
import wandb
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
    config_name="tutorials/05_titanic_wandb_benchmark_full",
)
def main(cfg: DictConfig) -> None:
    """
    Tutorial 05: Benchmark of all available actions on the Titanic dataset.

    This script serves as an integration test and a demonstration of the
    'Action Space'. It iterates through every action defined in the configuration,
    applies it to the dataset, and computes all metrics on the action result.

    Key Concepts:
    - **Hydra Composition**: Loading a list of actions from a config group.
    - **Fit/Transform**: Following the Scikit-Learn API standard to prevent data leakage.
    - **WandB Integration**: Logging metrics and interactive charts.

    Args:
        cfg (DictConfig): The Hydra configuration object containing dataset,
                          experiment settings, and the action space.
    """
    log.info(f"STARTING EXPERIMENT: {cfg.experiment.name}")

    run = wandb.init(
        name=cfg.experiment.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        **cfg.wandb,
    )

    log.info(f"WANDB RUN: {run.name}")

    # --------------------------------------------------------------------------
    # 1. DATA LOADING & PREPARATION
    # --------------------------------------------------------------------------
    log.info("[1] Loading Dataset...")
    data = load_and_split_data(cfg)

    # --------------------------------------------------------------------------
    # 2. INSTANTIATION
    # --------------------------------------------------------------------------
    log.info("[2] Instantiate Actions & Metrics...")

    # Hydra composition returns a Dictionary (index -> Config) for lists.
    # We resolve the config, convert values to a list, and instantiate them.
    actions: list[DataFrameAction] = instantiate_list(cfg.actions)
    distances: list[BaseDistance] = instantiate_list(cfg.distances)

    # --------------------------------------------------------------------------
    # 3. BENCHMARK LOOP
    # --------------------------------------------------------------------------
    results_data: list[dict[str, Any]] = []
    for action in actions:
        log.info(f"--- Processing: {action.name} ---")
        row_result: dict[str, Any] = {
            "Action_Name": action.name,
            "Type": action.logical_path,
            "Status": "OK",
        }
        try:
            # A. Fit: Learn statistics from the Training set
            # (e.g., Calculate the mean of 'Age' from X_train)
            action.fit(data.X_train, data.y_train)

            # B. Transform: Apply the logic to the Test set
            X_test_transformed = action.transform(data.X_test)

            # C. Compute distances
            for distance in distances:
                try:
                    d = distance.calculate(data.X_test, X_test_transformed)
                    log.info(f"{distance.name}: {d:.4f}")
                    row_result[distance.name] = d
                except Exception:
                    # If calculation fails (e.g., distance undefined for this data), log None
                    row_result[distance.name] = None
        except Exception as e:
            # If an action is incompatible with the dataset (e.g. OneHotEncoding
            # on a dataset with only numbers), we catch it and continue.
            row_result["Status"] = "Failed"
            row_result["Error"] = str(e)
            log.warning(f"Action failed: {e}")
            pass
        results_data.append(row_result)

    # 4. PROCESS RESULTS
    df_results = pd.DataFrame(results_data)

    # --------------------------------------------------------------------------
    # 6. INTERACTIVE HEATMAP (VIA PLOTLY)
    # --------------------------------------------------------------------------

    # 1. PREPARATION
    df_viz = df_results.copy()
    # If "Action_Name" is still a column, set it as index so it's not removed by select_dtypes
    if "Action_Name" in df_viz.columns:
        df_viz.set_index("Action_Name", inplace=True)

    # 2. FILTERING & NORMALIZATION
    # Keep only numeric metrics for the heatmap
    heatmap_df = df_viz.select_dtypes(include=["number"])

    if heatmap_df.empty:
        log.warning("No numeric metrics computed. Skipping heatmap.")
    else:
        # Min-Max Normalization
        # (0 = Min value of the column, 1 = Max value of the column)
        heatmap_df_norm = (heatmap_df - heatmap_df.min()) / (
            heatmap_df.max() - heatmap_df.min()
        )
        heatmap_df_norm = heatmap_df_norm.fillna(0)

        # 3. CREATE PLOTLY CHART
        # px.imshow automatically handles axes, colors, and hover tooltips
        fig = px.imshow(
            heatmap_df_norm,
            labels=dict(x="Metrics", y="Actions", color="Impact (Norm)"),
            x=heatmap_df_norm.columns,
            y=heatmap_df_norm.index,
            color_continuous_scale="RdYlGn_r",  # Red = 1 (High Impact/Bad), Green = 0
            aspect="auto",
            title=f"Cleaning Actions Impact ({cfg.experiment.name})",
            text_auto=".2f",
        )

        # Improve layout for web/dashboard display (margins for labels)
        fig.update_layout(
            autosize=True,
            margin=dict(l=50, r=50, b=100, t=100),
        )

        # 4. LOG TO WANDB

        # Log the interactive object directly
        # Note: In recent WandB versions, pass 'fig' directly without wrapping in wandb.Plotly()
        wandb.log({f"Cleaning Actions Impact": fig})

    # Standard table logging
    wandb.log({"benchmark_results": wandb.Table(dataframe=df_results)})

    wandb.finish()
    log.info("WandB Run finished. Check your dashboard!")


if __name__ == "__main__":
    # Set the project root environment variable for Hydra relative paths
    os.environ["PROJECT_ROOT"] = str(Path(__file__).resolve().parents[2])
    register_all_configs()
    main()
