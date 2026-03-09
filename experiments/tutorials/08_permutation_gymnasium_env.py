import logging
import os
from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig

from experiments.tools.instantiate_list import instantiate_list
from experiments.tools.load_and_split_data import load_and_split_data
from experiments.tools.setup_wandb_run import setup_wandb_run
from learn2clean.actions.data_frame_action import DataFrameAction
from learn2clean.configs import register_all_configs
from learn2clean.envs.permutations_cleaning_env import PermutationsCleaningEnv
from learn2clean.observers.data_stats_observer import DataStatsObserver
from learn2clean.rewards.completeness_retention_reward import (
    CompletenessRetentionReward,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="tutorials/08_permutation_gymnasium_env",
)
def main(cfg: DictConfig) -> None:
    """
    Tutorial 08: The Permutation Gymnasium Environment.

    Objective:
        Demonstrate the `PermutationsCleaningEnv` in action.
        This environment abstracts the combinatorial complexity (seen in Tutorial 07)
        behind a simple integer Action Space.

        We run a 'Random Bandit' experiment:
        1. We run N independent episodes.
        2. In each episode, we pick a random pipeline from the space.
        3. We measure which pipeline gives the best trade-off.
    """
    log.info(f"STARTING TUTORIAL: {cfg.experiment.name}")

    # A. Setup WandB
    run = setup_wandb_run(cfg)

    try:
        # B. Load Data & Actions
        log.info("Loading Dataset...")
        data = load_and_split_data(cfg)

        log.info("Instantiating Atomic Actions...")
        actions: list[DataFrameAction] = instantiate_list(cfg.actions)
        log.info(f"Loaded {len(actions)} atomic actions.")

        # Instantiate the custom reward functor
        reward_computer = CompletenessRetentionReward(
            initial_X=data.X, initial_y=data.y
        )

        # C. Instantiate the Permutation Environment
        log.info("Initializing PermutationsCleaningEnv...")

        # We inject the DataStatsObserver.
        observer = DataStatsObserver()

        # Note: This environment generates the combinatorial action space internally.
        env: PermutationsCleaningEnv = PermutationsCleaningEnv(
            X=data.X,
            y=data.y,
            actions=actions,
            reward_calculator=reward_computer,
            observer=observer,
            render_mode=cfg.env.render_mode,
            penalty_error=cfg.env.penalty_error,
        )

        log.info(f"Action Space Size: {env.action_space.n} unique pipelines.")

        # --- D. EXPERIMENT LOOP (Random Search) ---
        n_episodes = cfg.experiment.n_episodes
        results_trace = []

        log.info(f"Starting Random Search over {n_episodes} episodes...")

        for i in range(n_episodes):
            log.info(f"--- Episode {i+1}/{n_episodes} ---")

            # 1. Reset (New Context)
            # Seed ensures reproducibility of the random sampling
            obs, info = env.reset(seed=cfg.experiment.seed + i)

            # 2. Action: Sample a random integer (The "One-Shot")
            action_idx = env.action_space.sample()

            # 3. Decode for logging
            pipeline_tuple = env.action_space.idx_to_permutation(action_idx)
            pipeline_str = " -> ".join([a.name for a in pipeline_tuple])

            log.info(f"Testing Pipeline #{action_idx}: [ {pipeline_str} ]")

            # 4. Step: Execute the full pipeline
            obs, reward, terminated, truncated, info = env.step(action_idx)

            # 5. Metrics Extraction
            msg = info.get("msg", "")
            status = "FAILED" if info.get("error", False) else "SUCCESS"

            # Extract stats from the Observer's output
            stats = obs["dataset_stats"]
            rows, nulls = int(stats[0]), int(stats[2])

            log.info(
                f"Result: {status} | Reward: {reward:.4f} | Rows: {rows} | Msg: {msg}"
            )

            # Visual Render
            env.render()

            # 6. Logging
            wandb.log(
                {
                    "episode": i,
                    "reward": reward,
                    "pipeline_length": len(pipeline_tuple),
                    "action_index": action_idx,
                    "final_rows": rows,
                    "final_nulls": nulls,
                    "status": 1 if status == "SUCCESS" else 0,
                }
            )

            results_trace.append(
                [i, action_idx, pipeline_str, len(pipeline_tuple), status, reward]
            )

        log.info("Search Finished.")

        # E. Final Summary Table
        table = wandb.Table(
            data=results_trace,
            columns=[
                "Episode",
                "Action Index",
                "Pipeline",
                "Length",
                "Status",
                "Reward",
            ],
        )
        wandb.log({"permutation_comparison": table})

    except Exception as e:
        log.exception("An error occurred during the tutorial.")
        raise e
    finally:
        wandb.finish()
        log.info("WandB run closed.")


if __name__ == "__main__":
    os.environ["PROJECT_ROOT"] = str(Path(__file__).resolve().parents[2])
    register_all_configs()
    main()
