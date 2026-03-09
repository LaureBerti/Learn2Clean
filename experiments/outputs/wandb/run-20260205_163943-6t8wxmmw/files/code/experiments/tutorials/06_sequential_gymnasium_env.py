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
from learn2clean.envs.sequential_cleaning_env import SequentialCleaningEnv
from learn2clean.observers.data_stats_observer import DataStatsObserver
from learn2clean.rewards.base_reward import BaseReward
from learn2clean.types import Features, OptionalTarget

log = logging.getLogger(__name__)


# --- 1. REWARD CLASS ---
class CompletenessReward(BaseReward):
    """
    Simple reward class designed specifically for Tutorial 06.

    Goal: Achieve 100% Completeness.
    Logic: The utility score is defined as the ratio of non-missing values in the dataset.
    """

    def __init__(self, initial_X: Features, initial_y: OptionalTarget):
        super().__init__(initial_X, initial_y)

    def reset(self) -> None:
        """
        This reward strategy is stateless (it only looks at the current X),
        so the reset method performs no operation.
        """
        pass

    def __call__(self, X: Features, y: OptionalTarget) -> float:
        """
        Calculates the completeness ratio of the dataframe.
        Returns 0.0 if empty, 1.0 if no missing values.
        """
        if X.empty:
            return 0.0

        # Calculate the count of missing values vs total cells
        missing_cells = X.isna().sum().sum()
        total_cells = X.size

        # Calculate the ratio of "clean" (non-missing) cells
        completeness_ratio = (total_cells - missing_cells) / total_cells

        return float(completeness_ratio)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="tutorials/06_sequential_gymnasium_env",
)
def main(cfg: DictConfig) -> None:
    """
    Tutorial 06: The 'Hello World' of your RL Environment.

    Objective:
        Demonstrate how the Agent interacts with the SequentialCleaningEnv.
        We use a Random Agent (Random Policy) that picks actions blindly to
        visualize how the environment tracks state, updates data, and calculates rewards.
    """
    log.info(f"STARTING TUTORIAL: {cfg.experiment.name}")

    # A. Setup WandB (Tracking)
    setup_wandb_run(cfg)

    try:
        # B. Load Data & Actions
        # Note: Assumes 'load_and_split_data' returns an object with .X and .y attributes
        log.info("Loading Dataset...")
        data = load_and_split_data(cfg)

        log.info("Instantiating Action Space...")
        actions: list[DataFrameAction] = instantiate_list(cfg.actions)
        log.info(f"Loaded {len(actions)} available actions.")

        # C. Instantiate the Environment
        # We inject our simplified reward strategy here.
        log.info("Initializing SequentialCleaningEnv...")

        # We inject the DataStatsObserver. The Environment will configure it (n_actions).
        observer = DataStatsObserver()

        env = SequentialCleaningEnv(
            X=data.X,
            y=data.y,
            actions=actions,
            reward_calculator=CompletenessReward(data.X, data.y),
            observer=observer,
            max_steps=cfg.env.max_steps,
            render_mode=cfg.env.render_mode,
            penalty_error=cfg.env.penalty_error,
            penalty_repetition=cfg.env.penalty_repetition,
        )

        # --- D. THE AGENT LOOP (Random Policy) ---
        log.info("Starting Episode...")

        # 1. Reset: Get initial state and baseline score
        # Passing the seed ensures reproducibility of the episode
        obs, info = env.reset(seed=cfg.experiment.seed)

        print("\n--- INITIAL STATE ---")
        env.render()

        # Initialize loop variables
        step_count = 0
        total_reward = 0.0
        episode_trace = []  # List to store steps for the final WandB table

        done = False
        while not done:
            step_count += 1

            # 2. Policy: Select an action
            # For this tutorial, we sample randomly from the action space
            action_idx = env.action_space.sample()

            # Get the readable name for logging purposes
            action_name = env.actions[action_idx].name

            # 3. Step: Execute action in the environment
            # The environment handles: fit -> transform -> reward calculation -> error handling
            obs, reward, terminated, truncated, info = env.step(action_idx)

            # 4. Update Loop Variables
            done = terminated or truncated
            total_reward += reward
            current_score = info.get("score", 0.0)

            # Retrieve the message (Success, Error, or Invalid) from the environment
            msg = info.get("msg", "Success")
            is_success = "Success" in msg

            # 5. Logging & Rendering
            log.info(
                f"Step {step_count}: Applied '{action_name}' "
                f"-> Reward: {reward:+.4f} | Score: {current_score:.4f} | Msg: {msg}"
            )

            # Visual render of the new state
            env.render()

            # Send metrics to WandB
            metrics = {
                "step": step_count,
                "step_reward": reward,
                "cumulative_reward": total_reward,
                "quality_score": current_score,
                "action_index": action_idx,
                "action_name": action_name,
                "is_success": 1 if is_success else 0,
            }
            wandb.log(data=metrics, step=step_count)

            # Add step details to the trace table
            episode_trace.append(
                [
                    step_count,
                    action_name,
                    1 if is_success else 0,
                    reward,
                    current_score,
                    msg,
                ]
            )

        log.info(f"Episode Finished. Total Cumulative Reward: {total_reward:.4f}")

        # E. Final Summary
        if episode_trace:
            table = wandb.Table(
                data=episode_trace,
                columns=[
                    "Step",
                    "Action",
                    "Success",
                    "Reward",
                    "Quality Score",
                    "Message",
                ],
            )
            wandb.log({"episode_trace": table})

    except Exception as e:
        log.exception("An error occurred during the tutorial.")
        raise e
    finally:
        wandb.finish()
        log.info("WandB run closed.")


if __name__ == "__main__":
    # Ensure proper path resolution for Hydra config loading
    os.environ["PROJECT_ROOT"] = str(Path(__file__).resolve().parents[2])
    register_all_configs()
    main()
