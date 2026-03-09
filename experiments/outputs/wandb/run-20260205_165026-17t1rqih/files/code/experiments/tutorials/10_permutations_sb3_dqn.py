import logging
import os
from pathlib import Path

import hydra
import wandb
from gymnasium.utils.env_checker import check_env
from omegaconf import DictConfig
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback

from experiments.tools.instantiate_list import instantiate_list
from experiments.tools.load_and_split_data import load_and_split_data
from experiments.tools.setup_wandb_run import setup_wandb_run
from learn2clean.actions.data_frame_action import DataFrameAction
from learn2clean.configs import register_all_configs
from learn2clean.envs.permutations_cleaning_env import PermutationsCleaningEnv
from learn2clean.observers.data_stats_observer import DataStatsObserver
from learn2clean.rewards.base_reward import BaseReward
from learn2clean.rewards.completeness_retention_reward import (
    CompletenessRetentionReward,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="tutorials/10_permutations_sb3_dqn",
)
def main(cfg: DictConfig) -> None:
    """
    Tutorial 10: One-Shot Pipeline Selection with DQN (Contextual Bandit).

    Concept:
        Unlike the Sequential environment (Step-by-Step), this environment
        treats Data Cleaning as a "One-Shot" problem.
        The agent selects a SINGLE integer, which corresponds to a FULL pipeline
        (e.g., "Impute -> Scale -> OutlierRemoval").

    Algorithm:
        We use Deep Q-Network (DQN) from Stable Baselines3.
        Since there is no "next state" (gamma=0), the DQN effectively acts
        as a Contextual Bandit, learning to map dataset statistics (Observation)
        to the best pipeline (Action).
    """
    # Debug: Print config
    # print(OmegaConf.to_yaml(cfg))

    log.info(f"STARTING TUTORIAL: {cfg.experiment.name}")

    # 1. Setup WandB (Experiment Tracking)
    run = setup_wandb_run(cfg)

    try:
        # --- PATH CONFIGURATION ---
        tensorboard_dir = Path(cfg.paths.tensorboard_root_dir)
        models_dir = Path(cfg.paths.models_dir)

        # Create directories
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"TensorBoard logs directory: {tensorboard_dir}")
        log.info(f"Models directory: {models_dir}")

        # --- 2. LOAD DATA & ACTIONS ---
        log.info("Loading Data and Actions...")
        data = load_and_split_data(cfg)
        actions: list[DataFrameAction] = instantiate_list(cfg.actions)
        log.info(f"Loaded {len(actions)} atomic actions.")

        # --- 3. INSTANTIATE ENVIRONMENT COMPONENTS ---
        log.info("Initializing Components (Reward & Observer)...")
        reward_computer: BaseReward = CompletenessRetentionReward(
            initial_X=data.X, initial_y=data.y
        )
        observer = DataStatsObserver()

        log.info("Generating Permutations Environment...")
        # Note: This environment generates ALL possible pipelines up to a certain length.
        # This creates a large, discrete action space (e.g., 0 to ~200).
        raw_env = PermutationsCleaningEnv(
            X=data.X,
            y=data.y,
            actions=actions,
            observer=observer,
            reward_calculator=reward_computer,
            render_mode=cfg.env.render_mode,
            penalty_error=cfg.env.penalty_error,
        )

        # Validate Gym API compliance
        check_env(raw_env, skip_render_check=True)
        log.info(f"Action Space Size: {raw_env.action_space.n} unique pipelines.")

        # --- 4. WRAP ENVIRONMENT FOR SB3 ---
        # Monitor: Logs stats per episode
        env = Monitor(raw_env)
        # DummyVecEnv: Vectorization wrapper required by SB3
        env = DummyVecEnv([lambda: env])

        # --- 5. INITIALIZE DQN AGENT ---
        log.info(f"Initializing DQN Agent with policy: {cfg.agent.policy}")

        model = DQN(
            env=env,
            tensorboard_log=str(tensorboard_dir),
            seed=cfg.experiment.seed,
            # Load specific params (e.g., gamma=0.0 for Bandit setting)
            policy=cfg.agent.policy,
            **cfg.agent.params,
        )

        # --- 6. TRAINING LOOP ---
        # In this Bandit setting, 1 Episode = 1 Decision.
        total_timesteps = cfg.experiment.n_episodes
        log.info(f"Starting DQN training for {total_timesteps} episodes (trials)...")
        log.info("NOTE: Action logs are silenced during training.")

        # Callback for checkpointing
        wandb_callback = WandbCallback(
            gradient_save_freq=100,
            model_save_path=str(models_dir),
            verbose=2,
        )

        model.learn(
            total_timesteps=total_timesteps, callback=wandb_callback, progress_bar=True
        )

        log.info("Training Finished.")

        # --- 7. FINAL EVALUATION / DEMO ---
        log.info("\n" + "=" * 60)
        log.info("   DEMONSTRATION: BEST PIPELINE SELECTION")
        log.info("=" * 60)

        # Reset for a clean inference step
        obs = env.reset()

        # Predict the best action (Deterministic = Max Q-Value)
        action_idx, _ = model.predict(obs, deterministic=True)
        selected_action = int(action_idx[0])

        # Execute the selected action in the inner environment to get details
        # Hierarchy: DummyVecEnv -> Monitor -> PermutationsCleaningEnv
        raw_inner_env = env.envs[0].unwrapped

        # Note: step() returns (obs, reward, terminated, truncated, info)
        _, final_reward, _, _, info = raw_inner_env.step(selected_action)

        # Extract readable pipeline string from the 'info' dict
        pipeline_list = info.get("pipeline", [])
        pipeline_str = " -> ".join(pipeline_list)

        # --- 8. RESULTS DISPLAY ---
        print(f"\nDQN Selected Action ID: {selected_action}")
        print(f"Pipeline Structure:   [ {pipeline_str} ]")
        print(f"Final Reward:         {final_reward:.4f}")

        # Visual Render of the result table
        raw_inner_env.render()

        # Log final metrics to WandB
        wandb.log(
            {
                "final/selected_action_id": selected_action,
                "final/pipeline_str": pipeline_str,
                "final/reward": final_reward,
            }
        )

        if final_reward > 0.6:
            print("\nSUCCESS: The agent selected a valid, high-performing pipeline.")
        else:
            print("\nFAILURE: The agent selected a suboptimal pipeline.")

        # --- 9. SAVE MODEL ---
        final_save_path = models_dir / f"dqn_permutation_{run.id}"
        model.save(final_save_path)
        log.info(f"Model saved to {final_save_path}")

    except Exception as e:
        log.exception("An error occurred during the SB3 tutorial.")
        raise e
    finally:
        if wandb.run is not None:
            wandb.finish()
        log.info("WandB run closed.")


if __name__ == "__main__":
    os.environ["PROJECT_ROOT"] = str(Path(__file__).resolve().parents[2])
    register_all_configs()
    main()
