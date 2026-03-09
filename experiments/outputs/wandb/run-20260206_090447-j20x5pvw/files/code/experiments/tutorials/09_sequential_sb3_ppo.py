import logging
import os
from pathlib import Path

import hydra
import wandb
from gymnasium.utils.env_checker import check_env
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback

from experiments.tools.instantiate_list import instantiate_list
from experiments.tools.load_and_split_data import load_and_split_data
from experiments.tools.setup_wandb_run import setup_wandb_run
from experiments.tutorials.titanic_accuracy_reward import TitanicAccuracyReward
from learn2clean.actions.data_frame_action import DataFrameAction
from learn2clean.configs import register_all_configs
from learn2clean.envs.sequential_cleaning_env import SequentialCleaningEnv
from learn2clean.observers.data_stats_observer import DataStatsObserver

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="tutorials/09_sequential_sb3_ppo",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    """
    Tutorial 09: Sequential Decision Making with PPO (Stable Baselines3).

    Objective:
        Train a Deep Reinforcement Learning Agent (PPO) to clean the Titanic dataset.
        The agent must learn a sequence of actions (Pipeline) to maximize the
        accuracy of a Decision Tree classifier while retaining data.

    Key Concepts:
        1. Sequential Environment: The agent takes step-by-step actions.
        2. Smart Reward: Guiding the agent via 'Learnability' and 'Accuracy'.
        3. PPO Algorithm: Optimizing the policy using a neural network.
    """
    log.info(f"STARTING TUTORIAL: {cfg.experiment.name}")

    # 1. Setup WandB (Experiment Tracking)
    run = setup_wandb_run(cfg)

    try:
        # --- PATH CONFIGURATION ---
        tensorboard_dir = Path(cfg.paths.tensorboard_root_dir)
        models_dir = Path(cfg.paths.models_dir)

        # Create directories if they don't exist
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"TensorBoard logs directory: {tensorboard_dir}")
        log.info(f"Models directory: {models_dir}")

        # --- 2. LOAD DATA & ACTIONS ---
        log.info("Loading Data and Actions...")
        data = load_and_split_data(cfg)
        actions: list[DataFrameAction] = instantiate_list(cfg.actions)
        log.info(f"Loaded {len(actions)} available actions.")

        # --- 3. INSTANTIATE ENVIRONMENT COMPONENTS ---
        log.info("Initializing Components (Reward & Observer)...")

        # TitanicSmartReward: Handles the logic for calculating utility (Accuracy * Retention)
        reward_computer = TitanicAccuracyReward(data.X, data.y)

        # DataStatsObserver: Generates the state (observation) for the agent
        observer = DataStatsObserver()

        log.info("Initializing SequentialCleaningEnv...")
        raw_env = SequentialCleaningEnv(
            X=data.X,
            y=data.y,
            actions=actions,
            reward_calculator=reward_computer,
            observer=observer,
            max_steps=cfg.env.max_steps,
            render_mode=cfg.env.render_mode,
            penalty_error=cfg.env.penalty_error,
            penalty_repetition=cfg.env.penalty_repetition,
        )

        # Validate that the environment follows the standard Gym API
        check_env(raw_env, skip_render_check=True)

        # --- 4. WRAP ENVIRONMENT FOR SB3 ---
        # Monitor: Logs episode stats (reward, length) for SB3 and WandB
        env = Monitor(raw_env)

        # DummyVecEnv: Vectorizes the environment (Batch size = 1).
        # Essential to keep consistent API (obs, rewards, dones, infos) for the agent.
        # SB3 algorithms expect vectorized environments.
        env = DummyVecEnv([lambda: env])

        # --- 5. INITIALIZE PPO AGENT ---
        log.info(f"Initializing PPO Agent with policy: {cfg.agent.policy}")

        model = PPO(
            env=env,
            tensorboard_log=str(tensorboard_dir),
            seed=cfg.experiment.seed,
            # We inject parameters from configs/agent/ppo_sequential.yaml
            policy=cfg.agent.policy,  # Must be 'MultiInputPolicy' for Dict observations
            **cfg.agent.params,  # Learning rate, gamma, n_steps, etc.
        )

        # --- 6. TRAINING LOOP ---
        total_timesteps = cfg.experiment.get("total_timesteps", 5000)
        log.info(f"Starting training for {total_timesteps} timesteps...")

        # WandB Callback for cloud logging
        wandb_callback = WandbCallback(
            gradient_save_freq=100,
            model_save_path=str(models_dir),
            verbose=2,
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=[wandb_callback],
            progress_bar=True,
        )

        log.info("Training Finished.")

        # --- 7. DEMONSTRATION & EVALUATION ---
        log.info("\n" + "=" * 60)
        log.info("   DEMONSTRATION: BEST LEARNED STRATEGY")
        log.info("=" * 60)

        # Reset the environment for a fresh start
        obs = env.reset()

        # Access inner env to access custom attributes (render, last_score)
        # Hierarchy: DummyVecEnv -> Monitor -> SequentialCleaningEnv
        inner_env = env.envs[0].unwrapped

        # Hack: Force score initialization for cleaner rendering of the Start state
        initial_score = reward_computer.calculate_score(
            inner_env.current_X, inner_env.current_y
        )
        inner_env.last_score = initial_score

        # Show initial state
        inner_env.render()

        done = False
        step_count = 0
        total_reward = 0.0
        final_utility_score = 0.0  # Default value
        actions_taken = []

        while not done:
            # Predict action using the trained model.
            # deterministic=True is CRITICAL: it forces the agent to exploit its
            # best knowledge without random exploration.
            action, _ = model.predict(obs, deterministic=True)

            # Step the environment
            # Note: returns arrays because of DummyVecEnv
            obs, rewards, dones, infos = env.step(action)

            # Extract scalar values
            reward_val = float(rewards[0])
            done = bool(dones[0])
            info = infos[0]

            # Get human-readable action name
            action_idx = int(action[0])
            action_name = inner_env.actions[action_idx].name
            actions_taken.append(action_name)

            # Capture the score BEFORE reset happens (if done is True)
            current_score_from_info = info.get("score", 0.0)
            final_utility_score = current_score_from_info

            step_count += 1
            total_reward += reward_val

            # Log the step
            log.info(
                f"Step {step_count}: {action_name} -> Reward: {reward_val:+.4f} | Msg: {info.get('msg', '')}"
            )
            if not done:
                inner_env.render()

        # --- 8. FINAL SUMMARY (THE RECIPE) ---
        print("\n" + "=" * 50)
        print("   FINAL RESULT: OPTIMAL PIPELINE DISCOVERED   ")
        print("=" * 50)
        print(f"Agent's Recipe ({len(actions_taken)} steps):")
        for i, act in enumerate(actions_taken):
            print(f"   {i+1}. {act}")

        print(f"\nFinal Cumulative Reward: {total_reward:.4f}")
        print(f"Final Utility Score:    {final_utility_score:.4f}")

        # Basic validation of the result
        if len(set(actions_taken)) < len(actions_taken) and len(actions_taken) > 2:
            print(
                "\nWARNING: The agent seems to be looping. Consider increasing training time or simplifying actions."
            )
        elif final_utility_score > 0.6:
            print("\nSUCCESS: The agent found a valid cleaning strategy!")
        else:
            print("\nFAILURE: The agent failed to produce a high-quality dataset.")

        # Save the model
        save_path = models_dir / f"ppo_sequential_{run.id}"
        model.save(save_path)
        log.info(f"Model saved to {save_path}")

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
