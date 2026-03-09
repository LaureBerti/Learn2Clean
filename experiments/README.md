# Learn2Clean: Experimentation Platform

Welcome to the Learn2Clean experimentation platform. This directory contains all the necessary components to run, configure, and track data preparation and cleaning experiments using Reinforcement Learning. 

The platform relies heavily on **[Hydra](https://hydra.cc/)** for dynamic hierarchical configuration and (optionally) **[Weights & Biases (W&B)](https://wandb.ai/)** / **TensorBoard** for experiment tracking.

## Directory Structure Overview

Below is a detailed breakdown of the `experiments` directory and its purpose:

### 1. `configs/`
This is the core of the experimentation platform. It contains all the `.yaml` files used by Hydra to dynamically instantiate objects and define the experiment parameters.

* **`config.yaml`**: The primary entry point for Hydra. It defines the default composition (e.g., default dataset, default environment) and global variables.
* **`config_debug.yaml`**: A secondary entry point optimized for debugging (e.g., more verbose logging, smaller training steps).
* **`experiment/`**: Contains the composed experiment configurations. To run a new experiment, you typically create a file here overriding specific defaults (e.g., tying a specific agent to a specific dataset).
* **`action/`**: Configurations to customize an action.
* **`actions/`**: Configurations (as a list of actions) defining the available data preparation, scaling, and cleaning actions (e.g., imputation, outlier detection).
* **`agent/`**: Configurations for Reinforcement Learning agents (e.g., Stable Baselines 3 implementations like PPO or DQN).
* **`dataset/`**: Configurations detailing how to load and split datasets (e.g., CSV files, OpenML datasets like Titanic or Iris).
* **`distances/`**: Metrics configurations used to evaluate the state space or reward signals.
* **`env/`**: Configurations for the Gymnasium environments (e.g., sequential vs. permutation-based action spaces).
* **`hydra/`, `paths/`, `profiler/`, `wandb/`**: System and infrastructure configurations (logging formats, output directories, W&B project settings).
* **`tutorials/`**: Specific configurations tied to the step-by-step tutorial scripts.

### 2. `tutorials/`
This folder contains executable Python scripts designed to teach users how to use the Learn2Clean platform step-by-step. They serve as both documentation and functional integration tests.
* **`01_titanic_csv_dummy.py` to `05_titanic_wandb_benchmark_full.py`**: Introductory scripts showing how to load data, apply dummy actions, and integrate W&B tracking.
* **`06_sequential_gymnasium_env.py` to `10_permutations_sb3_dqn.py`**: Advanced scripts demonstrating the integration with Gymnasium environments and RL agents (Stable Baselines 3).
* **`titanic_accuracy_reward.py`**: An example of how to implement a custom reward function based on downstream machine learning model accuracy.

### 3. `tools/`
Contains utility scripts and helper functions used across various experiments to keep the main run scripts clean and DRY (Don't Repeat Yourself).
* **`create_df_table.py` / `data_profiler.py`**: Utilities for data profiling and visualization.
* **`load_and_split_data.py`**: Standardized data loading mechanisms.
* **`instantiate_list.py`**: Helper to instantiate a list of Hydra objects.
* **`setup_wandb_run.py`**: Boilerplate code to initialize Weights & Biases seamlessly with Hydra dictionaries.

### 4. `outputs/`
This folder is automatically generated and managed by Hydra and our tracking tools. It stores all artifacts from your runs.
* **`YYYY-MM-DD/`**: Hydra organizes standard runs by date and time. Inside, you will find run-specific logs, the exact `config.yaml` used (for reproducibility), and saved RL models (in `.zip` format under `models/`).
* **`wandb/`**: Local cache and synchronization files for Weights & Biases.
* **`tensorboard/`**: Event files for TensorBoard visualization, usually organized by the RL agent algorithm (e.g., `PPO_1/`, `DQN_1/`).

### 5. `sandbox/`
A playground directory meant for developers to test snippets of code, inspect datasets, or debug Hydra configurations (`inspect_hydra_config.py`, `inspect_dataset.py`) without cluttering the main experiment logic. Code here is generally transient.

---

## How to run an experiment

Thanks to Hydra, running an experiment is as simple as executing a Python script and overriding parameters via the CLI:

```bash
# Run a basic tutorial
poetry run python experiments/tutorials/09_sequential_sb3_ppo.py

# Override the dataset directly from the command line
poetry run python experiments/tutorials/09_sequential_sb3_ppo.py dataset=iris_openml
```