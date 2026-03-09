.. image:: ./docs/images/learn2clean-text.png

-----------------------

# Learn2Clean: Automated Data Cleaning with Reinforcement Learning


|Documentation Status| |PyPI version| |Build Status| |GitHub Issues| |codecov| |License|


**Learn2Clean V2** is a modular Python framework designed to optimize data preparation pipelines using **Deep Reinforcement
Learning (DRL)**. It wraps standard data cleaning tasks (Imputation, Deduplication, Outlier Removal, Normalization) into
**Gymnasium environments**, allowing agents (PPO, DQN) to learn optimal cleaning strategies automatically.

Learn2Clean V1 is a Python library for data preprocessing and cleaning based on Q-Learning, a model-free reinforcement learning technique. It selects, for a given dataset, a ML model, and a quality performance metric, the optimal sequence of tasks for preparing the data such that the quality of the ML model result is maximized. 

.. image:: ./docs/images/figure_Learn2Clean.jpeg


**For more details** about V1, please refer to the paper presented at the Web Conf 2019 and the related tutorial.

- Laure Berti-Equille. Learn2Clean: Optimizing the Sequence of Tasks for Web Data Preparation. Proceedings of the Web Conf 2019, San Francisco, May 2019. `Preprint <https://github.com/LaureBerti/Learn2Clean/tree/master/docs/publications/theWebConf2019-preprint.pdf>`__ 

- Laure Berti-Equille. ML to Data Management: A Round Trip. Tutorial Part I, ICDE 2018. `Tutorial <https://github.com/LaureBerti/Learn2Clean/tree/master/docs/publications/tutorial_ICDE2018.pdf>`__ 




---

## Key Features

* **Hydra-Powered Configuration**: Compose complex experiments (Datasets + Actions + Agents) using simple YAML files.
* **Gymnasium Environments**:
    * **Sequential**: Step-by-step pipeline construction (MDP).
    * **Permutation**: Combinatorial pipeline selection (Contextual Bandit).
* **Deep RL Integration**: Seamless compatibility with **Stable Baselines3** (PPO, DQN).
* **Comprehensive Benchmarking**: Track data drift (Wasserstein distance), model performance, and pipeline quality
  using **Weights & Biases**.
* **🔌 Universal Data Loaders**: Native support for **CSV**, **OpenML**, **Kaggle**, and **Hugging Face** datasets.

---

## 📦 Requirements

Before starting, make sure you have:

* **Python >=3.11, <3.14**
* **Poetry** (Dependency Manager)

### Install Poetry

If Poetry is not yet installed, run:

```bash
pipx install poetry
```

Or with the official installer

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

> Or follow the official
> guide: [https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation)

---

## Project Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/Learn2Clean.git](https://github.com/your-username/Learn2Clean.git)
   cd Learn2Clean
   ```

2. **Install dependencies via Poetry:**
   ```bash
   poetry install
   ```
   > This creates a virtual environment and installs all required libraries (Hydra, SB3, Pandas, Scikit-learn, etc.).

## 🚀 Weights & Biases (W&B) Setup Guide

Learn2Clean uses **Weights & Biases** for experiment tracking, hyperparameter logging, and visualization of the
Q-Learning optimization process. Follow these steps to get your environment ready.

### 1. Create a W&B Account

Before running the project, you need a W&B account:

1. Go to [wandb.ai/site](https://wandb.ai/site) and click **Sign Up**.
2. You can use your GitHub account, Google account, or an email address.
3. Once logged in, go to your **[User Settings](https://wandb.ai/settings)** (scroll down to the "API keys" section) or
   visit [wandb.ai/authorize](https://wandb.ai/authorize).
4. **Copy your API Key**; you will need it for the environment configuration.

### 2. Configure Environment Variables

To keep your API Key secure and avoid hardcoding secrets, we use a `.env` file.

1. Create a file named `.env` in the root directory of the project:

```bash
touch .env
```

2. Add your W&B API Key to the file:

```env
WANDB_API_KEY=your_secret_api_key_here
```

### 3. Authentication & Initialization

You can log in to W&B using the CLI through Poetry:

```bash
poetry run wandb login
```

*(If the `.env` is set up correctly, W&B will automatically detect the key in many environments, but manual login
ensures the session is active.)*

## Tutorials

The best way to learn the framework is to follow the **10-step tutorial series** located in `experiments/tutorials/`.

👉 **[Start the Step-by-Step Guide Here](docs/tutorials/01_foundations.md)** 👈

👉 **[See a detailed documentation Here](docs/index.md)** 👈

| ID     | Script                               | Description                                                                 |
|:-------|:-------------------------------------|:----------------------------------------------------------------------------|
| **01** | `01_titanic_csv_dummy.py`            | **Hello World**: Load a CSV and apply a single action.                      |
| **02** | `02_titanic_openml_dummy.py`         | **Hydra Basics**: Swap datasets (OpenML) & override params via config.      |
| **03** | `03_titanic_benchmark.py`            | **Action Space**: Run *every* available cleaning tool on Titanic.           |
| **04** | `04_titanic_wandb_benchmark.py`      | **Tracking**: Log distance metrics (Wasserstein) to **WandB**.              |
| **05** | `05_titanic_wandb_benchmark_full.py` | **Deep Analysis**: Generate impact heatmaps for all distance metrics.       |
| **06** | `06_sequential_gymnasium_env.py`     | **RL Env**: Interact with the `SequentialCleaningEnv` manually.             |
| **07** | `07_permutation_space.py`            | **Math**: Visualize the combinatorial explosion of pipelines.               |
| **08** | `08_permutation_gymnasium_env.py`    | **Bandit Env**: Interact with the `PermutationsCleaningEnv`.                |
| **09** | `09_sequential_sb3_ppo.py`           | **Deep RL**: Train a **PPO Agent** to build a cleaning pipeline.            |
| **10** | `10_permutations_sb3_dqn.py`         | **Bandit RL**: Train a **DQN Agent** to pick the best pipeline in one shot. |

### How to Run a Tutorial

Use `poetry run` to execute any script with the correct environment:

```bash
# Example: Train the PPO agent
poetry run python experiments/tutorials/09_sequential_sb3_ppo.py
```

---

## ⚙️ Configuration (Hydra)

Learn2Clean uses **Hydra** to manage configurations. All config files are in `experiments/configs/`.

You can override any parameter from the command line.

**Example 1: Change the dataset to OpenML**

```bash
poetry run python experiments/tutorials/01_titanic_csv_dummy.py dataset=openml
```

**Example 2: Change the RL Agent hyperparameters**

```bash
poetry run python experiments/tutorials/09_sequential_sb3_ppo.py \
    agent.params.learning_rate=0.0005 \
    experiment.total_timesteps=10000
```

---

## 🧰 Project Structure

```text
Learn2Clean/
├── data/                       # Local datasets and documentation
│   ├── README.md
│   └── titanic.csv
│
├── docs/                       # Project documentation
│   ├── actions/                # Documentation for atomic actions
│   ├── tutorials/              # Markdown guides for the 10-step tutorial
│   │   ├── 01_foundations.md
│   │   ├── 02_benchmarking.md
│   │   ├── 03_environments.md
│   │   └── 04_reinforcement_learning.md
│   └── index.md
│
├── experiments/                # Experimentation & Tutorials Zone
│   ├── configs/                # Hydra Configuration Files (.yaml)
│   │   ├── action/             # Single action overrides
│   │   ├── actions/            # Action sets (cleaning, preparation...)
│   │   ├── agent/              # RL Agent params (PPO, DQN)
│   │   ├── dataset/            # Data source definitions
│   │   ├── distances/          # Metric definitions
│   │   ├── env/                # Environment settings
│   │   ├── experiment/         # Global experiment params
│   │   ├── hydra/              # Hydra logging & output settings
│   │   ├── paths/              # Project path management
│   │   ├── profiler/           # Data profiling settings
│   │   ├── tutorials/          # Specific configs for the 10 tutorials
│   │   ├── wandb/              # Weights & Biases settings
│   │   └── config.yaml         # Entry point config
│   │
│   ├── outputs/                # Artifacts generated by runs (Logs, Models)
│   ├── sandbox/                # Inspection scripts
│   ├── tools/                  # Helper scripts (WandB setup, Profiling, etc.)
│   └── tutorials/              # The 10 Step-by-step executable scripts
│       ├── 01_titanic_csv_dummy.py
│       ├── 02_titanic_openml_dummy.py
│       ├── 03_titanic_benchmark.py
│       ├── 04_titanic_wandb_benchmark.py
│       ├── 05_titanic_wandb_benchmark_full.py
│       ├── 06_sequential_gymnasium_env.py
│       ├── 07_permutation_space.py
│       ├── 08_permutation_gymnasium_env.py
│       ├── 09_sequential_sb3_ppo.py
│       ├── 10_permutations_sb3_dqn.py
│       └── titanic_smart_reward.py
│
├── src/
│   └── learn2clean/            # Core Library Source Code
│       ├── actions/            # Atomic Action Implementations
│       │   ├── cleaning/       # Deduplication, Imputation, Outlier, Inconsistency
│       │   ├── preparation/    # Feature Selection, Scaling
│       │   └── data_frame_action.py
│       ├── agents/             # Agent logic placeholders
│       ├── configs/            # Structured Config Classes (Dataclasses)
│       ├── distance/           # Distance Metrics (Wasserstein, Skewness, etc.)
│       ├── envs/               # Gymnasium Environments
│       │   ├── permutations_cleaning_env.py
│       │   └── sequential_cleaning_env.py
│       ├── evaluation/         # ML Models for Reward Calculation
│       │   ├── classification/
│       │   ├── clustering/
│       │   └── regression/
│       ├── loaders/            # Data Loaders (CSV, OpenML, Kaggle, HF)
│       ├── observers/          # State Observers (Data Stats)
│       ├── rewards/            # Reward Functions
│       ├── spaces/             # Action Space Logic (Permutations)
│       ├── utils/              # Logging, Wrappers & Mixins
│       └── types.py            # Type definitions
│
├── tests/                      # Unit and Integration Tests
├── .gitignore
├── mypy.ini
├── poetry.lock
├── pyproject.toml
└── README.md
```

---

## 💾 Data Loading

Learn2Clean supports multiple data sources out of the box. Configure them in `experiments/configs/dataset/`.

### 1. Local CSV

```yaml
# configs/dataset/titanic_csv.yaml
_target_: learn2clean.loaders.CSVLoader
file_path: ${hydra:runtime.cwd}/data/titanic.csv

```

### 2. OpenML

```yaml
# configs/dataset/titanic_openml.yaml
_target_: learn2clean.loaders.OpenMLLoader
name: "titanic"
version: 1

```

### 3. Kaggle

Requires `~/.kaggle/kaggle.json` or environment variables.

```yaml
# configs/dataset/kaggle_custom.yaml
_target_: learn2clean.loaders.KaggleLoader
dataset_id: "zillow/zecon"
filename: "State_time_series.csv"

```

### 4. Hugging Face

```yaml
# configs/dataset/hf_custom.yaml
_target_: learn2clean.loaders.HuggingFaceLoader
path: "julien-c/titanic-survival"
split: "train"

```

---

## 🧪 Testing

To ensure everything is working correctly, run the test suite:

```bash
poetry run pytest

```

To see coverage reports:

```bash
poetry run pytest --cov=learn2clean

```

---
---

## 🤝 Contributing

1. Create a feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```
2. Commit your changes:
   ```bash
   git commit -m "Add a new feature"
   ```
3. Push the branch and open a Pull Request.

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

.. |Documentation Status| image:: https://readthedocs.org/projects/learn2clean/badge/?version=latest
   :target: https://learn2clean.readthedocs.io/en/latest/
.. |PyPI version| image:: https://badge.fury.io/py/learn2clean.svg
   :target: https://pypi.python.org/pypi/learn2clean
.. |Build Status| image:: https://travis-ci.org/LaureBerti/Learn2Clean.svg?branch=master
   :target: https://travis-ci.org/LaureBerti/Learn2Clean
.. |GitHub Issues| image:: https://img.shields.io/github/issues/LaureBerti/Learn2Clean.svg
   :target: https://github.com/LaureBerti/Learn2Clean/issues
.. |codecov| image:: https://codecov.io/gh/LaureBerti/Learn2Clean/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/LaureBerti/Learn2Clean
.. |License| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://github.com/LaureBerti/Learn2Clean/blob/master/LICENSE
 

## 🪪 License

This project is distributed under the **X License**.

---

## 👤 Author

**Laure Berti**  
[LaureBerti](https://github.com/LaureBerti)
