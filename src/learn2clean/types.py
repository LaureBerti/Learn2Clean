"""
This module centralizes custom type definitions used throughout the Learn2Clean project.
Using these semantic aliases improves code readability and ensures consistency
in function signatures across the pipeline (Evaluators, Actions, RL Agents).
"""

from typing import TypeAlias, Callable

import numpy as np
import pandas as pd

# --- Data Structures ---

# Represents the feature matrix (commonly denoted as X).
# In this project, we enforce the use of Pandas DataFrames for features
# to leverage column names and types during cleaning.
Features: TypeAlias = pd.DataFrame

# Represents an optional feature matrix.
# Useful for initialization phases (e.g., BaseReward) where data might not be loaded yet.
OptionalFeatures: TypeAlias = Features | None

# Represents the target variable or labels (commonly denoted as y).
# We accept either a Pandas Series or a NumPy array to maintain compatibility
# with Scikit-Learn's estimators.
Target: TypeAlias = pd.Series | np.ndarray

# Represents an optional target variable.
# This is useful for:
# 1. Unsupervised learning tasks (Clustering) where 'y' is not required.
# 2. Transformation steps (fit/transform) where 'y' might not be present.
# 3. Inference pipelines where the ground truth is unknown.
OptionalTarget: TypeAlias = Target | None


# --- Functional Signatures ---

# Defines the signature for a Reward Function used in Reinforcement Learning.
# It takes the current state of features (X) and the optional target (y),
# and returns a scalar float value representing the 'score' or 'reward'.
# Signature: (X, y) -> float
RewardFunction: TypeAlias = Callable[[Features, OptionalTarget], float]

# Defines the signature for a standard Evaluation Metric.
# It compares the ground truth (y_true) against predictions (y_pred)
# and returns a scalar float score (e.g., Accuracy, MSE, F1-Score).
#
# Signature: (y_true, y_pred) -> float
MetricFunction: TypeAlias = Callable[[Target, Target], float]

# Union type for configuration: can be a string name (e.g. 'accuracy')
# or a direct function reference.
MetricType: TypeAlias = str | MetricFunction