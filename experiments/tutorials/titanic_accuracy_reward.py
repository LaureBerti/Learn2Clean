import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from learn2clean.rewards.base_reward import BaseReward
from learn2clean.types import Features, OptionalTarget

log = logging.getLogger(__name__)


class TitanicAccuracyReward(BaseReward):
    """
    Reward function specifically tuned for the Titanic Tutorial.
    It uses a 2-tier system:
    1. Hygiene Tier (Score < 0.2): Focus on removing NaNs.
    2. ML Tier (Score > 0.5): Focus on maximizing Model Accuracy & Data Retention.
    """

    def __init__(self, initial_X: Features, initial_y: OptionalTarget):
        super().__init__(initial_X, initial_y)
        self.initial_rows = len(initial_X)
        self.last_score = 0.0

    def _evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Trains a quick Decision Tree on numeric columns only."""
        # 1. Select numeric features (Ignore text columns to avoid crashes)
        X_num = X.select_dtypes(include="number")
        if X_num.empty:
            return 0.0

        try:
            # 2. Split & Train (Hardcoded params for simplicity/speed)
            X_train, X_test, y_train, y_test = train_test_split(
                X_num, y, test_size=0.3, random_state=42, stratify=y
            )
            clf = DecisionTreeClassifier(max_depth=3, random_state=42)
            clf.fit(X_train, y_train)
            return float(accuracy_score(y_test, clf.predict(X_test)))
        except Exception:
            return 0.0

    def calculate_score(self, X: Features, y: OptionalTarget) -> float:
        """Computes the absolute utility score."""
        if X is None or X.empty:
            return 0.0

        # --- TIER 1: HYGIENE (Penalize NaNs) ---
        # If any NaN exists, the score is capped at 0.2.
        if X.isna().any().any():
            completeness = 1.0 - (X.isna().sum().sum() / X.size)
            return completeness * 0.2

        # --- TIER 2: PERFORMANCE (ML Accuracy) ---
        # No NaNs? We enter the ML zone with a massive bonus (+0.5).
        acc = self._evaluate_model(X, y)
        retention = len(X) / self.initial_rows

        # Formula: Bonus + (Accuracy * sqrt(Retention))
        return 0.5 + (acc * np.sqrt(retention))

    def __call__(self, X: Features, y: OptionalTarget) -> float:
        """Returns the Reward Delta (Improvement - Time Cost)."""
        current_score = self.calculate_score(X, y)

        # Reward = Improvement since last step - Small time penalty
        delta = current_score - self.last_score
        step_reward = delta - 0.005

        self.last_score = current_score
        return float(step_reward)

    def reset(self) -> None:
        self.last_score = 0.0
