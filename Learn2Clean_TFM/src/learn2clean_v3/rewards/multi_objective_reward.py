"""
MultiObjectiveReward — V3 improvement #4.

Scalarised weighted combination of:
  1. accuracy   — ML model cross-val score (classification / regression)
  2. retention  — row retention ratio vs. original
  3. quality    — data completeness × (1 − drift)
  4. drift      — Wasserstein drift penalty from original distribution (subtracted)

All weights are normalised so they sum to 1.  Every sub-score is in [0,1]
so the total reward is also in [0,1] before the drift penalty.

Built-in evaluation models
--------------------------
  "random_forest"       → RandomForestClassifier / RandomForestRegressor
  "logistic"            → LogisticRegression (classification only)
  "gradient_boosting"   → GradientBoostingClassifier / Regressor
  "tabpfn"              → TabPFN v2 (optional; install: pip install tabpfn>=2.0)
                          Uses a fixed train/test split (no CV) for speed.
                          TabPFN applies its own internal preprocessing so NaN
                          values are passed through without prior imputation.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

from learn2clean_v3.rewards.base_reward import BaseReward
from learn2clean_v3.types import Features, OptionalTarget, RewardComponents

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional TabPFN v2 import
# ---------------------------------------------------------------------------
try:
    import tabpfn as _tabpfn_module  # noqa: F401
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False

# Maximum rows forwarded to TabPFN per reward call — subsampled for loop speed.
_TABPFN_MAX_ROWS: int = 512

_MODELS: Dict[str, Any] = {
    "random_forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
    "logistic": LogisticRegression(max_iter=500, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
}


class MultiObjectiveReward(BaseReward):
    """
    Parameters
    ----------
    weight_accuracy : float
    weight_retention : float
    weight_quality : float
        Weights for each sub-objective (normalised internally).
    drift_penalty_coeff : float
        Coefficient for the Wasserstein drift penalty term.
    eval_model : str
        Which model to use for accuracy estimation.
        One of "random_forest", "logistic", "gradient_boosting", "tabpfn".
    eval_metric : str
        "accuracy" or "f1".
    eval_cv_folds : int
        Number of cross-validation folds (set to 1 for speed, 0 to skip).
        Ignored when eval_model="tabpfn" (always uses a single train/test split).
    """

    def __init__(
        self,
        weight_accuracy: float = 0.5,
        weight_retention: float = 0.3,
        weight_quality: float = 0.2,
        drift_penalty_coeff: float = 0.1,
        eval_model: str = "random_forest",
        eval_metric: str = "accuracy",
        eval_cv_folds: int = 3,
    ) -> None:
        raw = np.array([weight_accuracy, weight_retention, weight_quality], dtype=float)
        total = raw.sum()
        if total <= 0:
            raise ValueError("Reward weights must not all be zero.")
        self._w = raw / total

        self._drift_coeff = drift_penalty_coeff
        self._eval_model_name = eval_model
        self._eval_metric = eval_metric
        self._cv = eval_cv_folds

        self._original_X: Optional[Features] = None
        self._original_n_rows: int = 1
        self._original_n_cells: int = 1
        self._last_components: Optional[RewardComponents] = None

    # ------------------------------------------------------------------

    def reset(self, X_initial: Features, y_initial: OptionalTarget) -> None:
        self._original_X = X_initial.copy()
        self._original_n_rows = max(len(X_initial), 1)
        self._original_n_cells = max(X_initial.size, 1)
        self._last_components = None

    # ------------------------------------------------------------------

    def __call__(self, X: Features, y: OptionalTarget) -> float:
        if X is None or len(X) == 0:
            self._last_components = RewardComponents(total=-1.0)
            return -1.0

        acc = self._accuracy_score(X, y)
        ret = len(X) / self._original_n_rows
        qual = self._quality_score(X)
        drift = self._drift_score(X)

        total = (
            self._w[0] * acc
            + self._w[1] * ret
            + self._w[2] * qual
            - self._drift_coeff * drift
        )
        total = float(np.clip(total, -1.0, 1.0))

        self._last_components = RewardComponents(
            accuracy=acc,
            retention=ret,
            quality=qual,
            drift_penalty=drift,
            total=total,
        )
        return total

    @property
    def last_components(self) -> Optional[RewardComponents]:
        return self._last_components

    # ------------------------------------------------------------------
    # Sub-scores
    # ------------------------------------------------------------------

    def _accuracy_score(self, X: Features, y: OptionalTarget) -> float:
        if y is None:
            return 0.0

        # TabPFN path — dedicated method, no CV, no pre-imputation
        if self._eval_model_name == "tabpfn":
            return self._tabpfn_accuracy_score(X, y)

        numeric = X.select_dtypes(include="number")
        if numeric.shape[1] == 0:
            return 0.0
        df_clean = numeric.copy()
        df_clean = df_clean.fillna(df_clean.median())

        # Align y to X's current index (rows may have been dropped by cleaning actions)
        if isinstance(y, pd.Series):
            try:
                y_aligned = y.loc[df_clean.index]
            except KeyError:
                y_aligned = y.iloc[:len(df_clean)]
        else:
            y_arr_full = np.asarray(y)
            if len(y_arr_full) > len(df_clean):
                y_arr_full = y_arr_full[:len(df_clean)]
            y_aligned = y_arr_full

        y_arr = np.asarray(y_aligned)
        valid = ~np.isnan(y_arr.astype(float)) if y_arr.dtype != object else np.ones(len(y_arr), bool)
        if valid.sum() < 10:
            return 0.0
        X_fit = df_clean.values[valid]
        y_fit = y_arr[valid]

        le = LabelEncoder()
        try:
            y_enc = le.fit_transform(y_fit)
        except Exception:
            return 0.0

        model = _MODELS.get(self._eval_model_name, _MODELS["random_forest"])
        try:
            if self._cv > 1 and len(y_enc) >= self._cv * 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cv_obj = StratifiedKFold(n_splits=self._cv, shuffle=True, random_state=42)
                    scores = cross_val_score(model, X_fit, y_enc, cv=cv_obj, scoring=self._eval_metric)
                return float(np.clip(np.mean(scores), 0.0, 1.0))
            else:
                model.fit(X_fit, y_enc)
                preds = model.predict(X_fit)
                if self._eval_metric == "accuracy":
                    return float(accuracy_score(y_enc, preds))
                elif self._eval_metric == "f1":
                    avg = "binary" if len(le.classes_) == 2 else "macro"
                    return float(f1_score(y_enc, preds, average=avg, zero_division=0))
                else:
                    return float(accuracy_score(y_enc, preds))
        except Exception as exc:
            logger.debug("Accuracy evaluation failed: %s", exc)
            return 0.0

    def _tabpfn_accuracy_score(self, X: Features, y: OptionalTarget) -> float:
        """Evaluate cleaned data using TabPFN v2 with a single train/test split.

        TabPFN v2 applies its own internal preprocessing (z-normalisation,
        power scaling, missing-value masking), so numeric features — including
        NaN — are passed directly without prior imputation.

        Falls back to "random_forest" if the tabpfn package is not installed.
        """
        if not TABPFN_AVAILABLE:
            logger.warning(
                "tabpfn is not installed (pip install tabpfn>=2.0). "
                "Falling back to random_forest for this reward call."
            )
            self._eval_model_name = "random_forest"
            return self._accuracy_score(X, y)

        numeric = X.select_dtypes(include="number")
        if numeric.shape[1] == 0:
            return 0.0

        # Align y to the current (possibly row-reduced) index
        if isinstance(y, pd.Series):
            try:
                y_aligned = y.loc[numeric.index]
            except KeyError:
                y_aligned = y.iloc[:len(numeric)]
        else:
            y_arr_full = np.asarray(y)
            if len(y_arr_full) > len(numeric):
                y_arr_full = y_arr_full[:len(numeric)]
            y_aligned = y_arr_full

        y_arr = np.asarray(y_aligned)
        valid = (
            ~np.isnan(y_arr.astype(float))
            if y_arr.dtype != object
            else np.ones(len(y_arr), bool)
        )
        if valid.sum() < 20:
            return 0.0

        # Pass NaN values through — TabPFN v2 handles them internally
        X_fit = numeric.values[valid].astype(float)
        y_fit = y_arr[valid]

        le = LabelEncoder()
        try:
            y_enc = le.fit_transform(y_fit)
        except Exception:
            return 0.0

        if len(np.unique(y_enc)) < 2:
            return 0.0

        # Subsample for reward-loop speed (TabPFN is slower than RF per call)
        max_rows: int = getattr(self, "_tabpfn_max_rows", _TABPFN_MAX_ROWS)
        if len(X_fit) > max_rows:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_fit), size=max_rows, replace=False)
            X_fit = X_fit[idx]
            y_enc = y_enc[idx]
            if len(np.unique(y_enc)) < 2:
                return 0.0

        # Fixed train/test split — at least 10 test samples
        test_size = float(np.clip(10.0 / len(X_fit), 0.2, 0.4))
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_fit, y_enc,
                test_size=test_size,
                random_state=42,
                stratify=y_enc,
            )
        except ValueError:
            return 0.0

        try:
            from tabpfn import TabPFNClassifier  # local import — optional dep

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)

            if self._eval_metric == "f1":
                avg = "binary" if len(le.classes_) == 2 else "macro"
                return float(np.clip(f1_score(y_test, preds, average=avg, zero_division=0), 0.0, 1.0))
            return float(np.clip(accuracy_score(y_test, preds), 0.0, 1.0))

        except Exception as exc:
            logger.debug("TabPFN evaluation failed: %s", exc)
            return 0.0

    def _quality_score(self, X: Features) -> float:
        completeness = 1.0 - float(X.isna().sum().sum()) / max(X.size, 1)
        dup_penalty = float(X.duplicated().sum()) / max(len(X), 1)
        return float(np.clip(completeness * (1.0 - dup_penalty), 0.0, 1.0))

    def _drift_score(self, X: Features) -> float:
        if self._original_X is None:
            return 0.0
        numeric = X.select_dtypes(include="number")
        ref_numeric = self._original_X.select_dtypes(include="number")
        shared = [c for c in numeric.columns if c in ref_numeric.columns]
        if not shared:
            return 0.0
        dists: list[float] = []
        for col in shared:
            curr = numeric[col].dropna().values
            ref = ref_numeric[col].dropna().values
            if len(curr) < 2 or len(ref) < 2:
                continue
            try:
                w = float(stats.wasserstein_distance(curr, ref))
                ref_std = float(np.std(ref)) or 1.0
                dists.append(min(w / ref_std, 5.0))
            except Exception:
                continue
        return float(np.mean(dists)) if dists else 0.0


# ---------------------------------------------------------------------------
# Additional reward functions
# ---------------------------------------------------------------------------

class AccuracyReward(MultiObjectiveReward):
    """Pure ML accuracy reward — retention and quality are ignored."""

    def __init__(self, eval_model: str = "random_forest", eval_metric: str = "accuracy") -> None:
        super().__init__(
            weight_accuracy=1.0,
            weight_retention=0.0,
            weight_quality=0.0,
            drift_penalty_coeff=0.0,
            eval_model=eval_model,
            eval_metric=eval_metric,
        )


class DriftPenaltyReward(MultiObjectiveReward):
    """Accuracy with a strong Wasserstein drift penalty."""

    def __init__(
        self,
        drift_coeff: float = 0.5,
        eval_model: str = "random_forest",
        eval_metric: str = "accuracy",
    ) -> None:
        super().__init__(
            weight_accuracy=0.7,
            weight_retention=0.2,
            weight_quality=0.1,
            drift_penalty_coeff=drift_coeff,
            eval_model=eval_model,
            eval_metric=eval_metric,
        )


class IncrementalGainReward(BaseReward):
    """
    Rewards the *delta* in data quality since the last step rather than
    the absolute score.  This encourages the agent to keep improving
    instead of coasting on a single good action.
    """

    def __init__(self, base_reward: Optional[BaseReward] = None) -> None:
        self._base = base_reward or MultiObjectiveReward()
        self._prev_score: float = 0.0

    def reset(self, X_initial: Features, y_initial: OptionalTarget) -> None:
        self._base.reset(X_initial, y_initial)
        self._prev_score = self._base(X_initial, y_initial)

    def __call__(self, X: Features, y: OptionalTarget) -> float:
        current = self._base(X, y)
        delta = current - self._prev_score
        self._prev_score = current
        return float(np.clip(delta * 5.0, -1.0, 1.0))   # scale delta to approx [-1,1]


class TFMAwareReward(MultiObjectiveReward):
    """
    TFM-aware reward — optimises data cleaning for Tabular Foundation Models.

    Two key differences from ``MultiObjectiveReward``:

    1. **TabPFN v2 as evaluator** — the accuracy sub-score is computed by
       fitting TabPFN v2 on the cleaned data (train/test split, no CV).
       TabPFN applies its own internal preprocessing (z-normalisation,
       power scaling, missing-value masking), so upstream cleaning should
       improve the *input distribution* fed to these internal transforms —
       not replicate them.  This is what makes TFM-optimised pipelines
       differ from RF-optimised ones.

    2. **Non-linear context-size penalty** — in-context learners (TabPFN v2,
       TabICL) degrade non-linearly when training-context rows are removed:
       fewer rows means a smaller in-context training set, directly reducing
       prediction quality.  The retention sub-score is raised to
       ``context_penalty_power`` (default 2.0) so that, e.g., losing 20 % of
       rows (retention=0.80) yields a context-size score of 0.80²=0.64 instead
       of 0.80, penalising row deletion 25 % more harshly than linear retention.
       At 50 % row loss the penalty is 2.0× harsher than linear.

    Parameters
    ----------
    weight_accuracy : float
        Default 0.50.
    weight_retention : float
        Default 0.35 — slightly higher than the standard 0.30 to protect
        the in-context training set size.
    weight_quality : float
        Default 0.15.
    drift_penalty_coeff : float
        Default 0.05 — lower than the standard 0.10 because TabPFN's
        internal normalisation already compensates for moderate drift.
    eval_model : str
        Default "tabpfn".  Override to "random_forest" for ablation studies
        (produces the RF-reward baseline under identical settings).
    eval_metric : str
        "accuracy" or "f1".
    tabpfn_max_rows : int
        Maximum rows forwarded to TabPFN per reward call (stratified
        subsample for speed).  Default 512.
    context_penalty_power : float
        Exponent applied to the raw retention ratio.  Must be ≥ 1.0.
        Default 2.0 (quadratic).  Set to 1.0 to match MultiObjectiveReward.
    """

    def __init__(
        self,
        weight_accuracy: float = 0.50,
        weight_retention: float = 0.35,
        weight_quality: float = 0.15,
        drift_penalty_coeff: float = 0.05,
        eval_model: str = "tabpfn",
        eval_metric: str = "accuracy",
        tabpfn_max_rows: int = 512,
        context_penalty_power: float = 2.0,
    ) -> None:
        if context_penalty_power < 1.0:
            raise ValueError("context_penalty_power must be >= 1.0")
        super().__init__(
            weight_accuracy=weight_accuracy,
            weight_retention=weight_retention,
            weight_quality=weight_quality,
            drift_penalty_coeff=drift_penalty_coeff,
            eval_model=eval_model,
            eval_metric=eval_metric,
            eval_cv_folds=1,  # TabPFN always uses a single train/test split
        )
        self._tabpfn_max_rows = tabpfn_max_rows
        self._context_penalty_power = context_penalty_power

        if eval_model == "tabpfn" and not TABPFN_AVAILABLE:
            logger.warning(
                "TFMAwareReward: tabpfn package not found. "
                "Install with:  pip install tabpfn>=2.0  "
                "Falling back to random_forest until installed."
            )

    def __call__(self, X: Features, y: OptionalTarget) -> float:
        if X is None or len(X) == 0:
            self._last_components = RewardComponents(total=-1.0)
            return -1.0

        acc = self._accuracy_score(X, y)

        # Non-linear retention: in-context learners degrade faster with row loss
        raw_retention = len(X) / self._original_n_rows
        ret = float(raw_retention ** self._context_penalty_power)

        qual = self._quality_score(X)
        drift = self._drift_score(X)

        total = (
            self._w[0] * acc
            + self._w[1] * ret
            + self._w[2] * qual
            - self._drift_coeff * drift
        )
        total = float(np.clip(total, -1.0, 1.0))

        self._last_components = RewardComponents(
            accuracy=acc,
            retention=raw_retention,   # raw ratio stored for transparency
            quality=qual,
            drift_penalty=drift,
            total=total,
        )
        return total
