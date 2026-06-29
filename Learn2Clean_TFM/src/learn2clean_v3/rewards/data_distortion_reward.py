"""
DataDistortionPenaltyReward — new reward for Learn2Clean V3.

Measures how much the cleaning pipeline *distorts* the original data
distribution and uses it as the primary reward signal.  A cleaning
operation that preserves the data's statistical structure while fixing
quality issues (missing values, duplicates, outliers) scores near 1.
One that changes the underlying distribution unnecessarily scores near 0.

Why this matters
----------------
Classic rewards (accuracy, completeness) do not distinguish between:
  - Mean imputation  → collapses the missing column's distribution to a spike
  - KNN imputation   → preserves local density structure

Both may achieve similar ML accuracy on small datasets, but mean imputation
produces distorted data that misleads downstream users.  This reward
penalises distribution-changing operations so the agent learns to prefer
*faithful* cleaning.

Distortion score (5 components, each in [0, 1])
------------------------------------------------
1. Wasserstein distance   per-column 1-D Wasserstein, normalised by σ_ref
2. Jensen-Shannon (JS)    per-column JS divergence on binned histograms
3. Correlation shift      Frobenius norm of (Σ_clean − Σ_orig) normalised by √n
4. Variance ratio         mean |log(σ²_clean / σ²_orig)| per column, clipped
5. Skewness shift         mean |skew_clean − skew_orig| / (1 + |skew_orig|) per col

Total distortion = weighted average of the 5 components (weights configurable).
Reward           = (1 − distortion) + weight_accuracy × accuracy_score

All sub-scores are clipped to [0, 1] before averaging.

Parameters
----------
weight_wasserstein : float   Weight for Wasserstein component (default 0.30)
weight_js          : float   Weight for JS-divergence component (default 0.25)
weight_correlation : float   Weight for correlation-shift component (default 0.20)
weight_variance    : float   Weight for variance-ratio component (default 0.15)
weight_skewness    : float   Weight for skewness-shift component (default 0.10)
weight_accuracy    : float   Bonus for ML model accuracy (default 0.0 — pure distortion)
eval_model         : str     Model used when weight_accuracy > 0
eval_cv_folds      : int     CV folds for accuracy evaluation
n_bins             : int     Histogram bins for JS divergence (default 50)
eps                : float   Numerical stability constant (default 1e-10)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from learn2clean_v3.rewards.base_reward import BaseReward
from learn2clean_v3.types import Features, OptionalTarget

logger = logging.getLogger(__name__)


@dataclass
class DistortionComponents:
    """Per-component breakdown returned after each reward call."""
    wasserstein: float = 0.0
    js_divergence: float = 0.0
    correlation_shift: float = 0.0
    variance_ratio: float = 0.0
    skewness_shift: float = 0.0
    accuracy_bonus: float = 0.0
    total_distortion: float = 0.0
    reward: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "wasserstein": round(self.wasserstein, 5),
            "js_divergence": round(self.js_divergence, 5),
            "correlation_shift": round(self.correlation_shift, 5),
            "variance_ratio": round(self.variance_ratio, 5),
            "skewness_shift": round(self.skewness_shift, 5),
            "accuracy_bonus": round(self.accuracy_bonus, 5),
            "total_distortion": round(self.total_distortion, 5),
            "reward": round(self.reward, 5),
        }


class DataDistortionPenaltyReward(BaseReward):
    """
    Reward function that penalises data distribution distortion.

    See module docstring for full description.
    """

    def __init__(
        self,
        weight_wasserstein: float = 0.30,
        weight_js: float = 0.25,
        weight_correlation: float = 0.20,
        weight_variance: float = 0.15,
        weight_skewness: float = 0.10,
        weight_accuracy: float = 0.0,
        eval_model: str = "random_forest",
        eval_cv_folds: int = 1,
        n_bins: int = 50,
        eps: float = 1e-10,
    ) -> None:
        # Normalise component weights
        raw = np.array([
            weight_wasserstein,
            weight_js,
            weight_correlation,
            weight_variance,
            weight_skewness,
        ], dtype=float)
        if raw.sum() <= 0:
            raise ValueError("At least one distortion weight must be positive.")
        self._w = raw / raw.sum()

        self._w_accuracy = float(weight_accuracy)
        self._eval_model = eval_model
        self._cv = eval_cv_folds
        self._n_bins = n_bins
        self._eps = eps

        self._ref_X: Optional[Features] = None
        self._ref_numeric: Optional[pd.DataFrame] = None
        self._ref_stats: Dict[str, Dict] = {}   # per-column cached stats
        self._ref_corr: Optional[np.ndarray] = None

        self._last_components: Optional[DistortionComponents] = None

    # ------------------------------------------------------------------
    # BaseReward interface
    # ------------------------------------------------------------------

    def reset(self, X_initial: Features, y_initial: OptionalTarget) -> None:
        self._ref_X = X_initial.copy()
        self._ref_numeric = X_initial.select_dtypes(include="number").copy()
        self._ref_stats = self._compute_column_stats(self._ref_numeric)
        self._ref_corr = self._compute_corr_matrix(self._ref_numeric)
        self._last_components = None

    def __call__(self, X: Features, y: OptionalTarget) -> float:
        if X is None or len(X) == 0 or self._ref_numeric is None:
            self._last_components = DistortionComponents(reward=-1.0)
            return -1.0

        numeric = X.select_dtypes(include="number")
        shared = [c for c in numeric.columns if c in self._ref_numeric.columns]

        if not shared:
            self._last_components = DistortionComponents(reward=0.0)
            return 0.0

        cur_numeric = numeric[shared]
        ref_numeric = self._ref_numeric[shared]

        # --- Component 1: Wasserstein ---
        w_score = self._wasserstein_score(cur_numeric, ref_numeric)

        # --- Component 2: Jensen-Shannon divergence ---
        js_score = self._js_score(cur_numeric, ref_numeric)

        # --- Component 3: Correlation structure shift ---
        corr_score = self._correlation_score(cur_numeric, ref_numeric)

        # --- Component 4: Variance ratio ---
        var_score = self._variance_score(cur_numeric, ref_numeric)

        # --- Component 5: Skewness shift ---
        skew_score = self._skewness_score(cur_numeric, ref_numeric)

        # Weighted distortion (higher = more distortion).
        # NaN components (e.g. constant columns) are treated as maximum distortion.
        components = np.nan_to_num(
            np.array([w_score, js_score, corr_score, var_score, skew_score]),
            nan=1.0,
        )
        total_distortion = float(np.dot(self._w, components))

        # Accuracy bonus
        acc_bonus = 0.0
        if self._w_accuracy > 0:
            acc_bonus = self._accuracy_score(X, y) * self._w_accuracy

        reward = float(np.clip(1.0 - total_distortion + acc_bonus, -1.0, 1.0))

        self._last_components = DistortionComponents(
            wasserstein=w_score,
            js_divergence=js_score,
            correlation_shift=corr_score,
            variance_ratio=var_score,
            skewness_shift=skew_score,
            accuracy_bonus=acc_bonus,
            total_distortion=total_distortion,
            reward=reward,
        )
        return reward

    @property
    def last_components(self) -> Optional[DistortionComponents]:
        return self._last_components

    # ------------------------------------------------------------------
    # Component computations
    # ------------------------------------------------------------------

    def _wasserstein_score(
        self,
        cur: pd.DataFrame,
        ref: pd.DataFrame,
    ) -> float:
        """
        Mean normalised 1-D Wasserstein distance across columns.

        Normalise by reference std so the score is scale-invariant.
        Cap at 3 σ-units then rescale to [0, 1].
        """
        distances: List[float] = []
        for col in cur.columns:
            c = cur[col].dropna().values
            r = ref[col].dropna().values
            if len(c) < 2 or len(r) < 2:
                continue
            try:
                w = float(stats.wasserstein_distance(c, r))
                ref_std = float(np.std(r)) or 1.0
                # Normalise: 0 = identical, 1 = 3σ apart (capped)
                distances.append(min(w / (3.0 * ref_std + self._eps), 1.0))
            except Exception:
                continue
        return float(np.mean(distances)) if distances else 0.0

    def _js_score(
        self,
        cur: pd.DataFrame,
        ref: pd.DataFrame,
    ) -> float:
        """
        Mean Jensen-Shannon divergence across columns (binned histograms).

        JS divergence ∈ [0, 1] (base-2 log).  1 = completely disjoint.
        """
        scores: List[float] = []
        for col in cur.columns:
            c = cur[col].dropna().values
            r = ref[col].dropna().values
            if len(c) < 2 or len(r) < 2:
                continue
            try:
                lo = min(r.min(), c.min())
                hi = max(r.max(), c.max())
                if lo >= hi:
                    continue
                bins = np.linspace(lo, hi, self._n_bins + 1)
                p, _ = np.histogram(r, bins=bins, density=True)
                q, _ = np.histogram(c, bins=bins, density=True)
                p = p + self._eps
                q = q + self._eps
                p /= p.sum()
                q /= q.sum()
                js = float(jensenshannon(p, q, base=2.0))
                scores.append(float(np.clip(js, 0.0, 1.0)))
            except Exception:
                continue
        return float(np.mean(scores)) if scores else 0.0

    def _correlation_score(
        self,
        cur: pd.DataFrame,
        ref: pd.DataFrame,
    ) -> float:
        """
        Frobenius norm of (Σ_clean − Σ_orig) normalised to [0, 1].

        Maximum possible Frobenius norm for a correlation matrix difference
        is 2√(n_cols) (all entries flip from +1 to −1 or vice-versa).
        """
        if cur.shape[1] < 2:
            return 0.0
        try:
            cur_corr = self._compute_corr_matrix(cur)
            if cur_corr is None or self._ref_corr is None:
                return 0.0
            n = cur_corr.shape[0]
            # Align shapes (columns may differ after dropping)
            if cur_corr.shape != self._ref_corr.shape:
                ref_corr_local = self._compute_corr_matrix(
                    ref[cur.columns]
                )
                if ref_corr_local is None:
                    return 0.0
            else:
                ref_corr_local = self._ref_corr
            diff = cur_corr - ref_corr_local
            frob = float(np.linalg.norm(diff, "fro"))
            max_frob = 2.0 * np.sqrt(n)
            return float(np.clip(frob / (max_frob + self._eps), 0.0, 1.0))
        except Exception as exc:
            logger.debug("correlation_score error: %s", exc)
            return 0.0

    def _variance_score(
        self,
        cur: pd.DataFrame,
        ref: pd.DataFrame,
    ) -> float:
        """
        Mean |log(σ²_clean / σ²_orig)| per column.

        A ratio of 1 (no change) → 0 distortion.
        A ratio of e² ≈ 7.4× → 2 log-units → clipped to 1.0.
        """
        ratios: List[float] = []
        for col in cur.columns:
            c = cur[col].dropna().values
            r = ref[col].dropna().values
            if len(c) < 2 or len(r) < 2:
                continue
            var_c = float(np.var(c))
            var_r = float(np.var(r))
            if var_r < self._eps:
                continue
            log_ratio = abs(np.log((var_c + self._eps) / (var_r + self._eps)))
            # 2 log-units = one order-of-magnitude change → cap at 1
            ratios.append(float(np.clip(log_ratio / 2.0, 0.0, 1.0)))
        return float(np.mean(ratios)) if ratios else 0.0

    def _skewness_score(
        self,
        cur: pd.DataFrame,
        ref: pd.DataFrame,
    ) -> float:
        """
        Mean normalised skewness change per column.

        |skew_clean − skew_orig| / (1 + |skew_orig|) → [0, ∞), capped at 1.
        """
        scores: List[float] = []
        for col in cur.columns:
            c = cur[col].dropna().values
            r = ref[col].dropna().values
            if len(c) < 3 or len(r) < 3:
                continue
            try:
                import warnings as _w
                with _w.catch_warnings():
                    _w.simplefilter("ignore", RuntimeWarning)
                    skew_c = float(stats.skew(c))
                    skew_r = float(stats.skew(r))
                if np.isnan(skew_c):
                    # Constant column after cleaning → maximum skewness distortion
                    scores.append(1.0)
                    continue
                if np.isnan(skew_r):
                    continue
                delta = abs(skew_c - skew_r) / (1.0 + abs(skew_r) + self._eps)
                scores.append(float(np.clip(delta, 0.0, 1.0)))
            except Exception:
                continue
        return float(np.mean(scores)) if scores else 0.0

    # ------------------------------------------------------------------
    # Optional accuracy bonus
    # ------------------------------------------------------------------

    def _accuracy_score(self, X: Features, y: OptionalTarget) -> float:
        """Quick random-forest accuracy for the accuracy bonus."""
        if y is None:
            return 0.0
        try:
            import warnings

            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import LabelEncoder

            numeric = X.select_dtypes(include="number").fillna(0)
            if numeric.shape[1] == 0 or len(numeric) < 10:
                return 0.0
            y_arr = np.asarray(y)
            le = LabelEncoder()
            y_enc = le.fit_transform(y_arr)
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            if self._cv > 1 and len(y_enc) >= self._cv * 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scores = cross_val_score(model, numeric.values, y_enc, cv=self._cv)
                return float(np.clip(np.mean(scores), 0.0, 1.0))
            model.fit(numeric.values, y_enc)
            return float(np.mean(model.predict(numeric.values) == y_enc))
        except Exception as exc:
            logger.debug("accuracy_score error: %s", exc)
            return 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_column_stats(df: pd.DataFrame) -> Dict[str, Dict]:
        stats_map: Dict[str, Dict] = {}
        for col in df.columns:
            s = df[col].dropna()
            if len(s) < 2:
                continue
            stats_map[col] = {
                "mean": float(s.mean()),
                "std": float(s.std()),
                "skew": float(s.skew()),
                "var": float(s.var()),
            }
        return stats_map

    @staticmethod
    def _compute_corr_matrix(df: pd.DataFrame) -> Optional[np.ndarray]:
        if df.shape[1] < 2:
            return None
        try:
            filled = df.fillna(df.median())
            corr = filled.corr().values
            if np.any(np.isnan(corr)):
                return None
            return corr
        except Exception:
            return None
