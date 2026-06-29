"""
learn2clean_v3.profiling.data_profiler
======================================
Lightweight data-quality profiler that runs before any cleaning action.

Produces a :class:`DataQualityReport` with per-column and aggregate metrics.
The report is used in two ways:

1. **RL observation space** — numeric quality signals (missing_rate, outlier_rate,
   duplicate_rate, …) are concatenated into the state vector fed to the agent.
2. **Pipeline pruning** — the greedy oracle can skip action groups that are
   irrelevant for the current profile (e.g., dedup when duplicate_rate == 0).

Usage
-----
>>> from learn2clean_v3.profiling import DataProfiler
>>> profiler = DataProfiler()
>>> report = profiler.profile(X_dirty, y)
>>> print(report.summary())
>>> state_vec = report.to_state_vector()   # for RL observation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------

@dataclass
class ColumnProfile:
    name: str
    dtype: str                    # "numeric" | "categorical" | "datetime" | "mixed"
    missing_rate: float           # fraction of NaN
    outlier_rate_iqr: float       # fraction outside [Q1-1.5*IQR, Q3+1.5*IQR]
    outlier_rate_zscore: float    # fraction with |z| > 3
    skewness: float               # abs skewness of numeric column
    n_unique: int                 # number of distinct values
    cardinality_ratio: float      # n_unique / n_rows


@dataclass
class DataQualityReport:
    n_rows: int
    n_cols: int
    n_numeric: int
    n_categorical: int
    missing_rate: float           # overall fraction of missing cells
    duplicate_rate: float         # fraction of rows that are exact duplicates
    duplicate_rate_numeric: float # fraction of rows that are numeric-column duplicates
    outlier_rate_iqr: float       # mean IQR outlier rate across numeric cols
    outlier_rate_zscore: float    # mean z-score outlier rate across numeric cols
    skewness_mean: float          # mean abs skewness across numeric cols
    class_imbalance: float        # 1 - max_class_freq for target (0 = perfectly balanced)
    columns: List[ColumnProfile] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        lines = [
            f"Rows: {self.n_rows}  Cols: {self.n_cols}  "
            f"(numeric={self.n_numeric}, categorical={self.n_categorical})",
            f"  Missing:    {self.missing_rate:.1%}",
            f"  Duplicates: {self.duplicate_rate:.1%} "
            f"(numeric-only: {self.duplicate_rate_numeric:.1%})",
            f"  Outliers:   IQR={self.outlier_rate_iqr:.1%}  "
            f"z-score={self.outlier_rate_zscore:.1%}",
            f"  Skewness:   {self.skewness_mean:.2f}  (mean |skew| across numeric cols)",
            f"  Class imbalance: {self.class_imbalance:.3f}",
        ]
        return "\n".join(lines)

    def to_state_vector(self) -> np.ndarray:
        """Return a fixed-length float vector for use as RL observation features."""
        return np.array([
            self.missing_rate,
            self.duplicate_rate,
            self.duplicate_rate_numeric,
            self.outlier_rate_iqr,
            self.outlier_rate_zscore,
            self.skewness_mean,
            self.class_imbalance,
            self.n_numeric / max(self.n_cols, 1),
            self.n_categorical / max(self.n_cols, 1),
        ], dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Which action groups are relevant for this data profile
    # ------------------------------------------------------------------ #

    def relevant_action_groups(
        self,
        missing_threshold: float = 0.005,
        outlier_threshold: float = 0.005,
        duplicate_threshold: float = 0.005,
    ) -> Set[str]:
        """
        Return the set of action-group names that address detected issues.

        Can be used by the greedy oracle or the RL env's action mask to prune
        irrelevant pipeline steps (e.g., skip imputation when data is complete).
        """
        groups: Set[str] = {"scale"}   # normalisation is always potentially useful
        if self.missing_rate > missing_threshold:
            groups.add("impute")
        if self.outlier_rate_iqr > outlier_threshold:
            groups.add("outlier")
        if self.duplicate_rate > duplicate_threshold:
            groups.add("dedup")
        return groups


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class DataProfiler:
    """
    Compute a :class:`DataQualityReport` for a feature matrix.

    Parameters
    ----------
    iqr_multiplier : float
        IQR multiplier used to flag outliers in per-column analysis.
    zscore_threshold : float
        |z| threshold for z-score outlier flagging.
    """

    def __init__(
        self,
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
    ) -> None:
        self._iqr_k = iqr_multiplier
        self._z_thresh = zscore_threshold

    # ------------------------------------------------------------------ #

    def profile(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> DataQualityReport:
        """
        Profile *X* (and optionally *y* for class-imbalance metrics).

        Returns
        -------
        DataQualityReport
        """
        n_rows, n_cols = X.shape
        numeric_cols = X.select_dtypes(include="number").columns.tolist()
        cat_cols = X.select_dtypes(exclude="number").columns.tolist()

        # ── Global missing ─────────────────────────────────────────────
        missing_rate = float(X.isna().mean().mean())

        # ── Duplicates ─────────────────────────────────────────────────
        dup_mask_all = X.duplicated(keep=False)
        duplicate_rate = float(dup_mask_all.mean())

        if numeric_cols:
            dup_mask_num = X[numeric_cols].duplicated(keep=False)
            duplicate_rate_numeric = float(dup_mask_num.mean())
        else:
            duplicate_rate_numeric = 0.0

        # ── Per-column profiles ────────────────────────────────────────
        col_profiles: List[ColumnProfile] = []
        iqr_rates: List[float] = []
        z_rates: List[float] = []
        skews: List[float] = []

        for col in X.columns:
            series = X[col]
            miss = float(series.isna().mean())
            n_unique = int(series.nunique(dropna=True))
            card_ratio = n_unique / max(n_rows, 1)

            if col in numeric_cols:
                dtype_label = "numeric"
                s_clean = series.dropna()
                # IQR outlier rate
                if len(s_clean) > 3:
                    q1, q3 = s_clean.quantile(0.25), s_clean.quantile(0.75)
                    iqr = q3 - q1
                    lo, hi = q1 - self._iqr_k * iqr, q3 + self._iqr_k * iqr
                    iqr_rate = float(((s_clean < lo) | (s_clean > hi)).mean())
                else:
                    iqr_rate = 0.0
                # z-score outlier rate
                if len(s_clean) > 3 and s_clean.std() > 0:
                    z = (s_clean - s_clean.mean()) / s_clean.std()
                    z_rate = float((z.abs() > self._z_thresh).mean())
                else:
                    z_rate = 0.0
                # skewness
                try:
                    skew = abs(float(s_clean.skew())) if len(s_clean) > 3 else 0.0
                except Exception:
                    skew = 0.0

                iqr_rates.append(iqr_rate)
                z_rates.append(z_rate)
                skews.append(skew)
            else:
                dtype_label = "categorical"
                iqr_rate = 0.0
                z_rate = 0.0
                skew = 0.0

            col_profiles.append(ColumnProfile(
                name=col,
                dtype=dtype_label,
                missing_rate=miss,
                outlier_rate_iqr=iqr_rate,
                outlier_rate_zscore=z_rate,
                skewness=skew,
                n_unique=n_unique,
                cardinality_ratio=card_ratio,
            ))

        outlier_rate_iqr = float(np.mean(iqr_rates)) if iqr_rates else 0.0
        outlier_rate_z   = float(np.mean(z_rates))   if z_rates   else 0.0
        skewness_mean    = float(np.mean(skews))      if skews     else 0.0

        # ── Class imbalance ────────────────────────────────────────────
        if y is not None and len(y) > 0:
            counts = pd.Series(y).value_counts(normalize=True)
            class_imbalance = float(1.0 - counts.max())
        else:
            class_imbalance = 0.0

        return DataQualityReport(
            n_rows=n_rows,
            n_cols=n_cols,
            n_numeric=len(numeric_cols),
            n_categorical=len(cat_cols),
            missing_rate=missing_rate,
            duplicate_rate=duplicate_rate,
            duplicate_rate_numeric=duplicate_rate_numeric,
            outlier_rate_iqr=outlier_rate_iqr,
            outlier_rate_zscore=outlier_rate_z,
            skewness_mean=skewness_mean,
            class_imbalance=class_imbalance,
            columns=col_profiles,
        )
