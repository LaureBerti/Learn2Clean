
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd



@dataclass
class ColumnProfile:
    name: str
    dtype: str
    missing_rate: float
    outlier_rate_iqr: float
    outlier_rate_zscore: float
    skewness: float
    n_unique: int
    cardinality_ratio: float


@dataclass
class DataQualityReport:
    n_rows: int
    n_cols: int
    n_numeric: int
    n_categorical: int
    missing_rate: float
    duplicate_rate: float
    duplicate_rate_numeric: float
    outlier_rate_iqr: float
    outlier_rate_zscore: float
    skewness_mean: float
    class_imbalance: float
    columns: List[ColumnProfile] = field(default_factory=list)

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


    def relevant_action_groups(
        self,
        missing_threshold: float = 0.005,
        outlier_threshold: float = 0.005,
        duplicate_threshold: float = 0.005,
    ) -> Set[str]:
        groups: Set[str] = {"scale"}
        if self.missing_rate > missing_threshold:
            groups.add("impute")
        if self.outlier_rate_iqr > outlier_threshold:
            groups.add("outlier")
        if self.duplicate_rate > duplicate_threshold:
            groups.add("dedup")
        return groups



class DataProfiler:

    def __init__(
        self,
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
    ) -> None:
        self._iqr_k = iqr_multiplier
        self._z_thresh = zscore_threshold


    def profile(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> DataQualityReport:
        n_rows, n_cols = X.shape
        numeric_cols = X.select_dtypes(include="number").columns.tolist()
        cat_cols = X.select_dtypes(exclude="number").columns.tolist()

        missing_rate = float(X.isna().mean().mean())

        dup_mask_all = X.duplicated(keep=False)
        duplicate_rate = float(dup_mask_all.mean())

        if numeric_cols:
            dup_mask_num = X[numeric_cols].duplicated(keep=False)
            duplicate_rate_numeric = float(dup_mask_num.mean())
        else:
            duplicate_rate_numeric = 0.0

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
                if len(s_clean) > 3:
                    q1, q3 = s_clean.quantile(0.25), s_clean.quantile(0.75)
                    iqr = q3 - q1
                    lo, hi = q1 - self._iqr_k * iqr, q3 + self._iqr_k * iqr
                    iqr_rate = float(((s_clean < lo) | (s_clean > hi)).mean())
                else:
                    iqr_rate = 0.0
                if len(s_clean) > 3 and s_clean.std() > 0:
                    z = (s_clean - s_clean.mean()) / s_clean.std()
                    z_rate = float((z.abs() > self._z_thresh).mean())
                else:
                    z_rate = 0.0
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
