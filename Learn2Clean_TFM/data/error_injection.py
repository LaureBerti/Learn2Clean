
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ErrorProfile:

    error_type: str
    rate: float
    k: float = 3.0
    seed: int = 42

    def __post_init__(self) -> None:
        valid_types = {"mcar", "mar", "outlier", "duplicate", "none"}
        if self.error_type not in valid_types:
            raise ValueError(f"error_type must be one of {valid_types}, got {self.error_type!r}")
        if not (0.0 <= self.rate <= 1.0):
            raise ValueError(f"rate must be in [0, 1], got {self.rate}")
        if self.k <= 0:
            raise ValueError(f"k must be > 0, got {self.k}")

    @property
    def tag(self) -> str:
        rate_str = f"p{int(self.rate * 100):03d}"
        if self.error_type == "outlier":
            return f"out_k{int(self.k)}_{rate_str}"
        return f"{self.error_type}_{rate_str}"



def inject_missing_mcar(
    X: pd.DataFrame,
    rate: float,
    seed: int = 42,
) -> pd.DataFrame:
    if rate == 0.0:
        return X.copy()

    rng = np.random.default_rng(seed)
    X_out = X.copy()
    num_cols = X_out.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        logger.warning("inject_missing_mcar: no numeric columns found; returning unchanged.")
        return X_out

    n_cells = len(X_out) * len(num_cols)
    n_corrupt = max(1, int(round(n_cells * rate)))

    row_idx = rng.integers(0, len(X_out), size=n_corrupt)
    col_idx = rng.integers(0, len(num_cols), size=n_corrupt)

    for r, c in zip(row_idx, col_idx):
        X_out.iloc[r, X_out.columns.get_loc(num_cols[c])] = np.nan

    actual_rate = X_out[num_cols].isna().mean().mean()
    logger.debug("MCAR injected: target=%.2f actual=%.2f", rate, actual_rate)
    return X_out



def inject_missing_mar(
    X: pd.DataFrame,
    rate: float,
    seed: int = 42,
    anchor_col: Optional[str] = None,
) -> pd.DataFrame:
    if rate == 0.0:
        return X.copy()

    rng = np.random.default_rng(seed)
    X_out = X.copy()
    num_cols = X_out.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) < 2:
        logger.warning("inject_missing_mar: need ≥2 numeric columns; falling back to MCAR.")
        return inject_missing_mcar(X_out, rate, seed)

    if anchor_col is None:
        corr = X_out[num_cols].corr().abs()
        corr_arr = corr.to_numpy().copy()
        np.fill_diagonal(corr_arr, 0)
        corr = pd.DataFrame(corr_arr, index=corr.index, columns=corr.columns)
        anchor_col, target_col = corr.stack().idxmax()
    else:
        if anchor_col not in num_cols:
            raise ValueError(f"anchor_col {anchor_col!r} not in numeric columns.")
        corr = X_out[num_cols].corrwith(X_out[anchor_col]).abs()
        corr[anchor_col] = -1
        target_col = corr.idxmax()

    q75 = X_out[anchor_col].quantile(0.75)
    eligible_mask = X_out[anchor_col] > q75
    eligible_idx = X_out.index[eligible_mask].tolist()

    n_corrupt = max(1, int(round(len(X_out) * rate)))
    n_corrupt = min(n_corrupt, len(eligible_idx))

    chosen = rng.choice(eligible_idx, size=n_corrupt, replace=False)
    X_out.loc[chosen, target_col] = np.nan

    logger.debug(
        "MAR injected: anchor=%s → target=%s, rows corrupted=%d",
        anchor_col, target_col, n_corrupt,
    )
    return X_out



def inject_outliers(
    X: pd.DataFrame,
    rate: float,
    k: float = 3.0,
    seed: int = 42,
) -> pd.DataFrame:
    if rate == 0.0:
        return X.copy()

    rng = np.random.default_rng(seed)
    X_out = X.copy()
    num_cols = X_out.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        logger.warning("inject_outliers: no numeric columns found; returning unchanged.")
        return X_out

    for col in num_cols:
        if X_out[col].dtype != np.float64:
            X_out[col] = X_out[col].astype(np.float64)

    col_stats = {
        col: (X_out[col].mean(skipna=True), X_out[col].std(skipna=True))
        for col in num_cols
    }

    n_cells = len(X_out) * len(num_cols)
    n_corrupt = max(1, int(round(n_cells * rate)))

    row_idx = rng.integers(0, len(X_out), size=n_corrupt)
    col_idx = rng.integers(0, len(num_cols), size=n_corrupt)
    signs = rng.choice([-1.0, 1.0], size=n_corrupt)

    for r, c, sign in zip(row_idx, col_idx, signs):
        col = num_cols[c]
        mu, sigma = col_stats[col]
        if sigma == 0 or np.isnan(sigma):
            continue
        noise = rng.normal(0, sigma / 10)
        X_out.iloc[r, X_out.columns.get_loc(col)] = mu + sign * k * sigma + noise

    logger.debug("Outliers injected: rate=%.2f k=%.1f cells=%d", rate, k, n_corrupt)
    return X_out



def inject_duplicates(
    X: pd.DataFrame,
    y: pd.Series,
    rate: float,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    if rate == 0.0:
        return X.copy(), y.copy()

    rng = np.random.default_rng(seed)
    n_dup = max(1, int(round(len(X) * rate)))
    chosen_idx = rng.choice(len(X), size=n_dup, replace=True)

    X_dup = pd.concat([X, X.iloc[chosen_idx]], ignore_index=True)
    y_dup = pd.concat([y, y.iloc[chosen_idx]], ignore_index=True)

    logger.debug("Duplicates injected: %d rows added (rate=%.2f)", n_dup, rate)
    return X_dup, y_dup



def apply_error_profile(
    X: pd.DataFrame,
    y: pd.Series,
    profile: ErrorProfile,
) -> Tuple[pd.DataFrame, pd.Series]:
    if profile.error_type == "none" or profile.rate == 0.0:
        return X.copy(), y.copy()

    if profile.error_type == "mcar":
        return inject_missing_mcar(X, profile.rate, profile.seed), y.copy()

    if profile.error_type == "mar":
        return inject_missing_mar(X, profile.rate, profile.seed), y.copy()

    if profile.error_type == "outlier":
        return inject_outliers(X, profile.rate, profile.k, profile.seed), y.copy()

    if profile.error_type == "duplicate":
        return inject_duplicates(X, y, profile.rate, profile.seed)

    raise ValueError(f"Unknown error_type: {profile.error_type!r}")



def generate_all_profiles(
    include_none: bool = True,
) -> List[ErrorProfile]:
    profiles: List[ErrorProfile] = []

    if include_none:
        profiles.append(ErrorProfile("none", 0.0))

    for rate in [0.05, 0.10, 0.15, 0.20, 0.30]:
        profiles.append(ErrorProfile("mcar", rate))

    profiles.append(ErrorProfile("mar", 0.15))

    for k in [3.0, 5.0]:
        for rate in [0.05, 0.10]:
            profiles.append(ErrorProfile("outlier", rate, k=k))

    for rate in [0.05, 0.10, 0.20]:
        profiles.append(ErrorProfile("duplicate", rate))

    return profiles
