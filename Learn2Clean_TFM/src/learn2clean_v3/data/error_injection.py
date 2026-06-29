"""Synthetic error injection for controlled benchmark experiments.

Implements the four error types defined in contributions.md:
  - MCAR  : Missing Completely At Random
  - MAR   : Missing At Random (conditioned on another column)
  - OUT   : Outliers (replace cell with N(μ, k·σ))
  - DUP   : Duplicate rows

Design goals
------------
* All functions are pure: they return new DataFrames / Series; originals unchanged.
* Reproducible: every function accepts a ``seed`` parameter.
* ``apply_error_profile()`` is the single entry-point for experiment runners.
* ``generate_all_profiles()`` produces the full factorial grid used in paper experiments.

Usage::

    from learn2clean_v3.data.error_injection import apply_error_profile, ErrorProfile

    profile = ErrorProfile(error_type="mcar", rate=0.15)
    X_dirty, y_dirty = apply_error_profile(X, y, profile)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ErrorProfile
# ---------------------------------------------------------------------------

@dataclass
class ErrorProfile:
    """Specification for a single synthetic error injection run.

    Parameters
    ----------
    error_type:
        One of ``"mcar"``, ``"mar"``, ``"outlier"``, ``"duplicate"``, ``"none"``.
    rate:
        Fraction of rows / cells affected (interpretation depends on type).
    k:
        Multiplier for outlier magnitude: replacement = N(μ, k·σ). Default 3.0.
    seed:
        Random seed for reproducibility. Default 42.
    """

    error_type: str          # "mcar" | "mar" | "outlier" | "duplicate" | "none"
    rate: float              # fraction of cells/rows affected
    k: float = 3.0           # outlier severity
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
        """Short string tag for filenames, e.g. 'mcar_p015' or 'out_k3_p010'."""
        rate_str = f"p{int(self.rate * 100):03d}"
        if self.error_type == "outlier":
            return f"out_k{int(self.k)}_{rate_str}"
        return f"{self.error_type}_{rate_str}"


# ---------------------------------------------------------------------------
# MCAR — Missing Completely At Random
# ---------------------------------------------------------------------------

def inject_missing_mcar(
    X: pd.DataFrame,
    rate: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Replace a random fraction *rate* of cells in numeric columns with NaN.

    Parameters
    ----------
    X:
        Feature matrix. Only float/int columns are affected; object/category columns
        are left unchanged (they may already encode missing values as a category).
    rate:
        Fraction of cells to corrupt in eligible columns.
    seed:
        Random seed.

    Returns
    -------
    pd.DataFrame with the same shape as X.
    """
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


# ---------------------------------------------------------------------------
# MAR — Missing At Random (conditioned on anchor column)
# ---------------------------------------------------------------------------

def inject_missing_mar(
    X: pd.DataFrame,
    rate: float,
    seed: int = 42,
    anchor_col: Optional[str] = None,
) -> pd.DataFrame:
    """Introduce MAR missingness: NaN in target column when anchor column > Q75.

    Mimics a realistic mechanism where one sensor/feature fails when another is high.

    Parameters
    ----------
    X:
        Feature matrix.
    rate:
        Approximate fraction of rows that will gain a missing value in the
        highest-correlated column pair. Actual rate may differ slightly.
    seed:
        Random seed.
    anchor_col:
        Column used as the MAR trigger. If None, the pair with the highest
        absolute Pearson correlation is selected automatically.

    Returns
    -------
    pd.DataFrame with NaN values in the target column.
    """
    if rate == 0.0:
        return X.copy()

    rng = np.random.default_rng(seed)
    X_out = X.copy()
    num_cols = X_out.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) < 2:
        logger.warning("inject_missing_mar: need ≥2 numeric columns; falling back to MCAR.")
        return inject_missing_mcar(X_out, rate, seed)

    # Pick anchor column (highest-variance numeric column, or user-specified)
    if anchor_col is None:
        corr = X_out[num_cols].corr().abs()
        # Zero the diagonal on a writable copy to exclude self-correlations
        corr_arr = corr.to_numpy().copy()
        np.fill_diagonal(corr_arr, 0)
        corr = pd.DataFrame(corr_arr, index=corr.index, columns=corr.columns)
        anchor_col, target_col = corr.stack().idxmax()
    else:
        if anchor_col not in num_cols:
            raise ValueError(f"anchor_col {anchor_col!r} not in numeric columns.")
        # Target: highest corr with anchor
        corr = X_out[num_cols].corrwith(X_out[anchor_col]).abs()
        corr[anchor_col] = -1  # exclude self
        target_col = corr.idxmax()

    # Condition: anchor > Q75, then randomly set target to NaN (controlled by rate)
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


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------

def inject_outliers(
    X: pd.DataFrame,
    rate: float,
    k: float = 3.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Replace a fraction *rate* of cells in numeric columns with outlier values.

    Each replacement is drawn from N(μ_col ± k·σ_col), where the sign is random.

    Parameters
    ----------
    X:
        Feature matrix.
    rate:
        Fraction of cells in numeric columns to replace.
    k:
        Outlier severity. Replacement = μ ± k·σ + ε where ε ~ N(0, σ/10).
    seed:
        Random seed.

    Returns
    -------
    pd.DataFrame with the same shape as X; non-numeric columns untouched.
    """
    if rate == 0.0:
        return X.copy()

    rng = np.random.default_rng(seed)
    X_out = X.copy()
    num_cols = X_out.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        logger.warning("inject_outliers: no numeric columns found; returning unchanged.")
        return X_out

    # Upcast all numeric columns to float64 so injected float outlier values can be stored.
    # float32 columns also need upcasting — pandas rejects writing a float64 scalar into them.
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
        # Outlier value: draw from N(μ ± k·σ, σ/10) for natural spread
        noise = rng.normal(0, sigma / 10)
        X_out.iloc[r, X_out.columns.get_loc(col)] = mu + sign * k * sigma + noise

    logger.debug("Outliers injected: rate=%.2f k=%.1f cells=%d", rate, k, n_corrupt)
    return X_out


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------

def inject_duplicates(
    X: pd.DataFrame,
    y: pd.Series,
    rate: float,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Append exact duplicate rows to the dataset.

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Target series.
    rate:
        Fraction of *original* rows to duplicate (appended to end).
    seed:
        Random seed.

    Returns
    -------
    Tuple (X_with_dups, y_with_dups) — larger DataFrames; indices reset.
    """
    if rate == 0.0:
        return X.copy(), y.copy()

    rng = np.random.default_rng(seed)
    n_dup = max(1, int(round(len(X) * rate)))
    chosen_idx = rng.choice(len(X), size=n_dup, replace=True)

    X_dup = pd.concat([X, X.iloc[chosen_idx]], ignore_index=True)
    y_dup = pd.concat([y, y.iloc[chosen_idx]], ignore_index=True)

    logger.debug("Duplicates injected: %d rows added (rate=%.2f)", n_dup, rate)
    return X_dup, y_dup


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def apply_error_profile(
    X: pd.DataFrame,
    y: pd.Series,
    profile: ErrorProfile,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply a single ErrorProfile to (X, y) and return the dirty copies.

    This is the primary entry-point for experiment runners.

    Parameters
    ----------
    X:
        Clean feature matrix.
    y:
        Clean target series.
    profile:
        Which error type and rate to inject.

    Returns
    -------
    (X_dirty, y_dirty) — new DataFrames; X and y are not modified.
    """
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


# ---------------------------------------------------------------------------
# Grid helpers for experiments
# ---------------------------------------------------------------------------

def generate_all_profiles(
    include_none: bool = True,
) -> List[ErrorProfile]:
    """Return the full factorial set of ErrorProfiles used in paper experiments.

    Grid (from contributions.md):
      - MCAR:      rate ∈ {0.05, 0.10, 0.15, 0.20, 0.30}
      - MAR:       rate = 0.15
      - Outliers:  k ∈ {3, 5} × rate ∈ {0.05, 0.10}
      - Duplicates: rate ∈ {0.05, 0.10, 0.20}
      - (none):    rate = 0.0  — baseline clean run (optional)
    """
    profiles: List[ErrorProfile] = []

    if include_none:
        profiles.append(ErrorProfile("none", 0.0))

    # MCAR sweep (primary experiment axis)
    for rate in [0.05, 0.10, 0.15, 0.20, 0.30]:
        profiles.append(ErrorProfile("mcar", rate))

    # MAR (fixed rate)
    profiles.append(ErrorProfile("mar", 0.15))

    # Outliers
    for k in [3.0, 5.0]:
        for rate in [0.05, 0.10]:
            profiles.append(ErrorProfile("outlier", rate, k=k))

    # Duplicates
    for rate in [0.05, 0.10, 0.20]:
        profiles.append(ErrorProfile("duplicate", rate))

    return profiles
