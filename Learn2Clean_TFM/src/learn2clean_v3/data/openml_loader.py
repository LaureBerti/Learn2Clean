"""OpenML dataset loader for Learn2Clean V3 benchmark experiments.

Loads the 10 datasets (D1–D10) used in TabPFN v1/v2 and TabICL benchmark papers.
Caches each dataset as a Parquet file under outputs/datasets/ to avoid repeated
network calls. Falls back gracefully when openml is not installed.

Usage::

    from learn2clean_v3.data import load_dataset, load_all_datasets

    X, y, spec = load_dataset("diabetes")
    all_datasets = load_all_datasets()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentinel for optional openml dependency
# ---------------------------------------------------------------------------
try:
    import openml  # type: ignore

    OPENML_AVAILABLE = True
except ImportError:  # pragma: no cover
    OPENML_AVAILABLE = False
    logger.warning(
        "openml not installed — load_dataset() will only work from local cache. "
        "Install with: pip install openml"
    )

# ---------------------------------------------------------------------------
# DatasetSpec
# ---------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    """Specification for one benchmark dataset."""

    name: str
    openml_id: int
    eval_metric: str = "f1"
    max_rows: Optional[int] = None          # stratified subsample cap (None = use all)
    zero_as_nan_cols: List[str] = field(default_factory=list)  # columns where 0 → NaN
    seed: int = 42


# ---------------------------------------------------------------------------
# Benchmark dataset registry (D1–D10)
# ---------------------------------------------------------------------------

BENCHMARK_DATASETS: Dict[str, DatasetSpec] = {
    # --- XS tier (<400 rows) ---
    "hepatitis": DatasetSpec(
        name="hepatitis",
        openml_id=55,
        eval_metric="f1",
        max_rows=None,
    ),
    "heart_statlog": DatasetSpec(
        name="heart_statlog",
        openml_id=53,
        eval_metric="f1",
        max_rows=None,
    ),
    "ionosphere": DatasetSpec(
        name="ionosphere",
        openml_id=59,
        eval_metric="accuracy",
        max_rows=None,
    ),
    # --- S tier (<1K rows) ---
    "blood_transfusion": DatasetSpec(
        name="blood_transfusion",
        openml_id=1464,
        eval_metric="f1",
        max_rows=None,
    ),
    "diabetes": DatasetSpec(
        name="diabetes",
        openml_id=37,
        eval_metric="f1",
        max_rows=None,
        # Pima: physiologically impossible zeros represent missing data
        zero_as_nan_cols=["plas", "pres", "skin", "insu", "mass"],
    ),
    "credit_g": DatasetSpec(
        name="credit_g",
        openml_id=31,
        eval_metric="f1",
        max_rows=None,
    ),
    # --- M tier (1K–10K rows) ---
    "kr_vs_kp": DatasetSpec(
        name="kr_vs_kp",
        openml_id=3,
        eval_metric="accuracy",
        max_rows=None,
    ),
    "phoneme": DatasetSpec(
        name="phoneme",
        openml_id=1489,
        eval_metric="f1",
        max_rows=None,
    ),
    # --- L tier (>10K rows, capped at 10K for RL training) ---
    "adult": DatasetSpec(
        name="adult",
        openml_id=1590,
        eval_metric="f1",
        max_rows=10_000,
    ),
    "bank_marketing": DatasetSpec(
        name="bank_marketing",
        openml_id=1461,
        eval_metric="f1",
        max_rows=10_000,
    ),
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).parents[3] / "outputs" / "datasets"


def _cache_path(spec: DatasetSpec) -> Path:
    return _CACHE_DIR / f"{spec.name}_raw.parquet"


def _save_cache(X: pd.DataFrame, y: pd.Series, spec: DatasetSpec) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    combined = X.copy()
    combined["__target__"] = y.values
    combined.to_parquet(_cache_path(spec), index=False)
    logger.debug("Cached %s → %s", spec.name, _cache_path(spec))


def _load_cache(spec: DatasetSpec) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    p = _cache_path(spec)
    if not p.exists():
        return None
    combined = pd.read_parquet(p)
    y = combined.pop("__target__")
    return combined, y


# ---------------------------------------------------------------------------
# Core load logic
# ---------------------------------------------------------------------------

def _fetch_from_openml(spec: DatasetSpec) -> Tuple[pd.DataFrame, pd.Series]:
    """Download a dataset via the openml API and return (X, y) as DataFrames."""
    if not OPENML_AVAILABLE:
        raise RuntimeError(
            f"openml is not installed and no local cache exists for '{spec.name}'. "
            "Install with: pip install openml"
        )
    dataset = openml.datasets.get_dataset(
        spec.openml_id,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )
    target_attr = dataset.default_target_attribute
    X_raw, y_raw, _, attribute_names = dataset.get_data(
        dataset_format="dataframe",
        target=target_attr,
    )
    if y_raw is None:
        raise ValueError(
            f"Dataset {spec.name!r} (id={spec.openml_id}) has no target column "
            f"(default_target_attribute={target_attr!r})."
        )
    y = y_raw.rename("target").astype("category").cat.codes.astype(int)
    X = X_raw if isinstance(X_raw, pd.DataFrame) else pd.DataFrame(X_raw, columns=attribute_names)
    return X, y


def _preprocess(X: pd.DataFrame, y: pd.Series, spec: DatasetSpec) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply dataset-specific preprocessing and ordinal-encode categoricals."""
    X = X.copy()

    # Pima diabetes: replace physiologically impossible zeros with NaN
    for col in spec.zero_as_nan_cols:
        # case-insensitive match for robustness across openml naming variants
        matched = [c for c in X.columns if c.lower() == col.lower()]
        for c in matched:
            X[c] = X[c].replace(0, np.nan)

    # Ordinal-encode categorical / object columns
    from sklearn.preprocessing import OrdinalEncoder  # local import (optional dep guard)

    cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
        X[cat_cols] = enc.fit_transform(X[cat_cols]).astype(float)

    # Convert all remaining columns to float32
    X = X.astype(np.float32)

    return X, y


def _subsample(
    X: pd.DataFrame, y: pd.Series, max_rows: int, seed: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """Stratified subsample to at most *max_rows* rows."""
    from sklearn.model_selection import train_test_split

    if len(X) <= max_rows:
        return X, y
    # train_test_split gives us an easy stratified split
    _, X_sub, _, y_sub = train_test_split(
        X, y,
        test_size=max_rows,
        random_state=seed,
        stratify=y,
    )
    return X_sub.reset_index(drop=True), y_sub.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(
    name: str,
    *,
    use_cache: bool = True,
    force_download: bool = False,
    preprocess: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, DatasetSpec]:
    """Load a benchmark dataset by name.

    Parameters
    ----------
    name:
        One of the keys in BENCHMARK_DATASETS.
    use_cache:
        If True (default), return from local Parquet cache when available.
    force_download:
        If True, re-download from OpenML even if a cache exists.
    preprocess:
        If True (default), apply ordinal encoding and Pima zeros→NaN fix.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (float32).
    y : pd.Series
        Integer-coded target.
    spec : DatasetSpec
        The dataset specification.
    """
    if name not in BENCHMARK_DATASETS:
        raise ValueError(
            f"Unknown dataset {name!r}. Available: {sorted(BENCHMARK_DATASETS)}"
        )
    spec = BENCHMARK_DATASETS[name]

    # Try cache first
    if use_cache and not force_download:
        cached = _load_cache(spec)
        if cached is not None:
            X, y = cached
            logger.info("Loaded '%s' from cache (%d rows, %d cols)", name, len(X), X.shape[1])
            if preprocess:
                X, y = _preprocess(X, y, spec)
            if spec.max_rows is not None:
                X, y = _subsample(X, y, spec.max_rows, spec.seed)
            return X, y, spec

    # Download
    logger.info("Downloading '%s' (openml id=%d) …", name, spec.openml_id)
    X, y = _fetch_from_openml(spec)
    _save_cache(X, y, spec)

    if preprocess:
        X, y = _preprocess(X, y, spec)
    if spec.max_rows is not None:
        X, y = _subsample(X, y, spec.max_rows, spec.seed)

    logger.info("Loaded '%s': %d rows × %d cols", name, len(X), X.shape[1])
    return X, y, spec


def load_all_datasets(
    *,
    use_cache: bool = True,
    force_download: bool = False,
    preprocess: bool = True,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series, DatasetSpec]]:
    """Load all 10 benchmark datasets.

    Returns
    -------
    dict mapping dataset name → (X, y, spec).
    Failed datasets are skipped with a warning (network issues, missing deps).
    """
    results: Dict[str, Tuple[pd.DataFrame, pd.Series, DatasetSpec]] = {}
    for name in BENCHMARK_DATASETS:
        try:
            results[name] = load_dataset(
                name,
                use_cache=use_cache,
                force_download=force_download,
                preprocess=preprocess,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping '%s': %s", name, exc)
    return results
