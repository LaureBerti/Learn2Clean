
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import openml

    OPENML_AVAILABLE = True
except ImportError:
    OPENML_AVAILABLE = False
    logger.warning(
        "openml not installed — load_dataset() will only work from local cache. "
        "Install with: pip install openml"
    )


@dataclass
class DatasetSpec:

    name: str
    openml_id: int
    eval_metric: str = "f1"
    max_rows: Optional[int] = None
    zero_as_nan_cols: List[str] = field(default_factory=list)
    seed: int = 42



BENCHMARK_DATASETS: Dict[str, DatasetSpec] = {
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
        zero_as_nan_cols=["plas", "pres", "skin", "insu", "mass"],
    ),
    "credit_g": DatasetSpec(
        name="credit_g",
        openml_id=31,
        eval_metric="f1",
        max_rows=None,
    ),
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



def _fetch_from_openml(spec: DatasetSpec) -> Tuple[pd.DataFrame, pd.Series]:
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
    X = X.copy()

    for col in spec.zero_as_nan_cols:
        matched = [c for c in X.columns if c.lower() == col.lower()]
        for c in matched:
            X[c] = X[c].replace(0, np.nan)

    from sklearn.preprocessing import OrdinalEncoder

    cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
        X[cat_cols] = enc.fit_transform(X[cat_cols]).astype(float)

    X = X.astype(np.float32)

    return X, y


def _subsample(
    X: pd.DataFrame, y: pd.Series, max_rows: int, seed: int
) -> Tuple[pd.DataFrame, pd.Series]:
    from sklearn.model_selection import train_test_split

    if len(X) <= max_rows:
        return X, y
    _, X_sub, _, y_sub = train_test_split(
        X, y,
        test_size=max_rows,
        random_state=seed,
        stratify=y,
    )
    return X_sub.reset_index(drop=True), y_sub.reset_index(drop=True)



def load_dataset(
    name: str,
    *,
    use_cache: bool = True,
    force_download: bool = False,
    preprocess: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, DatasetSpec]:
    if name not in BENCHMARK_DATASETS:
        raise ValueError(
            f"Unknown dataset {name!r}. Available: {sorted(BENCHMARK_DATASETS)}"
        )
    spec = BENCHMARK_DATASETS[name]

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
    results: Dict[str, Tuple[pd.DataFrame, pd.Series, DatasetSpec]] = {}
    for name in BENCHMARK_DATASETS:
        try:
            results[name] = load_dataset(
                name,
                use_cache=use_cache,
                force_download=force_download,
                preprocess=preprocess,
            )
        except Exception as exc:
            logger.warning("Skipping '%s': %s", name, exc)
    return results
