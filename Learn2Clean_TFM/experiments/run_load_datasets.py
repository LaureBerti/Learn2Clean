"""experiments/run_load_datasets.py

Step 0 — Download and cache all 10 OpenML benchmark datasets.
============================================================
Must be run FIRST before any other experiment.  Each dataset is saved
as a Parquet file under outputs/datasets/<name>_raw.parquet.

Usage::

    PYTHONPATH=src python experiments/run_load_datasets.py
    PYTHONPATH=src python experiments/run_load_datasets.py --datasets hepatitis diabetes
    PYTHONPATH=src python experiments/run_load_datasets.py --force
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap — works whether the script is run directly or via PYTHONPATH
# ---------------------------------------------------------------------------
_SRC = Path(__file__).parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from learn2clean_v3.data.openml_loader import BENCHMARK_DATASETS, load_dataset

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Cache directory mirrors the one used by openml_loader
_CACHE_DIR = Path(__file__).parents[1] / "outputs" / "datasets"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_path(name: str) -> Path:
    return _CACHE_DIR / f"{name}_raw.parquet"


def _missing_rate(df: pd.DataFrame) -> float:
    """Return the overall cell-level missing rate (excluding __target__)."""
    feature_cols = [c for c in df.columns if c != "__target__"]
    if not feature_cols:
        return 0.0
    return float(df[feature_cols].isna().mean().mean())


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def load_all(
    dataset_names: List[str],
    force: bool,
) -> None:
    """Download and cache each requested dataset; print a summary table."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    total = len(dataset_names)
    successes = 0
    failures: List[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    col_w = (20, 7, 5, 13, 50)
    header = (
        f"{'dataset':<{col_w[0]}} "
        f"{'rows':>{col_w[1]}} "
        f"{'cols':>{col_w[2]}} "
        f"{'missing_rate':>{col_w[3]}} "
        f"{'cache_path':<{col_w[4]}}"
    )
    print()
    print(header)
    print("-" * (sum(col_w) + len(col_w) - 1))

    for name in dataset_names:
        if name not in BENCHMARK_DATASETS:
            print(
                f"  [WARN] '{name}' is not a recognised benchmark dataset — skipping. "
                f"Available: {sorted(BENCHMARK_DATASETS)}"
            )
            continue

        cache = _cache_path(name)

        # If the cache already exists and --force was not requested, skip download
        if cache.exists() and not force:
            try:
                cached_df = pd.read_parquet(cache)
                n_rows = len(cached_df)
                n_cols = cached_df.shape[1] - 1  # exclude __target__
                miss = _missing_rate(cached_df)
                successes += 1
                print(
                    f"{'[cached] ' + name:<{col_w[0]}} "
                    f"{n_rows:>{col_w[1]}} "
                    f"{n_cols:>{col_w[2]}} "
                    f"{miss:>{col_w[3]}.2%} "
                    f"{str(cache):<{col_w[4]}}"
                )
                continue
            except Exception as exc:
                print(f"  [WARN] Could not read existing cache for '{name}': {exc}. Re-downloading.")

        # Download
        t0 = time.perf_counter()
        try:
            X, y, _spec = load_dataset(
                name,
                use_cache=False,        # do not use cache on read side
                force_download=True,    # always fetch from OpenML
                preprocess=False,       # keep raw values; preprocessing happens per-experiment
            )
        except Exception as exc:
            failures.append(name)
            elapsed = time.perf_counter() - t0
            print(
                f"{'[FAIL] ' + name:<{col_w[0]}} "
                f"{'—':>{col_w[1]}} "
                f"{'—':>{col_w[2]}} "
                f"{'—':>{col_w[3]}} "
                f"FAILED in {elapsed:.1f}s — {exc}"
            )
            continue

        elapsed = time.perf_counter() - t0

        # Verify the parquet was created by load_dataset (it writes __target__ internally)
        if not cache.exists():
            failures.append(name)
            print(
                f"{'[FAIL] ' + name:<{col_w[0]}} "
                f"{'—':>{col_w[1]}} "
                f"{'—':>{col_w[2]}} "
                f"{'—':>{col_w[3]}} "
                f"Cache file missing after download — unexpected error."
            )
            continue

        # Re-read the cached file for accurate statistics (before preprocessing)
        try:
            cached_df = pd.read_parquet(cache)
        except Exception as exc:
            failures.append(name)
            print(
                f"{'[FAIL] ' + name:<{col_w[0]}} "
                f"{'—':>{col_w[1]}} "
                f"{'—':>{col_w[2]}} "
                f"{'—':>{col_w[3]}} "
                f"Parquet unreadable: {exc}"
            )
            continue

        n_rows = len(cached_df)
        n_cols = cached_df.shape[1] - 1  # exclude __target__
        miss = _missing_rate(cached_df)
        successes += 1

        print(
            f"{'[ok] ' + name:<{col_w[0]}} "
            f"{n_rows:>{col_w[1]}} "
            f"{n_cols:>{col_w[2]}} "
            f"{miss:>{col_w[3]}.2%} "
            f"{str(cache):<{col_w[4]}}  ({elapsed:.1f}s)"
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    print("-" * (sum(col_w) + len(col_w) - 1))
    print(f"\nResult: {successes}/{total} datasets loaded successfully.")
    if failures:
        print(f"Failed: {', '.join(failures)}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and cache all 10 OpenML benchmark datasets to outputs/datasets/. "
            "Must be run before any other experiment script."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        metavar="NAME",
        default=None,
        help=(
            "Subset of dataset names to load (default: all 10). "
            f"Available: {sorted(BENCHMARK_DATASETS)}"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-download from OpenML even if a local cache already exists.",
    )
    return parser.parse_args()


def main(
    dataset_names: Optional[List[str]] = None,
    force: bool = False,
) -> None:
    names = dataset_names if dataset_names is not None else list(BENCHMARK_DATASETS.keys())
    print(f"Benchmark dataset loader — {len(names)} dataset(s) requested")
    if force:
        print("  --force: existing caches will be overwritten.")
    load_all(names, force=force)


if __name__ == "__main__":
    args = _parse_args()
    main(dataset_names=args.datasets, force=args.force)
