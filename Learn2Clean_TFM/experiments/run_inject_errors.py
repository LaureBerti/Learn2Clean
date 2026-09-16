
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

_SRC = Path(__file__).parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from learn2clean_v3.data.error_injection import (
    ErrorProfile,
    apply_error_profile,
    generate_all_profiles,
)
from learn2clean_v3.data.openml_loader import BENCHMARK_DATASETS

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parents[1] / "outputs" / "datasets"



def _raw_cache_path(name: str) -> Path:
    return _CACHE_DIR / f"{name}_raw.parquet"


def _output_path(name: str, profile: ErrorProfile) -> Path:
    return _CACHE_DIR / f"{name}_{profile.tag}.parquet"


def _load_raw(name: str) -> tuple[pd.DataFrame, pd.Series]:
    path = _raw_cache_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw cache not found: {path}\n"
            "Run 'PYTHONPATH=src python experiments/run_load_datasets.py' first."
        )
    combined = pd.read_parquet(path)
    y = combined.pop("__target__")
    return combined, y


def _save_variant(
    X_dirty: pd.DataFrame,
    y_dirty: pd.Series,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = X_dirty.copy()
    df["__target__"] = y_dirty.values
    df.to_parquet(out_path, index=False, engine="pyarrow")



def inject_all(
    dataset_names: List[str],
    profiles: List[ErrorProfile],
    force: bool,
) -> None:
    n_datasets = len(dataset_names)
    n_profiles = len(profiles)
    n_total = n_datasets * n_profiles

    generated = 0
    skipped = 0
    failed = 0

    print(
        f"\nError injection — {n_datasets} dataset(s) × {n_profiles} profile(s) "
        f"= {n_total} file(s) planned\n"
    )

    for name in dataset_names:
        if name not in BENCHMARK_DATASETS:
            print(
                f"  [WARN] '{name}' is not a recognised benchmark dataset — skipping. "
                f"Available: {sorted(BENCHMARK_DATASETS)}"
            )
            continue

        try:
            X_raw, y_raw = _load_raw(name)
        except FileNotFoundError as exc:
            print(f"  [FAIL] {name}: {exc}")
            failed += n_profiles
            continue
        except Exception as exc:
            print(f"  [FAIL] {name}: could not read raw cache — {exc}")
            failed += n_profiles
            continue

        for profile in profiles:
            out_path = _output_path(name, profile)

            if out_path.exists() and not force:
                skipped += 1
                logger.debug("skip existing: %s", out_path)
                continue

            try:
                X_dirty, y_dirty = apply_error_profile(X_raw, y_raw, profile)
            except Exception as exc:
                failed += 1
                print(f"  [FAIL] {name} {profile.tag} — injection error: {exc}")
                continue

            try:
                _save_variant(X_dirty, y_dirty, out_path)
            except Exception as exc:
                failed += 1
                print(f"  [FAIL] {name} {profile.tag} — save error: {exc}")
                continue

            generated += 1
            print(f"  {name:<20} {profile.tag:<16} → {out_path}")

    print()
    print(
        f"Done — generated: {generated}, "
        f"skipped (already exist): {skipped}, "
        f"failed: {failed}"
    )
    if failed:
        print("  Tip: run with --force to retry failed variants.")
    print()



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate all synthetic error variants for cached benchmark datasets. "
            "Requires run_load_datasets.py to have been run first."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        metavar="NAME",
        default=None,
        help=(
            "Subset of dataset names to process (default: all 10). "
            f"Available: {sorted(BENCHMARK_DATASETS)}"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing output parquet files.",
    )
    return parser.parse_args()


def main(
    dataset_names: Optional[List[str]] = None,
    force: bool = False,
) -> None:
    names = dataset_names if dataset_names is not None else list(BENCHMARK_DATASETS.keys())
    profiles = generate_all_profiles(include_none=True)

    if force:
        print("  --force: existing output files will be overwritten.")

    inject_all(names, profiles, force=force)


if __name__ == "__main__":
    args = _parse_args()
    main(dataset_names=args.datasets, force=args.force)
