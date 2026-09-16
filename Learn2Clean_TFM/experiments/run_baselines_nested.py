
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import run_c2_tfm_reward_nested as G
from learn2clean_v3.data.error_injection import ErrorProfile, apply_error_profile
from learn2clean_v3.data.openml_loader import BENCHMARK_DATASETS, load_dataset


def _numeric_shared(X_sel: pd.DataFrame, X_test: pd.DataFrame):
    sel = X_sel.select_dtypes(include="number").copy()
    test = X_test.select_dtypes(include="number").copy()
    shared = [c for c in sel.columns if c in test.columns]
    return sel[shared], test[shared], shared


def _impute(sel, test, shared, imputer):
    imputer.fit(sel.values)
    sel_t = pd.DataFrame(imputer.transform(sel.values), columns=shared, index=sel.index)
    test_t = pd.DataFrame(imputer.transform(test.values), columns=shared, index=test.index)
    return sel_t, test_t


def b0_none(X_sel, y_sel, X_test):
    sel, test, _ = _numeric_shared(X_sel, X_test)
    return sel, test


def b1_standard(X_sel, y_sel, X_test):
    sel, test, shared = _numeric_shared(X_sel, X_test)
    sel, test = _impute(sel, test, shared, SimpleImputer(strategy="mean"))
    scaler = MinMaxScaler().fit(sel.values)
    sel = pd.DataFrame(scaler.transform(sel.values), columns=shared, index=sel.index)
    test = pd.DataFrame(scaler.transform(test.values), columns=shared, index=test.index)
    return sel, test


def b2_full(X_sel, y_sel, X_test):
    sel, test, shared = _numeric_shared(X_sel, X_test)
    sel, test = _impute(sel, test, shared, SimpleImputer(strategy="median"))
    sel = _drop_outliers_iqr(sel)
    scaler = StandardScaler().fit(sel.values)
    sel = pd.DataFrame(scaler.transform(sel.values), columns=shared, index=sel.index)
    test = pd.DataFrame(scaler.transform(test.values), columns=shared, index=test.index)
    return sel, test


def _cm_impute(strategy):
    def fn(X_sel, y_sel, X_test):
        sel, test, shared = _numeric_shared(X_sel, X_test)
        imp = (KNNImputer(n_neighbors=5) if strategy == "knn"
               else SimpleImputer(strategy=strategy))
        return _impute(sel, test, shared, imp)
    return fn


def _drop_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    q1, q3 = df.quantile(0.25), df.quantile(0.75)
    iqr = (q3 - q1).replace(0, np.nan)
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = ((df >= lo) | lo.isna()) & ((df <= hi) | hi.isna())
    keep = mask.all(axis=1)
    return df[keep] if keep.sum() >= 10 else df


def _cm_outlier(method):
    def fn(X_sel, y_sel, X_test):
        sel, test, shared = _numeric_shared(X_sel, X_test)
        sel, test = _impute(sel, test, shared, SimpleImputer(strategy="median"))
        if method == "iqr":
            sel = _drop_outliers_iqr(sel)
        elif method == "zscore":
            z = (sel - sel.mean()) / sel.std(ddof=0).replace(0, np.nan)
            keep = (z.abs() <= 3.0).all(axis=1)
            sel = sel[keep] if keep.sum() >= 10 else sel
        elif method == "isof":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                flags = IsolationForest(contamination=0.05, random_state=0).fit_predict(sel.values)
            keep = flags == 1
            sel = sel[keep] if keep.sum() >= 10 else sel
        return sel, test
    return fn


def cm_dedup(X_sel, y_sel, X_test):
    sel, test, shared = _numeric_shared(X_sel, X_test)
    sel, test = _impute(sel, test, shared, SimpleImputer(strategy="median"))
    sel = sel[~sel.duplicated()]
    return sel, test


CLEANML_CLEANERS: Dict[str, Callable] = {
    "cm_mean":   _cm_impute("mean"),
    "cm_median": _cm_impute("median"),
    "cm_knn":    _cm_impute("knn"),
    "cm_iqr":    _cm_outlier("iqr"),
    "cm_zscore": _cm_outlier("zscore"),
    "cm_isof":   _cm_outlier("isof"),
    "cm_dedup":  cm_dedup,
}

FIXED_BASELINES: Dict[str, Callable] = {
    "b0_none": b0_none,
    "b1_standard": b1_standard,
    "b2_full": b2_full,
    **CLEANML_CLEANERS,
}


METRICS: Tuple[str, ...] = ("acc", "ece", "f1", "prec", "rec")
_NAN5 = (float("nan"),) * 5


def eval_baseline(cleaner, X_sel, y_sel, X_test, y_test, seed) -> Tuple[float, ...]:
    try:
        X_sel_clean, X_test_prep = cleaner(X_sel, y_sel, X_test)
    except Exception:
        return _NAN5
    if X_sel_clean is None or len(X_sel_clean) == 0:
        return _NAN5
    return G.final_test_tabpfn(X_sel_clean, y_sel, X_test_prep, y_test, seed)


def select_cm_best(X_sel, y_sel, X_test, y_test, seed) -> Tuple[str, Tuple[float, ...]]:
    best_name, best_val = "b0_none", -np.inf
    for name, cleaner in CLEANML_CLEANERS.items():
        try:
            X_sel_clean, _ = cleaner(X_sel, y_sel, X_test)
        except Exception:
            continue
        if X_sel_clean is None or len(X_sel_clean) == 0:
            continue
        acc = G.inner_val_tabpfn_acc(X_sel_clean, y_sel, seed)
        if np.isfinite(acc) and acc > best_val:
            best_val, best_name = acc, name
    return best_name, eval_baseline(CLEANML_CLEANERS[best_name], X_sel, y_sel, X_test, y_test, seed)


def eval_b3_random(X_sel, y_sel, X_test, y_test, seed) -> Tuple[float, ...]:
    mat = [eval_baseline(c, X_sel, y_sel, X_test, y_test, seed)
           for c in (_cm_impute("mean"), _cm_impute("median"), b1_standard)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tuple(np.nanmean(np.asarray(mat, dtype=float), axis=0))


def run_one(ds_name, seed, table3_only: bool = False) -> Optional[Dict]:
    try:
        X, y, _ = load_dataset(ds_name, use_cache=True)
    except Exception as exc:
        print(f"  [SKIP] {ds_name}: {exc}")
        return None
    if len(X) > G.SUBSAMPLE_CAP:
        X, _, y, _ = train_test_split(X, y, train_size=G.SUBSAMPLE_CAP, random_state=seed, stratify=y)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)

    Xd, yd = apply_error_profile(X, y, ErrorProfile("mcar", rate=G.MCAR_RATE, seed=seed))
    try:
        X_sel, X_test, y_sel, y_test = train_test_split(
            Xd, yd, test_size=G.OUTER_TEST_SIZE, random_state=seed, stratify=yd)
    except ValueError:
        X_sel, X_test, y_sel, y_test = train_test_split(
            Xd, yd, test_size=G.OUTER_TEST_SIZE, random_state=seed)
    X_sel, y_sel = X_sel.reset_index(drop=True), y_sel.reset_index(drop=True)
    X_test, y_test = X_test.reset_index(drop=True), y_test.reset_index(drop=True)

    row: Dict = {"dataset": ds_name, "seed": seed}
    cleaners = {k: FIXED_BASELINES[k] for k in ("b0_none", "b1_standard", "b2_full")} \
        if table3_only else FIXED_BASELINES
    for name, cleaner in cleaners.items():
        for m, v in zip(METRICS, eval_baseline(cleaner, X_sel, y_sel, X_test, y_test, seed)):
            row[f"{name}_{m}"] = v
    if not table3_only:
        best_name, best_vals = select_cm_best(X_sel, y_sel, X_test, y_test, seed)
        row["cm_best_pick"] = best_name
        for m, v in zip(METRICS, best_vals):
            row[f"cm_best_{m}"] = v
    for m, v in zip(METRICS, eval_b3_random(X_sel, y_sel, X_test, y_test, seed)):
        row[f"b3_random_{m}"] = v
    return row


def main(dataset_names=None, output_dir=None, seeds=(42,), table3_only=False) -> None:
    if dataset_names is None:
        dataset_names = list(BENCHMARK_DATASETS.keys())
    out_dir = Path(output_dir) if output_dir else (
        Path(__file__).parents[1] / "outputs" / "paper_ready" / "baselines_nested")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    t0 = time.time()
    for ds in dataset_names:
        for seed in seeds:
            print(f"\n── {ds} | seed {seed} ──", flush=True)
            ts = time.time()
            r = run_one(ds, seed, table3_only=table3_only)
            if r is not None:
                rows.append(r)
                extra = "" if table3_only else f"cm_best={r['cm_best_acc']:.4f} ({r['cm_best_pick']}) "
                print(f"   b0={r['b0_none_acc']:.4f} b1={r['b1_standard_acc']:.4f} "
                      f"b2={r['b2_full_acc']:.4f} b3={r['b3_random_acc']:.4f} {extra}"
                      f"({time.time()-ts:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(out_dir / "baselines_per_seed.csv", index=False)

    if not rows:
        print("No results."); return
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "baselines_per_seed.csv", index=False)
    metric_cols = [c for c in df.columns if c.rsplit("_", 1)[-1] in METRICS]
    agg = df.groupby("dataset")[metric_cols].agg(["mean", "std", "count"])
    agg.to_csv(out_dir / "baselines_aggregated.csv")
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min → {out_dir}")


def diffprep_cleaner(X_sel, y_sel, X_test):
    raise NotImplementedError("DiffPrep wrapper pending — see docstring.")


def autosklearn_cleaner(X_sel, y_sel, X_test):
    raise NotImplementedError("Auto-sklearn wrapper pending — see docstring.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--table3-only", action="store_true",
                    help="only B-NC/SP/SC/RAND (skip CleanML cleaners + cm_best) — ~1/3 the cost")
    a = ap.parse_args()
    main(a.datasets, a.output_dir, tuple(a.seeds), table3_only=a.table3_only)
