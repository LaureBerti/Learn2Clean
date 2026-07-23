"""
experiments/run_corruption_sweep.py

Corruption-TYPE sweep + label-aware reward arm — where does cleaning (and a TFM-aware
reward) actually matter for TabPFN? Held-out protocol, on OpenML(10) + SAGA(3) datasets.

Motivation (from prior results): TabPFN is robust to the corruptions we tried (MCAR,
outliers) → cleaning is inert → no reward can win. The exception should be corruptions
TabPFN is NOT robust to — chiefly LABEL NOISE, which directly pollutes the in-context set
(RF is comparatively robust via ensembling). So we sweep corruption type and add a
label-aware arm to see if a genuine, scoped win appears.

Conditions (corruption type @ rate):
  none · mcar@0.2 · mar@0.2 · outlier@0.2 · duplicate@0.2 · label@0.2 · label@0.35
Feature corruptions hit ALL data (test features also dirty, realistic). LABEL noise hits
ONLY the training/selection labels (y_sel); test labels stay ground truth (we measure
accuracy against the TRUE labels — cleaning's job is to de-noise the in-context set).

Arms (all held-out protocol: inner-val selection, final eval on untouched test):
  no_clean : raw (dirty) D_sel → TabPFN
  R3       : RF-reward selection over base ops (impute/outlier/scale)
  R7       : TabPFN-reward selection over base ops
  R7+label : TabPFN-reward over base ops + a MISLABEL-REMOVAL operator (the label-aware arm)

Key gaps: (R7+label − R7) isolates the label operator's value; (R7 − no_clean) is generic
cleaning's value. Prediction: ≈0 for feature corruptions (inert), >0 for label noise via R7+label.

Usage
-----
  PYTHONPATH=src:experiments python experiments/run_corruption_sweep.py --datasets hepatitis EEG --seeds 42 1 2
"""
from __future__ import annotations

import argparse
import time
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import LabelEncoder

import run_c2_tfm_reward_nested as G
import run_weight_robustness as W
from learn2clean_v3.actions import (DataFrameAction, ParameterizedImputer,
                                    ParameterizedOutlierCleaner, ParameterizedScaler)
from learn2clean_v3.data.error_injection import ErrorProfile, apply_error_profile
from learn2clean_v3.data.openml_loader import BENCHMARK_DATASETS, load_dataset
from run_saga_comparison import DATASETS as SAGA_DS, load_saga

CONDITIONS = [("none", 0.0), ("mcar", 0.2), ("mar", 0.2), ("outlier", 0.2),
              ("duplicate", 0.2), ("label", 0.2), ("label", 0.35)]


# --------------------------------------------------------------------------- #
# Label-noise injection (training labels only) + mislabel-removal operator
# --------------------------------------------------------------------------- #
def inject_label_noise(y: pd.Series, rate: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    yv = np.asarray(y).copy()
    classes = np.unique(yv)
    if len(classes) < 2:
        return y
    flip = rng.random(len(yv)) < rate
    for i in np.where(flip)[0]:
        yv[i] = rng.choice(classes[classes != yv[i]])
    return pd.Series(yv, index=y.index)


class LabelCleaner:
    """Mislabel removal (confident-learning lite): drop rows where a CV LogReg confidently
    disagrees with the given label. Uses y → only valid on the training/selection set.
    Plain class (apply_pipeline only needs .reset() and __call__)."""
    def reset(self):  # noqa: D401
        return self

    def __call__(self, X: pd.DataFrame, y) -> pd.DataFrame:
        num = X.select_dtypes(include="number")
        if num.shape[1] == 0 or len(X) < 40:
            return X
        try:
            yy = y.loc[num.index] if hasattr(y, "loc") else pd.Series(np.asarray(y)[:len(num)], index=num.index)
        except Exception:
            return X
        Xn = num.fillna(num.median())
        ye = LabelEncoder().fit_transform(np.asarray(yy).astype(str))
        if len(np.unique(ye)) < 2:
            return X
        try:
            proba = cross_val_predict(LogisticRegression(max_iter=300), Xn.values, ye, cv=3, method="predict_proba")
        except Exception:
            return X
        drop = (proba.argmax(1) != ye) & (proba.max(1) > 0.70)
        keep = num.index[~drop]
        return X.loc[keep] if len(keep) >= 20 else X


BASE = [ParameterizedImputer(strategy="mean"), ParameterizedImputer(strategy="median"),
        ParameterizedImputer(strategy="knn", n_neighbors=5),
        ParameterizedOutlierCleaner(method="iqr", threshold=1.5),
        ParameterizedOutlierCleaner(method="zscore", threshold=3.0),
        ParameterizedScaler(method="minmax"), ParameterizedScaler(method="zscore")]
BASE_GROUP = ["impute", "impute", "impute", "outlier", "outlier", "scale", "scale"]
EXT = BASE + [LabelCleaner()]
EXT_GROUP = BASE_GROUP + ["label"]


def enum_pipes(groups, max_len=3):
    out = [()]
    for L in range(1, max_len + 1):
        for seq in permutations(range(len(groups)), L):
            if len({groups[i] for i in seq}) == L:
                out.append(seq)
    return out


def select(X_sel, y_sel, pipes, actions, seed, estimator) -> Tuple[int, ...]:
    n0 = len(X_sel); best, bs = (), -np.inf
    for seq in pipes:
        Xc = G.apply_pipeline(X_sel, y_sel, seq, actions)
        if Xc is None or len(Xc) == 0:
            continue
        acc = W.inner_val_acc(Xc, y_sel, seed, estimator)
        if not np.isfinite(acc):
            continue
        miss = float(Xc.isna().mean().mean()); dup = float(Xc.duplicated().sum()) / max(len(Xc), 1)
        s = 0.50 * acc + 0.35 * (len(Xc) / n0) ** 2 + 0.15 * (1 - miss) * (1 - dup)
        if s > bs:
            bs, best = s, seq
    return best


def load_any(name):
    if name in SAGA_DS:
        return load_saga(name)
    X, y, _ = load_dataset(name, use_cache=True)
    return X, y


def eval_arm(X_sel, y_sel, X_test, y_test, seq, actions, seed) -> Dict[str, float]:
    Xc = G.apply_pipeline(X_sel, y_sel, seq, actions)
    if Xc is None or len(Xc) == 0:
        return {"tabpfn": np.nan, "logreg": np.nan}
    Xtp = G.prepare_test_like_train(X_sel, X_test, seq)
    Xtr, ytr, le = G._encode_align(Xc, y_sel)
    shared = [c for c in Xc.select_dtypes(include="number").columns if c in Xtp.columns]
    try:
        yte = le.transform(np.asarray(y_test)); Xte = Xtp[shared].values.astype(float)
        if len(np.unique(ytr)) < 2 or len(yte) == 0:
            return {"tabpfn": np.nan, "logreg": np.nan}
        panel = W.panel_on_test(Xtr, ytr, Xte, yte, seed)
    except Exception:
        return {"tabpfn": np.nan, "logreg": np.nan}
    return {"tabpfn": panel.get("tabpfn", {}).get("acc", np.nan),
            "logreg": panel.get("logreg", {}).get("acc", np.nan)}


def run_one(ds, ctype, rate, seed, base_pipes, ext_pipes) -> Optional[Dict]:
    loaded = load_any(ds)
    if loaded is None:
        return None
    X, y = loaded
    if len(X) > G.SUBSAMPLE_CAP:
        X, _, y, _ = train_test_split(X, y, train_size=G.SUBSAMPLE_CAP, random_state=seed, stratify=y)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)

    if ctype in ("mcar", "mar", "outlier", "duplicate"):
        prof = ErrorProfile(ctype, rate=rate, k=5.0) if ctype == "outlier" else ErrorProfile(ctype, rate=rate)
        Xd, yd = apply_error_profile(X, y, prof)
    else:
        Xd, yd = X.copy(), y.copy()
    try:
        X_sel, X_test, y_sel, y_test = train_test_split(Xd, yd, test_size=0.30, random_state=seed, stratify=yd)
    except ValueError:
        X_sel, X_test, y_sel, y_test = train_test_split(Xd, yd, test_size=0.30, random_state=seed)
    X_sel, y_sel = X_sel.reset_index(drop=True), y_sel.reset_index(drop=True)
    X_test, y_test = X_test.reset_index(drop=True), y_test.reset_index(drop=True)
    # LABEL noise: corrupt only the training labels; test labels stay ground truth
    if ctype == "label":
        y_sel = inject_label_noise(y_sel, rate, seed)

    row = {"dataset": ds, "ctype": f"{ctype}_{rate}", "seed": seed}
    # arms
    arms = {
        "no_clean": ((), BASE),
        "R3": (select(X_sel, y_sel, base_pipes, BASE, seed, "rf"), BASE),
        "R7": (select(X_sel, y_sel, base_pipes, BASE, seed, "tabpfn"), BASE),
        "R7label": (select(X_sel, y_sel, ext_pipes, EXT, seed, "tabpfn"), EXT),
    }
    for name, (seq, acts) in arms.items():
        m = eval_arm(X_sel, y_sel, X_test, y_test, seq, acts, seed)
        row[f"{name}_tabpfn"], row[f"{name}_logreg"] = m["tabpfn"], m["logreg"]
        row[f"{name}_uses_label"] = int(any(EXT_GROUP[i] == "label" for i in seq) if acts is EXT else 0)
    return row


def main(datasets, seeds, max_pipelines, output_dir=None) -> None:
    out_dir = Path(output_dir) if output_dir else (Path(__file__).parents[1] / "outputs" / "paper_ready" / "corruption_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    base_pipes = G.sample_pipelines(enum_pipes(BASE_GROUP), max_pipelines, 42)
    ext_pipes = G.sample_pipelines(enum_pipes(EXT_GROUP), max_pipelines + 6, 42)
    rows: List[Dict] = []; t0 = time.time()
    for ds in datasets:
        for (ctype, rate) in CONDITIONS:
            for seed in seeds:
                r = run_one(ds, ctype, rate, seed, base_pipes, ext_pipes)
                if r:
                    rows.append(r)
                pd.DataFrame(rows).to_csv(out_dir / "corruption_per_run.csv", index=False)
            sub = [x for x in rows if x["dataset"] == ds and x["ctype"] == f"{ctype}_{rate}"]
            if sub:
                nc = np.nanmean([x["no_clean_tabpfn"] for x in sub])
                r7 = np.nanmean([x["R7_tabpfn"] for x in sub])
                r7l = np.nanmean([x["R7label_tabpfn"] for x in sub])
                print(f"  {ds:14} {ctype+'_'+str(rate):14}: no_clean={nc:.4f} R7={r7:.4f} R7+label={r7l:.4f} "
                      f"| Δlabel={r7l-r7:+.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "corruption_per_run.csv", index=False)
    if len(df):
        agg = df.groupby("ctype").agg(
            no_clean=("no_clean_tabpfn", "mean"), R3=("R3_tabpfn", "mean"),
            R7=("R7_tabpfn", "mean"), R7label=("R7label_tabpfn", "mean"),
            label_used=("R7label_uses_label", "mean")).reset_index()
        agg["gap_label_op"] = agg["R7label"] - agg["R7"]
        agg["gap_clean"] = agg["R7"] - agg["no_clean"]
        agg.to_csv(out_dir / "corruption_by_ctype.csv", index=False)
        print("\n=== BY CORRUPTION TYPE (TabPFN acc) ===")
        print(agg.to_string(index=False))
    print(f"\nWall-clock {(time.time()-t0)/60:.1f} min → {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(BENCHMARK_DATASETS.keys()) + list(SAGA_DS.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2, 3, 4])
    ap.add_argument("--max-pipelines", type=int, default=12)
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    main(a.datasets, tuple(a.seeds), a.max_pipelines, a.output_dir)
