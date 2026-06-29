"""

C2 (REVISION) — TFM-Aware Reward with a LEAK-FREE nested evaluation protocol
============================================================================
This is the revision-round replacement for ``run_c2_tfm_reward.py``. It exists to
close the selection-leakage concern raised by PVLDB Reviewer #3 (W1) and elevated
by the AC to the explicit acceptance gate:

    "If the same held-out split is used both to select/optimize cleaning pipelines
     and to report final performance, this leads to selection leakage and
     invalidates the empirical results and claims."

Root cause in the original script
----------------------------------
``build_tabpfn_cache`` scored every pipeline on ONE seed-42 train/test split, then
``best_from_tfm_cache`` selected the argmax of that cached accuracy, and the SAME
cached number was reported as the final result (run_c2_tfm_reward.py:522,541,542).
Selection metric == reported metric == same split → optimistic selection bias.

Leak-free protocol implemented here
-----------------------------------
For each dataset and each random seed:

  1. Inject MCAR 15% on the full dataset (seed).
  2. OUTER split (stratified, fixed seed): D_sel (80%) and D_test (20%).
     D_test is NEVER seen by any selection or reward computation.
  3. SELECTION — operates only on D_sel:
       * clean D_sel with each candidate pipeline;
       * score each cleaned D_sel with an INNER train/val split (TabPFN for the
         TFM mode, cross-validated RandomForest for the RF mode);
       * pick the argmax pipeline per mode.
  4. FINAL EVALUATION — reported on the untouched D_test:
       * refit the selected cleaning transforms on D_sel (train-fitted imputation
         and scaling), apply them to D_test WITHOUT deleting test rows
         (outlier-removal / dedup only shape the training context, never the test);
       * fit TabPFN on the cleaned D_sel context, predict on the prepared D_test;
       * report accuracy and ECE on D_test only.

The number reported in step 4 is computed on data that played no role in selecting
the pipeline — this is the separation R3 and the AC asked for.

Multi-seed
----------
Pass ``--seeds 42 1 2 3 4`` to repeat the whole protocol over seeds and aggregate
mean ± 95% CI per dataset (closes R3-W4 variance concern in the same run).

Usage
-----
  PYTHONPATH=src python experiments/run_c2_tfm_reward_nested.py
  PYTHONPATH=src python experiments/run_c2_tfm_reward_nested.py --seeds 42 1 2 3 4
  PYTHONPATH=src python experiments/run_c2_tfm_reward_nested.py --datasets hepatitis ionosphere --seeds 42
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# --------------------------------------------------------------------------- #
# Dependency check
# --------------------------------------------------------------------------- #
try:
    import tabpfn as _tabpfn_check  # noqa: F401
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False

if not TABPFN_AVAILABLE:
    sys.exit("Install tabpfn>=2.0 first (pip install tabpfn>=2.0)")

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from learn2clean_v3.actions import (
    DataFrameAction,
    ParameterizedImputer,
    ParameterizedOutlierCleaner,
    ParameterizedScaler,
)
from learn2clean_v3.data.error_injection import ErrorProfile, apply_error_profile
from learn2clean_v3.data.openml_loader import BENCHMARK_DATASETS, load_dataset
from learn2clean_v3.rewards import MultiObjectiveReward

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
NATURAL_MISSING: frozenset = frozenset({"hepatitis", "diabetes", "adult"})
MCAR_RATE: float = 0.15
N_BINS_ECE: int = 10
OUTER_TEST_SIZE: float = 0.20          # untouched final-test fraction
INNER_VAL_SIZE: float = 0.25           # validation fraction WITHIN D_sel (selection only)
CONTEXT_CAP: int = 1024                # max TabPFN training-context rows
SUBSAMPLE_CAP: int = 4096              # cap very large datasets before splitting

ACTION_GROUPS: Dict[int, str] = {0: "impute", 1: "impute", 2: "impute",
                                 3: "outlier", 4: "outlier", 5: "scale", 6: "scale"}
ACTION_LABELS: Dict[int, str] = {0: "impute(mean)", 1: "impute(median)", 2: "impute(knn)",
                                 3: "outlier(iqr)", 4: "outlier(zscore)",
                                 5: "scale(minmax)", 6: "scale(zscore)"}


# --------------------------------------------------------------------------- #
# Pipeline enumeration / application (training context only)
# --------------------------------------------------------------------------- #
def build_actions() -> List[DataFrameAction]:
    return [
        ParameterizedImputer(strategy="mean"),
        ParameterizedImputer(strategy="median"),
        ParameterizedImputer(strategy="knn", n_neighbors=5),
        ParameterizedOutlierCleaner(method="iqr", threshold=1.5),
        ParameterizedOutlierCleaner(method="zscore", threshold=3.0),
        ParameterizedScaler(method="minmax"),
        ParameterizedScaler(method="zscore"),
    ]


def enumerate_valid_pipelines(max_len: int = 3) -> List[Tuple[int, ...]]:
    result: List[Tuple[int, ...]] = [()]
    for length in range(1, max_len + 1):
        for seq in permutations(range(len(ACTION_GROUPS)), length):
            groups = [ACTION_GROUPS[i] for i in seq]
            if len(groups) == len(set(groups)):
                result.append(seq)
    return result


def sample_pipelines(pipelines: List[Tuple[int, ...]], max_n: int, seed: int) -> List[Tuple[int, ...]]:
    if max_n <= 0 or max_n >= len(pipelines):
        return pipelines
    noop = [p for p in pipelines if len(p) == 0]
    one = [p for p in pipelines if len(p) == 1]
    rest = [p for p in pipelines if len(p) >= 2]
    budget = max_n - len(noop) - len(one)
    if budget <= 0:
        return (noop + one)[:max_n]
    rng = np.random.default_rng(seed)
    pick = list(rng.choice(len(rest), size=min(budget, len(rest)), replace=False))
    return noop + one + [rest[i] for i in sorted(pick)]


def pipeline_label(pipeline: Tuple[int, ...]) -> str:
    return "no_op" if not pipeline else " → ".join(ACTION_LABELS[i] for i in pipeline)


def apply_pipeline(X: pd.DataFrame, y: pd.Series, pipeline: Tuple[int, ...],
                   actions: List[DataFrameAction]) -> Optional[pd.DataFrame]:
    """Apply a sequence of actions to the TRAINING context. Returns None on failure."""
    X_out = X.copy()
    for idx in pipeline:
        try:
            actions[idx].reset()
            X_out = actions[idx](X_out.copy(), y)
        except Exception:
            return None
    return X_out


# --------------------------------------------------------------------------- #
# Leak-free TEST preparation: fit transforms on D_sel, apply to D_test.
# Row-removing actions (outlier, dedup) are NOT applied to the test set — they
# only shape the training context. Imputation and scaling are fit on the training
# context and applied to the test features (TabPFN also normalises internally).
# --------------------------------------------------------------------------- #
def prepare_test_like_train(
    X_sel: pd.DataFrame, X_test: pd.DataFrame, pipeline: Tuple[int, ...],
) -> pd.DataFrame:
    sel = X_sel.select_dtypes(include="number").copy()
    test = X_test.select_dtypes(include="number").copy()
    shared = [c for c in sel.columns if c in test.columns]
    sel, test = sel[shared], test[shared]

    for idx in pipeline:
        # .get() tolerates extended action spaces (e.g. the corruption sweep's label-cleaner
        # at idx 7): any action that is neither impute nor scale is row-removing and applies to
        # the TRAINING context only, so it falls through and is correctly skipped on the test set.
        label = ACTION_LABELS.get(idx, "")
        if label.startswith("impute"):
            if "knn" in label:
                imp = KNNImputer(n_neighbors=5)
            elif "mean" in label:
                imp = SimpleImputer(strategy="mean")
            else:
                imp = SimpleImputer(strategy="median")
            imp.fit(sel.values)
            sel = pd.DataFrame(imp.transform(sel.values), columns=shared, index=sel.index)
            test = pd.DataFrame(imp.transform(test.values), columns=shared, index=test.index)
        elif label.startswith("scale"):
            scaler = MinMaxScaler() if "minmax" in label else StandardScaler()
            scaler.fit(sel.values)
            sel = pd.DataFrame(scaler.transform(sel.values), columns=shared, index=sel.index)
            test = pd.DataFrame(scaler.transform(test.values), columns=shared, index=test.index)
        # outlier(*) → row removal: training-context only, skip on test
    return test


# --------------------------------------------------------------------------- #
# TabPFN helpers
# --------------------------------------------------------------------------- #
def _encode_align(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    numeric = X.select_dtypes(include="number")
    try:
        y_aligned = y.loc[numeric.index]
    except (KeyError, AttributeError):
        y_aligned = np.asarray(y)[: len(numeric)]
    le = LabelEncoder()
    y_enc = le.fit_transform(np.asarray(y_aligned))
    return numeric.values.astype(float), y_enc, le


def _tabpfn_cfg() -> dict:
    """TabPFN configuration, env-overridable for the tuning ablation (defaults match
    the D1 run, so behaviour is unchanged unless these env vars are set)."""
    import os
    return {
        "n_estimators": int(os.environ.get("TABPFN_N_ESTIMATORS", "8")),
        "softmax_temperature": float(os.environ.get("TABPFN_TEMP", "0.9")),
        "balance_probabilities": os.environ.get("TABPFN_BALANCE", "0") == "1",
        "ctx_cap": int(os.environ.get("TABPFN_CTX_CAP", str(CONTEXT_CAP))),
    }


def _cap_context(X: np.ndarray, y: np.ndarray, seed: int, cap: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if cap is None:
        cap = _tabpfn_cfg()["ctx_cap"]
    if len(X) <= cap:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=cap, replace=False)
    return X[idx], y[idx]


def _tabpfn_fit_predict(Xtr, ytr, Xte, seed: int):
    """Return (y_pred, y_prob) from a TabPFN v2 fit on (Xtr,ytr), predict on Xte."""
    from tabpfn import TabPFNClassifier
    cfg = _tabpfn_cfg()
    Xtr, ytr = _cap_context(Xtr, ytr, seed, cfg["ctx_cap"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = TabPFNClassifier(
            device="cpu", ignore_pretraining_limits=True,
            n_estimators=cfg["n_estimators"],
            softmax_temperature=cfg["softmax_temperature"],
            balance_probabilities=cfg["balance_probabilities"],
        )
        clf.fit(Xtr, ytr)
        y_prob = clf.predict_proba(Xte)
        y_pred = clf.predict(Xte)
    return y_pred, y_prob


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS_ECE) -> float:
    n_total = len(y_true)
    if n_total == 0:
        return float("nan")
    if y_prob.ndim > 1 and y_prob.shape[1] > 1:
        conf = y_prob.max(axis=1)
        correct = (y_prob.argmax(axis=1) == y_true).astype(int)
    else:
        conf = y_prob.ravel()
        correct = y_true.astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf >= bins[i]) & (conf < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += abs(conf[mask].mean() - correct[mask].mean()) * mask.sum() / n_total
    return float(ece)


def inner_val_tabpfn_acc(X_clean: pd.DataFrame, y: pd.Series, seed: int) -> float:
    """SELECTION score: TabPFN accuracy via an inner train/val split WITHIN D_sel.
    Never touches the outer test set."""
    X_arr, y_enc, _ = _encode_align(X_clean, y)
    if len(y_enc) < 20 or len(np.unique(y_enc)) < 2:
        return float("nan")
    try:
        Xtr, Xval, ytr, yval = train_test_split(
            X_arr, y_enc, test_size=INNER_VAL_SIZE, random_state=seed, stratify=y_enc)
    except ValueError:
        return float("nan")
    if len(np.unique(ytr)) < 2:
        return float("nan")
    try:
        y_pred, _ = _tabpfn_fit_predict(Xtr, ytr, Xval, seed)
        return float(accuracy_score(yval, y_pred))
    except Exception as exc:
        logger.debug("inner-val TabPFN failed: %s", exc)
        return float("nan")


def final_test_tabpfn(
    X_sel_clean: pd.DataFrame, y_sel: pd.Series,
    X_test_prepared: pd.DataFrame, y_test: pd.Series, seed: int,
) -> Tuple[float, float]:
    """FINAL metric: fit TabPFN on cleaned D_sel context, evaluate on untouched D_test."""
    Xtr, ytr, le = _encode_align(X_sel_clean, y_sel)
    shared = [c for c in X_sel_clean.select_dtypes(include="number").columns
              if c in X_test_prepared.columns]
    Xte = X_test_prepared[shared].values.astype(float)
    try:
        yte = le.transform(np.asarray(y_test))
    except Exception:
        return float("nan"), float("nan")
    if len(np.unique(ytr)) < 2 or len(yte) == 0:
        return float("nan"), float("nan")
    try:
        y_pred, y_prob = _tabpfn_fit_predict(Xtr, ytr, Xte, seed)
        return float(accuracy_score(yte, y_pred)), compute_ece(yte, y_prob)
    except Exception as exc:
        logger.debug("final-test TabPFN failed: %s", exc)
        return float("nan"), float("nan")


# --------------------------------------------------------------------------- #
# Selection (D_sel only)
# --------------------------------------------------------------------------- #
def select_best(
    X_sel: pd.DataFrame, y_sel: pd.Series, pipelines, actions, seed: int,
    rf_reward: MultiObjectiveReward,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return (best_rf_pipeline, best_tfm_pipeline) selected on D_sel only."""
    n0 = len(X_sel)
    best_rf, best_rf_score = (), -np.inf
    best_tfm, best_tfm_score = (), -np.inf
    w_acc, w_ret, w_qual, alpha = 0.50, 0.35, 0.15, 2.0

    for seq in pipelines:
        X_clean = apply_pipeline(X_sel, y_sel, seq, actions)
        if X_clean is None or len(X_clean) == 0:
            continue

        # RF mode: MultiObjectiveReward computes its own internal CV on D_sel (leak-free wrt test)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf_reward.reset(X_sel, y_sel)
            rf_score = rf_reward(X_clean, y_sel)
        if np.isfinite(rf_score) and rf_score > best_rf_score:
            best_rf_score, best_rf = rf_score, seq

        # TFM mode: inner-val TabPFN accuracy + retention + quality, all on D_sel
        acc = inner_val_tabpfn_acc(X_clean, y_sel, seed)
        if not np.isfinite(acc):
            continue
        retention = (len(X_clean) / n0) ** alpha
        miss = float(X_clean.isna().mean().mean())
        dup = float(X_clean.duplicated().sum()) / max(len(X_clean), 1)
        quality = (1.0 - miss) * (1.0 - dup)
        tfm_score = w_acc * acc + w_ret * retention + w_qual * quality
        if tfm_score > best_tfm_score:
            best_tfm_score, best_tfm = tfm_score, seq

    return best_rf, best_tfm


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_one(ds_name: str, seed: int, pipelines, actions) -> Optional[Dict]:
    try:
        X, y, spec = load_dataset(ds_name, use_cache=True)
    except Exception as exc:
        print(f"  [SKIP] {ds_name}: load failed: {exc}")
        return None

    # Subsample very large datasets before everything (stratified, seeded)
    if len(X) > SUBSAMPLE_CAP:
        Xs, _, ys, _ = train_test_split(X, y, train_size=SUBSAMPLE_CAP,
                                        random_state=seed, stratify=y)
        X, y = Xs.reset_index(drop=True), ys.reset_index(drop=True)

    mcar = ErrorProfile("mcar", rate=MCAR_RATE, seed=seed)
    X_dirty, y_dirty = apply_error_profile(X, y, mcar)

    # OUTER split — D_test is untouched by selection
    try:
        X_sel, X_test, y_sel, y_test = train_test_split(
            X_dirty, y_dirty, test_size=OUTER_TEST_SIZE,
            random_state=seed, stratify=y_dirty)
    except ValueError:
        X_sel, X_test, y_sel, y_test = train_test_split(
            X_dirty, y_dirty, test_size=OUTER_TEST_SIZE, random_state=seed)
    X_sel = X_sel.reset_index(drop=True); y_sel = y_sel.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True); y_test = y_test.reset_index(drop=True)

    rf_reward = MultiObjectiveReward(
        weight_accuracy=0.5, weight_retention=0.3, weight_quality=0.2,
        drift_penalty_coeff=0.1, eval_model="random_forest",
        eval_metric=spec.eval_metric, eval_cv_folds=3,
    )

    best_rf, best_tfm = select_best(X_sel, y_sel, pipelines, actions, seed, rf_reward)

    out = {"dataset": ds_name, "seed": seed,
           "rf_pipeline": pipeline_label(best_rf),
           "tfm_pipeline": pipeline_label(best_tfm),
           "pipelines_match": int(best_rf == best_tfm)}

    for mode, best in (("rf", best_rf), ("tfm", best_tfm)):
        X_clean = apply_pipeline(X_sel, y_sel, best, actions)
        if X_clean is None:
            out[f"{mode}_acc"], out[f"{mode}_ece"] = float("nan"), float("nan")
            continue
        X_test_prep = prepare_test_like_train(X_sel, X_test, best)
        acc, ece = final_test_tabpfn(X_clean, y_sel, X_test_prep, y_test, seed)
        out[f"{mode}_acc"], out[f"{mode}_ece"] = acc, ece
    return out


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± 95% CI per dataset across seeds."""
    rows = []
    for ds, g in df.groupby("dataset"):
        row = {"dataset": ds, "n_seeds": len(g)}
        for col in ("rf_acc", "tfm_acc", "rf_ece", "tfm_ece"):
            vals = g[col].dropna().values
            if len(vals) == 0:
                row[f"{col}_mean"], row[f"{col}_ci95"] = float("nan"), float("nan")
            else:
                m = float(np.mean(vals))
                ci = 1.96 * float(np.std(vals, ddof=1)) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                row[f"{col}_mean"], row[f"{col}_ci95"] = m, ci
        rows.append(row)
    return pd.DataFrame(rows)


def main(dataset_names=None, output_dir=None, seeds=(42,), max_pipelines=20) -> None:
    if dataset_names is None:
        dataset_names = list(BENCHMARK_DATASETS.keys())
    out_dir = Path(output_dir) if output_dir else (
        Path(__file__).parents[1] / "outputs" / "paper_ready" / "c2_tfm_reward_nested")
    out_dir.mkdir(parents=True, exist_ok=True)

    actions = build_actions()
    all_pipelines = enumerate_valid_pipelines(max_len=3)

    rows: List[Dict] = []
    t0 = time.time()
    for ds in dataset_names:
        for seed in seeds:
            pipelines = sample_pipelines(all_pipelines, max_pipelines, seed)
            print(f"\n── {ds} | seed {seed} | {len(pipelines)} pipelines ──", flush=True)
            ts = time.time()
            r = run_one(ds, seed, pipelines, actions)
            if r is not None:
                rows.append(r)
                print(f"   rf  acc={r['rf_acc']:.4f} ece={r['rf_ece']:.4f}  |  "
                      f"tfm acc={r['tfm_acc']:.4f} ece={r['tfm_ece']:.4f}  "
                      f"({time.time()-ts:.0f}s)", flush=True)
            # incremental save (long CPU run — never lose progress)
            pd.DataFrame(rows).to_csv(out_dir / "results_per_seed.csv", index=False)

    if not rows:
        print("No results."); return
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results_per_seed.csv", index=False)
    agg = aggregate(df)
    agg.to_csv(out_dir / "results_aggregated.csv", index=False)

    # Paired Wilcoxon on per-dataset mean accuracy (TFM > RF)
    pivot = agg.set_index("dataset")
    rf = pivot["rf_acc_mean"].dropna(); tfm = pivot["tfm_acc_mean"].dropna()
    shared = rf.index.intersection(tfm.index)
    print(f"\n{'='*60}\nC2 NESTED (leak-free) — {len(shared)} datasets, seeds={list(seeds)}")
    if len(shared) >= 2:
        try:
            stat, p = wilcoxon(tfm[shared].values, rf[shared].values, alternative="greater")
            print(f"Wilcoxon TFM>RF (mean acc): stat={stat:.3f} p={p:.4f}")
        except Exception as exc:
            print(f"Wilcoxon failed: {exc}")
        wins = int((tfm[shared].values > rf[shared].values).sum())
        print(f"TFM wins: {wins}/{len(shared)} | "
              f"TFM mean acc={tfm[shared].mean():.4f} RF={rf[shared].mean():.4f}")
    print(f"Total wall-clock: {(time.time()-t0)/60:.1f} min")
    print(f"Saved → {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--max-pipelines", type=int, default=20)
    a = ap.parse_args()
    main(a.datasets, a.output_dir, tuple(a.seeds), a.max_pipelines)
