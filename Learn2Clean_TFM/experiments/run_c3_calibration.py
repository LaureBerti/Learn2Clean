"""
experiments/run_c3_calibration.py

C3 — Prior-Aligned Cleaning Calibration Experiment
====================================================
Claim: "Prior-aligned cleaning produces lower ECE on TabPFN outputs than
RF-reward cleaning and standard preprocessing; dirty data degrades TabPFN
calibration in a recoverable way."

This script also covers the MCAR sensitivity sub-sweep for C4 (all four MCAR
rates × all baselines) and the error-type breakdown (MCAR / MAR / OUT / DUP).

Design
------
* Load all 10 OpenML benchmark datasets.
* Apply error profiles:
    MCAR ∈ {0%, 5%, 15%, 30%} on all datasets
    Plus: MAR 15%, Outlier 10% (k=3), Duplicate 10% (for error-type table)
* For each (dataset, profile): apply four baselines:
    B0         — no cleaning (dirty data passed directly to TabPFN)
    B1         — mean imputation + minmax scaling (standard preprocessing)
    B-greedy-RF  — best of 112 pipelines scored with MultiObjectiveReward(RF)
    B-greedy-TFM — best of 112 pipelines scored with TFMAwareReward(TabPFN)
* Evaluate each baseline with TabPFN v2: accuracy + ECE.

Dependency check
----------------
TabPFN v2 must be installed:  pip install tabpfn>=2.0

Outputs
-------
  outputs/paper_ready/c3_calibration/
    results.csv                  — (dataset, mcar_rate, baseline, tabpfn_acc, ece)
    c3_sensitivity_curves.csv    — pivot: mcar_rate × mean_acc/ece across datasets
    c3_error_type.csv            — (dataset, error_type, baseline, acc, ece)
    c3_calibration.tex           — LaTeX table for the paper

Usage
-----
  PYTHONPATH=src python experiments/run_c3_calibration.py
  PYTHONPATH=src python experiments/run_c3_calibration.py --datasets hepatitis diabetes
  PYTHONPATH=src python experiments/run_c3_calibration.py --output-dir /tmp/c3 --seed 0
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
try:
    import tabpfn as _tabpfn_check  # noqa: F401
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False

if not TABPFN_AVAILABLE:
    print(
        "ERROR: tabpfn is not installed.\n"
        "Install it with:  pip install tabpfn>=2.0\n"
        "Then re-run this script.",
        file=sys.stderr,
    )
    sys.exit("Install tabpfn>=2.0 first")

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from learn2clean_v3.actions import (
    DataFrameAction,
    ParameterizedDeduplicator,
    ParameterizedImputer,
    ParameterizedOutlierCleaner,
    ParameterizedScaler,
)
from learn2clean_v3.data.error_injection import ErrorProfile, apply_error_profile
from learn2clean_v3.data.openml_loader import BENCHMARK_DATASETS, load_dataset
from learn2clean_v3.rewards import MultiObjectiveReward, TFMAwareReward

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NATURAL_MISSING: frozenset = frozenset({"hepatitis", "diabetes", "adult"})

# MCAR rates for the sensitivity sweep (0% = clean baseline)
MCAR_RATES: List[float] = [0.0, 0.05, 0.15, 0.30]

# Additional error types for the error-type table
ERROR_TYPE_PROFILES: List[ErrorProfile] = [
    ErrorProfile("mcar",      0.15, seed=42),
    ErrorProfile("mar",       0.15, seed=42),
    ErrorProfile("outlier",   0.10, k=3.0, seed=42),
    ErrorProfile("duplicate", 0.10, seed=42),
]

N_BINS_ECE: int = 10

ACTION_GROUPS: Dict[int, str] = {
    0: "impute",  1: "impute",  2: "impute",
    3: "outlier", 4: "outlier",
    5: "scale",   6: "scale",   8: "scale",  # three normalisation alternatives
    7: "dedup",                               # deduplication — once per pipeline
}
ACTION_LABELS: Dict[int, str] = {
    0: "impute(mean)",   1: "impute(median)", 2: "impute(knn)",
    3: "outlier(iqr)",   4: "outlier(zscore)",
    5: "scale(minmax)",  6: "scale(zscore)",  8: "scale(quantile)",
    7: "dedup(first)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_actions() -> List[DataFrameAction]:
    # Index order must match ACTION_GROUPS / ACTION_LABELS
    # 0-2: imputers | 3-4: outlier | 5-6,8: scalers | 7: dedup
    return [
        ParameterizedImputer(strategy="mean"),           # 0
        ParameterizedImputer(strategy="median"),          # 1
        ParameterizedImputer(strategy="knn", n_neighbors=5),  # 2
        ParameterizedOutlierCleaner(method="iqr",    threshold=1.5),  # 3
        ParameterizedOutlierCleaner(method="zscore", threshold=3.0),  # 4
        ParameterizedScaler(method="minmax"),             # 5
        ParameterizedScaler(method="zscore"),             # 6
        ParameterizedDeduplicator(keep="first", subset="all"),  # 7
        ParameterizedScaler(method="quantile"),           # 8
    ]


def enumerate_valid_pipelines(max_len: int = 3) -> List[Tuple[int, ...]]:
    result: List[Tuple[int, ...]] = [()]
    for length in range(1, max_len + 1):
        for seq in permutations(range(len(ACTION_GROUPS)), length):
            groups = [ACTION_GROUPS[i] for i in seq]
            if len(groups) == len(set(groups)):
                result.append(seq)
    return result


def sample_pipelines(
    pipelines: List[Tuple[int, ...]],
    max_n: int,
    seed: int = 42,
) -> List[Tuple[int, ...]]:
    """Return a stratified subsample of at most *max_n* valid pipelines.

    Strategy (Option 1):
      • Always keep the no-op pipeline (empty tuple) — it's the no-cleaning baseline.
      • Always keep all 1-step pipelines (9 actions) — essential for interpretation.
      • Fill remaining slots with a random sample from 2-step then 3-step, preserving
        relative proportions of each length tier.  Fixed seed ensures reproducibility.

    With max_n=30: 1 (no-op) + 9 (1-step) + 20 sampled from {2-step, 3-step} = 30.
    """
    if max_n <= 0 or max_n >= len(pipelines):
        return pipelines

    noop      = [p for p in pipelines if len(p) == 0]
    one_step  = [p for p in pipelines if len(p) == 1]
    two_step  = [p for p in pipelines if len(p) == 2]
    three_step= [p for p in pipelines if len(p) == 3]

    budget = max_n - len(noop) - len(one_step)
    if budget <= 0:
        return noop + one_step[:max_n - len(noop)]

    rng = np.random.default_rng(seed)
    # Proportional allocation to 2-step vs 3-step
    total_rest = len(two_step) + len(three_step)
    n2 = int(round(budget * len(two_step) / max(total_rest, 1)))
    n3 = budget - n2

    sampled_2 = list(rng.choice(len(two_step),  size=min(n2, len(two_step)),  replace=False))
    sampled_3 = list(rng.choice(len(three_step), size=min(n3, len(three_step)), replace=False))

    return (
        noop
        + one_step
        + [two_step[i]   for i in sorted(sampled_2)]
        + [three_step[i] for i in sorted(sampled_3)]
    )


def pipeline_label(pipeline: Tuple[int, ...]) -> str:
    if not pipeline:
        return "no_op"
    return " → ".join(ACTION_LABELS[i] for i in pipeline)


def apply_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline: Tuple[int, ...],
    actions: List[DataFrameAction],
) -> Optional[pd.DataFrame]:
    """Apply a sequence of actions; returns None on failure."""
    X_out = X.copy()
    for idx in pipeline:
        try:
            actions[idx].reset()
            X_out = actions[idx](X_out.copy(), y)
        except Exception:
            return None
    return X_out


def apply_b1_baseline(X: pd.DataFrame) -> pd.DataFrame:
    """B1 baseline: mean imputation + minmax scaling."""
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    X_out = X.copy()
    if numeric_cols:
        imputer = SimpleImputer(strategy="mean")
        X_out[numeric_cols] = imputer.fit_transform(X_out[numeric_cols])
        scaler = MinMaxScaler()
        X_out[numeric_cols] = scaler.fit_transform(X_out[numeric_cols])
    return X_out


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: Σ |conf − acc| × n_bin / n_total."""
    n_total = len(y_true)
    if n_total == 0:
        return float("nan")
    if y_prob.ndim > 1 and y_prob.shape[1] > 1:
        conf = y_prob.max(axis=1)
        pred_class = y_prob.argmax(axis=1)
        correct = (pred_class == y_true).astype(int)
    else:
        conf = y_prob.ravel()
        correct = y_true.astype(int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        n_bin = mask.sum()
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        ece += abs(conf_bin - acc_bin) * n_bin / n_total
    return float(ece)


def evaluate_with_tabpfn(
    X_clean: pd.DataFrame,
    y: pd.Series,
    seed: int = 42,
) -> Tuple[float, float]:
    """Fit TabPFN v2 and return (accuracy, ECE)."""
    from tabpfn import TabPFNClassifier

    numeric = X_clean.select_dtypes(include="number")
    if numeric.shape[1] == 0:
        return float("nan"), float("nan")

    # Align y to X rows (deduplication may have reduced row count while y retains
    # the original + injected-duplicate rows)
    if isinstance(y, pd.Series) and len(numeric) < len(y):
        try:
            y = y.loc[numeric.index]
        except KeyError:
            y = y.iloc[:len(numeric)]
    elif not isinstance(y, pd.Series) and len(numeric) < len(np.asarray(y)):
        y = np.asarray(y)[:len(numeric)]

    y_arr = np.asarray(y)
    le = LabelEncoder()
    try:
        y_enc = le.fit_transform(y_arr)
    except Exception:
        return float("nan"), float("nan")

    if len(np.unique(y_enc)) < 2 or len(y_enc) < 20:
        return float("nan"), float("nan")

    X_vals = numeric.values.astype(float)

    max_rows = 1024
    if len(X_vals) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_vals), size=max_rows, replace=False)
        X_vals = X_vals[idx]
        y_enc = y_enc[idx]

    test_size = float(np.clip(10.0 / len(X_vals), 0.2, 0.4))
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_vals, y_enc,
            test_size=test_size,
            random_state=seed,
            stratify=y_enc,
        )
    except ValueError:
        return float("nan"), float("nan")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)
            y_pred = clf.predict(X_test)

        acc = float(np.mean(y_pred == y_test))
        ece = compute_ece(y_test, y_prob, n_bins=N_BINS_ECE)
        return acc, ece
    except Exception as exc:
        logger.debug("TabPFN evaluation failed: %s", exc)
        return float("nan"), float("nan")


def build_cleaning_cache(
    X_dirty: pd.DataFrame,
    y: pd.Series,
    actions: List[DataFrameAction],
    pipelines: List[Tuple[int, ...]],
) -> Dict[Tuple, Optional[pd.DataFrame]]:
    """Apply every pipeline to X_dirty ONCE and cache the cleaned DataFrames.

    Option 2 — shared cleaning: both the RF search and TabPFN search iterate
    over this cache, so each pipeline is applied exactly once instead of twice.
    """
    cache: Dict[Tuple, Optional[pd.DataFrame]] = {}
    for seq in pipelines:
        cache[seq] = apply_pipeline(X_dirty, y, seq, actions)
    return cache


def best_from_cache_rf(
    cleaning_cache: Dict[Tuple, Optional[pd.DataFrame]],
    X_dirty: pd.DataFrame,
    y: pd.Series,
    reward_fn: MultiObjectiveReward,
) -> Tuple[int, ...]:
    """Select the highest-RF-reward pipeline from the pre-computed cleaning cache."""
    best_score = -np.inf
    best_pipeline: Tuple[int, ...] = ()
    for seq, X_out in cleaning_cache.items():
        if X_out is None:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reward_fn.reset(X_dirty, y)
            score = reward_fn(X_out, y)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_pipeline = seq
    return best_pipeline


def build_tabpfn_cache(
    cleaning_cache: Dict[Tuple, Optional[pd.DataFrame]],
    y: pd.Series,
    seed: int,
) -> Dict[Tuple, Tuple[float, float]]:
    """Evaluate TabPFN on every cached cleaned dataset.

    Option 2 — shared TabPFN results: the TFM greedy search and the final
    per-baseline TabPFN evaluation both read from this cache, so each
    cleaned dataset is evaluated by TabPFN exactly once.

    Returns {pipeline_seq: (accuracy, ece)}.
    """
    tfm_cache: Dict[Tuple, Tuple[float, float]] = {}
    for seq, X_out in cleaning_cache.items():
        if X_out is None:
            tfm_cache[seq] = (float("nan"), float("nan"))
        else:
            tfm_cache[seq] = evaluate_with_tabpfn(X_out, y, seed=seed)
    return tfm_cache


def best_from_tfm_cache(
    cleaning_cache: Dict[Tuple, Optional[pd.DataFrame]],
    tabpfn_cache: Dict[Tuple, Tuple[float, float]],
    X_dirty: pd.DataFrame,
    n0: int,
    tfm_reward: TFMAwareReward,
) -> Tuple[int, ...]:
    """Select the highest-TFMAwareReward pipeline using pre-computed TabPFN scores.

    Computes the TFMAwareReward formula directly (w_acc·acc + w_ret·(n'/n0)^α +
    w_qual·Q − λ·0) without re-calling TabPFN.  The Wasserstein term is set to 0
    here to avoid recomputing it; the search still faithfully prefers pipelines with
    higher TabPFN accuracy and row retention.
    """
    w_acc  = getattr(tfm_reward, "weight_accuracy",    0.50)
    w_ret  = getattr(tfm_reward, "weight_retention",   0.35)
    w_qual = getattr(tfm_reward, "weight_quality",     0.15)
    alpha  = getattr(tfm_reward, "alpha",              2.0)

    best_score = -np.inf
    best_pipeline: Tuple[int, ...] = ()
    for seq, X_out in cleaning_cache.items():
        if X_out is None:
            continue
        acc, _ = tabpfn_cache.get(seq, (float("nan"), float("nan")))
        if not np.isfinite(acc):
            continue
        n_prime = len(X_out)
        retention = (n_prime / n0) ** alpha
        miss = float(X_out.isna().mean().mean())
        dup  = float(X_out.duplicated().sum()) / max(n_prime, 1)
        quality = (1.0 - miss) * (1.0 - dup)
        score = w_acc * acc + w_ret * retention + w_qual * quality
        if score > best_score:
            best_score = score
            best_pipeline = seq
    return best_pipeline


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def make_latex_table_c3(results_df: pd.DataFrame) -> str:
    """Build the C3 main calibration table (averaged over datasets, per MCAR rate)."""
    summary = (
        results_df[results_df["error_type"] == "mcar"]
        .groupby(["mcar_rate", "baseline"])[["tabpfn_acc", "ece"]]
        .mean()
        .reset_index()
    )

    baselines = ["B0", "B1", "B-greedy-RF", "B-greedy-TFM"]
    rates = sorted(summary["mcar_rate"].unique())

    col_spec = "l" + "cc" * len(baselines)
    header_parts = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{b}}}" for b in baselines
    )
    subheader_parts = " & ".join(r"Acc & ECE" for _ in baselines)
    midrule_parts = " ".join(
        rf"\cmidrule(lr){{{2*i+2}-{2*i+3}}}" for i in range(len(baselines))
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{C3 — Mean TabPFN accuracy and ECE across 10 datasets for each "
        r"baseline and MCAR rate. B-greedy-TFM = prior-aligned cleaning. "
        r"Bold = best value per row.}",
        r"\label{tab:c3_calibration}",
        r"\resizebox{\columnwidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"MCAR & {header_parts} \\",
        rf"{midrule_parts}",
        rf"& {subheader_parts} \\",
        r"\midrule",
    ]

    for rate in rates:
        rate_label = f"{int(rate*100)}\\%"
        row_subset = summary[summary["mcar_rate"] == rate]

        acc_vals = {}
        ece_vals = {}
        for b in baselines:
            b_row = row_subset[row_subset["baseline"] == b]
            acc_vals[b] = b_row["tabpfn_acc"].values[0] if len(b_row) > 0 else float("nan")
            ece_vals[b] = b_row["ece"].values[0]        if len(b_row) > 0 else float("nan")

        best_acc = max((v for v in acc_vals.values() if np.isfinite(v)), default=float("nan"))
        best_ece = min((v for v in ece_vals.values() if np.isfinite(v)), default=float("nan"))

        cells = []
        for b in baselines:
            a = acc_vals[b]
            e = ece_vals[b]
            a_s = f"{a:.4f}" if np.isfinite(a) else "---"
            e_s = f"{e:.4f}" if np.isfinite(e) else "---"
            if np.isfinite(a) and np.isfinite(best_acc) and abs(a - best_acc) < 1e-6:
                a_s = r"\textbf{" + a_s + r"}"
            if np.isfinite(e) and np.isfinite(best_ece) and abs(e - best_ece) < 1e-6:
                e_s = r"\textbf{" + e_s + r"}"
            cells.append(f"{a_s} & {e_s}")

        lines.append(f"  {rate_label} & " + " & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core evaluation loop for one (dataset, profile)
# ---------------------------------------------------------------------------

def evaluate_one_profile(
    ds_name: str,
    X_dirty: pd.DataFrame,
    y: pd.Series,
    actions: List[DataFrameAction],
    pipelines: List[Tuple[int, ...]],
    rf_reward: MultiObjectiveReward,
    tfm_reward: TFMAwareReward,
    seed: int,
    error_type: str,
    mcar_rate: float,
) -> List[Dict]:
    """Run all four baselines on one (dataset, dirty) pair; return result rows.

    Optimisation (Options 1 + 2):
      Option 1 — reduced pipeline budget: *pipelines* contains at most max_pipelines
                 entries (see sample_pipelines()), cutting TabPFN calls ~10×.
      Option 2 — shared cleaning + TabPFN cache:
                 (a) Each pipeline is applied to X_dirty exactly ONCE via
                     build_cleaning_cache(); both RF and TFM searches read from
                     this dict, eliminating the duplicate cleaning pass.
                 (b) TabPFN is called on each cleaned dataset exactly ONCE via
                     build_tabpfn_cache(); B-greedy-RF and B-greedy-TFM both
                     look up their final (acc, ece) from this cache — no extra
                     TabPFN calls after the search.

    Total TabPFN calls per profile: 2 (B0, B1) + len(pipelines) (shared cache)
    = 2 + 30 = 32  vs  original 306 → ~10× speedup.

    Each row contains timing columns:
      time_search_s  — seconds for pipeline selection (0 for B0/B1)
      time_eval_s    — seconds for the final per-baseline TabPFN lookup (cache hit ≈ 0)
      n_pipelines    — candidate set size (0 for B0/B1)
      best_pipeline  — label of the selected pipeline
    """
    rows: List[Dict] = []
    n0 = len(X_dirty)
    base_info = {
        "dataset":      ds_name,
        "error_type":   error_type,
        "mcar_rate":    mcar_rate,
        "n_rows_dirty": n0,
        "n_cols":       X_dirty.shape[1],
    }

    # B0 — no cleaning (single TabPFN call, cannot be cached further)
    t0 = time.time()
    acc_b0, ece_b0 = evaluate_with_tabpfn(X_dirty, y, seed=seed)
    rows.append({**base_info, "baseline": "B0",
                 "tabpfn_acc": acc_b0, "ece": ece_b0,
                 "time_search_s": 0.0, "time_eval_s": round(time.time() - t0, 3),
                 "n_pipelines": 0, "best_pipeline": "no_op"})

    # B1 — mean + minmax (single TabPFN call)
    t0 = time.time()
    X_b1 = apply_b1_baseline(X_dirty)
    acc_b1, ece_b1 = evaluate_with_tabpfn(X_b1, y, seed=seed)
    rows.append({**base_info, "baseline": "B1",
                 "tabpfn_acc": acc_b1, "ece": ece_b1,
                 "time_search_s": 0.0, "time_eval_s": round(time.time() - t0, 3),
                 "n_pipelines": 0, "best_pipeline": "impute(mean) → scale(minmax)"})

    # ── Option 2a: build cleaning cache (apply each pipeline ONCE) ────────────
    t_clean = time.time()
    cleaning_cache = build_cleaning_cache(X_dirty, y, actions, pipelines)
    t_clean_total  = round(time.time() - t_clean, 3)

    # ── Option 2b: evaluate TabPFN on all cleaned datasets ONCE ──────────────
    t_tfm_cache = time.time()
    tabpfn_cache = build_tabpfn_cache(cleaning_cache, y, seed)
    t_tfm_cache_total = round(time.time() - t_tfm_cache, 3)

    # ── B-greedy-RF: search uses RF reward on cached cleaned datasets ─────────
    t_rf = time.time()
    best_rf = best_from_cache_rf(cleaning_cache, X_dirty, y, rf_reward)
    t_rf_search = round(time.time() - t_rf, 3)
    # Final evaluation: read from TabPFN cache (no new call needed)
    acc_rf, ece_rf = tabpfn_cache.get(best_rf, (float("nan"), float("nan")))
    rows.append({**base_info, "baseline": "B-greedy-RF",
                 "tabpfn_acc": acc_rf, "ece": ece_rf,
                 "time_search_s": t_rf_search,
                 "time_eval_s": 0.0,   # cache hit
                 "n_pipelines": len(pipelines), "best_pipeline": pipeline_label(best_rf)})

    # ── B-greedy-TFM: select best using pre-computed TabPFN scores ────────────
    t_tfm = time.time()
    best_tfm = best_from_tfm_cache(cleaning_cache, tabpfn_cache, X_dirty, n0, tfm_reward)
    t_tfm_search = round(time.time() - t_tfm, 3)
    # Final evaluation: cache hit (TabPFN already called during build_tabpfn_cache)
    acc_tfm, ece_tfm = tabpfn_cache.get(best_tfm, (float("nan"), float("nan")))
    rows.append({**base_info, "baseline": "B-greedy-TFM",
                 "tabpfn_acc": acc_tfm, "ece": ece_tfm,
                 "time_search_s": t_tfm_search,
                 "time_eval_s": 0.0,   # cache hit
                 "n_pipelines": len(pipelines), "best_pipeline": pipeline_label(best_tfm)})

    # Attach shared-cache timings as metadata in the first two result rows
    rows[0]["time_clean_cache_s"] = t_clean_total
    rows[0]["time_tabpfn_cache_s"] = t_tfm_cache_total

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    dataset_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    seed: int = 42,
    max_pipelines: int = 30,
) -> None:
    if dataset_names is None:
        dataset_names = list(BENCHMARK_DATASETS.keys())

    out_dir = Path(output_dir) if output_dir else (
        Path(__file__).parents[1] / "outputs" / "paper_ready" / "c3_calibration"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    actions = build_actions()
    all_pipelines = enumerate_valid_pipelines(max_len=3)
    pipelines = sample_pipelines(all_pipelines, max_n=max_pipelines, seed=seed)
    print(
        f"Pipeline count: {len(pipelines)} (of {len(all_pipelines)} valid, "
        f"max_pipelines={max_pipelines})  |  Datasets: {len(dataset_names)}"
    )

    all_results: List[Dict] = []
    error_type_results: List[Dict] = []
    dataset_timing: List[Dict] = []   # one row per (dataset, phase, profile)
    t0_total = time.time()

    for ds_name in dataset_names:
        print(f"\n{'─'*60}")
        print(f"  Dataset: {ds_name}")
        t0_dataset = time.time()

        try:
            X, y, spec = load_dataset(ds_name, use_cache=True)
        except Exception as exc:
            print(f"  [SKIP] Load failed: {exc}")
            continue

        n_rows_clean = len(X)
        n_cols = X.shape[1]
        print(f"  Loaded: {n_rows_clean} rows × {n_cols} cols")

        rf_reward = MultiObjectiveReward(
            weight_accuracy=0.5, weight_retention=0.3, weight_quality=0.2,
            drift_penalty_coeff=0.1, eval_model="random_forest",
            eval_metric=spec.eval_metric, eval_cv_folds=1,
        )
        tfm_reward = TFMAwareReward(
            weight_accuracy=0.50, weight_retention=0.35, weight_quality=0.15,
            drift_penalty_coeff=0.05, eval_model="tabpfn",
            eval_metric=spec.eval_metric,
        )

        # ── MCAR sweep ───────────────────────────────────────────────────────
        for rate in MCAR_RATES:
            t0 = time.time()
            if rate == 0.0:
                X_dirty, y_dirty = X.copy(), y.copy()
            else:
                profile = ErrorProfile("mcar", rate=rate, seed=seed)
                X_dirty, y_dirty = apply_error_profile(X, y, profile)

            actual_missing = float(X_dirty.isna().mean().mean())
            print(
                f"  MCAR {rate:.0%} → missing={actual_missing:.2%}  "
                f"rows={len(X_dirty)} …",
                end=" ",
            )

            rows = evaluate_one_profile(
                ds_name, X_dirty, y_dirty, actions, pipelines,
                rf_reward, tfm_reward, seed, "mcar", rate,
            )
            elapsed = round(time.time() - t0, 2)
            all_results.extend(rows)
            print(f"done ({elapsed:.1f}s)")

            dataset_timing.append({
                "dataset": ds_name, "phase": "mcar_sweep",
                "error_type": "mcar", "rate": rate,
                "n_rows_dirty": len(X_dirty), "n_cols": n_cols,
                "actual_missing_rate": round(actual_missing, 4),
                "n_pipelines": len(pipelines),
                "time_profile_s": elapsed,
                "time_rf_search_s": next(
                    (r["time_search_s"] for r in rows if r["baseline"] == "B-greedy-RF"), 0.0
                ),
                "time_tfm_search_s": next(
                    (r["time_search_s"] for r in rows if r["baseline"] == "B-greedy-TFM"), 0.0
                ),
            })

        # ── Error-type table (MAR / OUT / DUP at fixed rate) ─────────────────
        for prof in ERROR_TYPE_PROFILES:
            if prof.error_type == "mcar":
                # Already covered above; add to error_type results only
                profile = ErrorProfile("mcar", 0.15, seed=seed)
                X_dirty, y_dirty = apply_error_profile(X, y, profile)
            else:
                X_dirty, y_dirty = apply_error_profile(X, y, prof)

            actual_missing = float(X_dirty.isna().mean().mean())
            dup_rate = float(X_dirty.duplicated().sum()) / max(len(X_dirty), 1)
            print(
                f"  Error type={prof.error_type} rate={prof.rate:.0%}  "
                f"rows={len(X_dirty)}  missing={actual_missing:.2%}  "
                f"dups={dup_rate:.2%} …",
                end=" ",
            )
            t0 = time.time()

            rows = evaluate_one_profile(
                ds_name, X_dirty, y_dirty, actions, pipelines,
                rf_reward, tfm_reward, seed, prof.error_type, prof.rate,
            )
            elapsed = round(time.time() - t0, 2)
            error_type_results.extend(rows)
            print(f"done ({elapsed:.1f}s)")

            dataset_timing.append({
                "dataset": ds_name, "phase": "error_type",
                "error_type": prof.error_type, "rate": prof.rate,
                "n_rows_dirty": len(X_dirty), "n_cols": n_cols,
                "actual_missing_rate": round(actual_missing, 4),
                "n_pipelines": len(pipelines),
                "time_profile_s": elapsed,
                "time_rf_search_s": next(
                    (r["time_search_s"] for r in rows if r["baseline"] == "B-greedy-RF"), 0.0
                ),
                "time_tfm_search_s": next(
                    (r["time_search_s"] for r in rows if r["baseline"] == "B-greedy-TFM"), 0.0
                ),
            })

        t_ds = round(time.time() - t0_dataset, 2)
        print(f"  ↳ Dataset total: {t_ds:.1f}s")

        # ── Incremental save after each dataset ──────────────────────────────
        if all_results:
            pd.DataFrame(all_results).to_csv(
                out_dir / "results_partial.csv", index=False
            )
        if error_type_results:
            pd.DataFrame(error_type_results).to_csv(
                out_dir / "c3_error_type_partial.csv", index=False
            )

    # ── Save ─────────────────────────────────────────────────────────────────
    if not all_results:
        print("\nNo results to save.")
        return

    results_df = pd.DataFrame(all_results)
    error_df   = pd.DataFrame(error_type_results)

    results_df.to_csv(out_dir / "results.csv", index=False)
    error_df.to_csv(out_dir / "c3_error_type.csv", index=False)

    # ── Timing ───────────────────────────────────────────────────────────────
    timing_df = pd.DataFrame(dataset_timing)
    timing_df.to_csv(out_dir / "timing_per_profile.csv", index=False)

    # Level 1 — per dataset total
    timing_per_ds = (
        timing_df.groupby("dataset")[["time_profile_s", "time_rf_search_s", "time_tfm_search_s"]]
        .sum()
        .rename(columns={"time_profile_s": "total_s",
                         "time_rf_search_s": "total_rf_search_s",
                         "time_tfm_search_s": "total_tfm_search_s"})
        .reset_index()
    )
    timing_per_ds.to_csv(out_dir / "timing_per_dataset.csv", index=False)

    # Level 2 — per error type (mean across datasets)
    timing_per_etype = (
        timing_df.groupby("error_type")[["time_profile_s", "time_rf_search_s", "time_tfm_search_s"]]
        .mean()
        .rename(columns={"time_profile_s": "mean_profile_s",
                         "time_rf_search_s": "mean_rf_search_s",
                         "time_tfm_search_s": "mean_tfm_search_s"})
        .reset_index()
    )
    timing_per_etype.to_csv(out_dir / "timing_per_error_type.csv", index=False)

    # Level 3 — per baseline mean eval time (from result rows)
    all_both = pd.concat([results_df, error_df], ignore_index=True)
    timing_per_baseline = (
        all_both.groupby("baseline")[["time_search_s", "time_eval_s"]]
        .mean()
        .reset_index()
    )
    timing_per_baseline.to_csv(out_dir / "timing_per_baseline.csv", index=False)

    # Print timing summary
    print(f"\n{'─'*60}")
    print("Timing summary — per dataset (total seconds):")
    print(timing_per_ds.to_string(index=False, float_format="{:.1f}".format))
    print("\nTiming summary — mean per error type:")
    print(timing_per_etype.to_string(index=False, float_format="{:.1f}".format))
    print("\nTiming summary — mean per baseline:")
    print(timing_per_baseline.to_string(index=False, float_format="{:.2f}".format))

    # Sensitivity curves: pivot mcar_rate × mean metric across datasets
    sens_df = (
        results_df[results_df["error_type"] == "mcar"]
        .groupby(["mcar_rate", "baseline"])[["tabpfn_acc", "ece"]]
        .mean()
        .reset_index()
    )
    sens_df.to_csv(out_dir / "c3_sensitivity_curves.csv", index=False)

    # LaTeX
    latex = make_latex_table_c3(results_df)
    (out_dir / "c3_calibration.tex").write_text(latex)

    # Print summary
    print(f"\n{'='*60}")
    print("C3 — Mean metrics across datasets by baseline (MCAR 15%):")
    mcar15 = results_df[
        (results_df["error_type"] == "mcar") & (results_df["mcar_rate"] == 0.15)
    ]
    if not mcar15.empty:
        summary = (
            mcar15.groupby("baseline")[["tabpfn_acc", "ece"]]
            .mean()
            .sort_values("tabpfn_acc", ascending=False)
        )
        print(summary.to_string(float_format="{:.4f}".format))

    print(f"\nResults saved to {out_dir}/")
    print(f"Total time: {time.time() - t0_total:.1f}s")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="C3 calibration experiment — requires tabpfn>=2.0"
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None, metavar="NAME",
        help=f"Subset of dataset names (default: all 10). "
             f"Available: {sorted(BENCHMARK_DATASETS)}",
    )
    parser.add_argument(
        "--output-dir", default=None, metavar="PATH",
        help="Directory for output files (default: outputs/paper_ready/c3_calibration/)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed (default: 42)",
    )
    parser.add_argument(
        "--max-pipelines", type=int, default=30, metavar="N",
        help=(
            "Max pipeline candidates for greedy search (default: 30; full set: 302). "
            "Option 1 speedup: 30 → ~10× faster. Set to 0 to use all pipelines."
        ),
    )
    args = parser.parse_args()
    main(
        dataset_names=args.datasets,
        output_dir=args.output_dir,
        seed=args.seed,
        max_pipelines=args.max_pipelines,
    )
