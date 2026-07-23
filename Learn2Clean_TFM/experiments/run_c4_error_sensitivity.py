"""
experiments/run_c4_error_sensitivity.py

C4 — Error Sensitivity Sweep
==============================
Claim: "Accuracy and calibration benefit of prior-aligned cleaning over B1 grows
monotonically with MCAR rate 5%→30%."

Design
------
* Use 5 representative datasets:
    D1 hepatitis, D3 ionosphere, D5 diabetes, D7 kr_vs_kp, D9 adult
* Sweep MCAR ∈ {0%, 5%, 10%, 15%, 20%, 30%} on each dataset.
* For each (dataset, rate): compute TabPFN accuracy and ECE for:
    B0           — no cleaning
    B1           — mean imputation + minmax scaling
    B-greedy-TFM — best of 112 pipelines scored with TFMAwareReward(TabPFN)
* Compute "benefit" = B-greedy-TFM acc − B1 acc; check monotonicity across rates.
* Monotonicity criterion: Spearman ρ(rate, benefit) > 0.8 on ≥4/5 datasets.

Dependency check
----------------
TabPFN v2 must be installed:  pip install tabpfn>=2.0

Outputs
-------
  outputs/paper_ready/c4_error_sensitivity/
    results.csv          — (dataset, mcar_rate, baseline, tabpfn_acc, ece)
    c4_sensitivity.tex   — LaTeX table for the paper

Usage
-----
  PYTHONPATH=src python experiments/run_c4_error_sensitivity.py
  PYTHONPATH=src python experiments/run_c4_error_sensitivity.py --output-dir /tmp/c4
  PYTHONPATH=src python experiments/run_c4_error_sensitivity.py --seed 0
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
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

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
from learn2clean_v3.rewards import TFMAwareReward

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Five representative datasets (D1/D3/D5/D7/D9)
REPRESENTATIVE_DATASETS: List[str] = [
    "hepatitis",    # D1 — XS, natural missing
    "ionosphere",   # D3 — XS, no natural missing
    "diabetes",     # D5 — S,  natural missing (Pima zeros)
    "kr_vs_kp",     # D7 — M,  no natural missing
    "adult",        # D9 — L,  natural missing
]

MCAR_RATES: List[float] = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]

N_BINS_ECE: int = 10

ACTION_GROUPS: Dict[int, str] = {
    0: "impute",  1: "impute",  2: "impute",
    3: "outlier", 4: "outlier",
    5: "scale",   6: "scale",   8: "scale",  # three normalisation alternatives
    7: "dedup",
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
    X_out = X.copy()
    for idx in pipeline:
        try:
            actions[idx].reset()
            X_out = actions[idx](X_out.copy(), y)
        except Exception:
            return None
    return X_out


def apply_b1_baseline(X: pd.DataFrame) -> pd.DataFrame:
    """B1: mean imputation + minmax scaling on numeric columns."""
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
        ece += abs(conf[mask].mean() - correct[mask].mean()) * n_bin / n_total
    return float(ece)


def evaluate_with_tabpfn(
    X_clean: pd.DataFrame,
    y: pd.Series,
    seed: int = 42,
) -> Tuple[float, float]:
    """Fit TabPFN v2 on X_clean; return (accuracy, ECE)."""
    from tabpfn import TabPFNClassifier

    numeric = X_clean.select_dtypes(include="number")
    if numeric.shape[1] == 0:
        return float("nan"), float("nan")

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


def sample_pipelines(
    pipelines: List[Tuple[int, ...]],
    max_n: int,
    seed: int = 42,
) -> List[Tuple[int, ...]]:
    """Stratified subsample: always keep no-op + all 1-step; fill rest proportionally."""
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
    total_rest = len(two_step) + len(three_step)
    n2 = int(round(budget * len(two_step) / max(total_rest, 1)))
    n3 = budget - n2

    sampled_2 = list(rng.choice(len(two_step),   size=min(n2, len(two_step)),   replace=False))
    sampled_3 = list(rng.choice(len(three_step),  size=min(n3, len(three_step)), replace=False))

    return (
        noop
        + one_step
        + [two_step[i]    for i in sorted(sampled_2)]
        + [three_step[i]  for i in sorted(sampled_3)]
    )


def build_cleaning_cache(
    X_dirty: pd.DataFrame,
    y: pd.Series,
    actions: List[DataFrameAction],
    pipelines: List[Tuple[int, ...]],
) -> Dict[Tuple, Optional[pd.DataFrame]]:
    """Apply every pipeline to X_dirty ONCE and cache the results."""
    return {seq: apply_pipeline(X_dirty, y, seq, actions) for seq in pipelines}


def build_tabpfn_cache(
    cleaning_cache: Dict[Tuple, Optional[pd.DataFrame]],
    y: pd.Series,
    seed: int,
) -> Dict[Tuple, Tuple[float, float]]:
    """Call TabPFN on each cached cleaned dataset exactly once."""
    result: Dict[Tuple, Tuple[float, float]] = {}
    for seq, X_out in cleaning_cache.items():
        if X_out is None:
            result[seq] = (float("nan"), float("nan"))
        else:
            result[seq] = evaluate_with_tabpfn(X_out, y, seed=seed)
    return result


def best_from_tfm_cache(
    cleaning_cache: Dict[Tuple, Optional[pd.DataFrame]],
    tabpfn_cache: Dict[Tuple, Tuple[float, float]],
    X_dirty: pd.DataFrame,
    n0: int,
    tfm_reward: TFMAwareReward,
) -> Tuple[int, ...]:
    """Select best pipeline using pre-computed TabPFN scores (no new TabPFN calls)."""
    w_acc  = getattr(tfm_reward, "weight_accuracy",  0.50)
    w_ret  = getattr(tfm_reward, "weight_retention", 0.35)
    w_qual = getattr(tfm_reward, "weight_quality",   0.15)
    alpha  = getattr(tfm_reward, "alpha",             2.0)

    best_score = -np.inf
    best_pipeline: Tuple[int, ...] = ()
    for seq, X_out in cleaning_cache.items():
        if X_out is None:
            continue
        acc, _ = tabpfn_cache.get(seq, (float("nan"), float("nan")))
        if not np.isfinite(acc):
            continue
        n_prime   = len(X_out)
        retention = (n_prime / n0) ** alpha
        miss      = float(X_out.isna().mean().mean())
        dup       = float(X_out.duplicated().sum()) / max(n_prime, 1)
        quality   = (1.0 - miss) * (1.0 - dup)
        score = w_acc * acc + w_ret * retention + w_qual * quality
        if score > best_score:
            best_score = score
            best_pipeline = seq
    return best_pipeline


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def make_latex_table_c4(results_df: pd.DataFrame) -> str:
    """C4 LaTeX table: per-dataset rows, columns = (rate × baseline acc+ece)."""
    rates = sorted(results_df["mcar_rate"].unique())
    baselines = ["B0", "B1", "B-greedy-TFM"]

    # Pivot: one row per (dataset, rate), columns = baselines
    pivot = results_df.pivot_table(
        index=["dataset", "mcar_rate"],
        columns="baseline",
        values="tabpfn_acc",
    ).reset_index()

    # Benefit column
    if "B1" in pivot.columns and "B-greedy-TFM" in pivot.columns:
        pivot["benefit"] = pivot["B-greedy-TFM"] - pivot["B1"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{C4 — TabPFN accuracy across MCAR rates for B0, B1, and "
        r"B-greedy-TFM. Benefit = TFM acc $-$ B1 acc. "
        r"Monotonically increasing benefit validates C4.}",
        r"\label{tab:c4_sensitivity}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Dataset & MCAR & B0 & B1 & B-greedy-TFM & Benefit \\",
        r"\midrule",
    ]

    for ds in sorted(results_df["dataset"].unique()):
        ds_tex = ds.replace("_", r"\_")
        ds_pivot = pivot[pivot["dataset"] == ds].sort_values("mcar_rate")
        first = True
        for _, row in ds_pivot.iterrows():
            rate_label = f"{int(row['mcar_rate']*100)}\\%"
            b0  = f"{row.get('B0', float('nan')):.4f}"          if np.isfinite(row.get("B0", float("nan")))          else "---"
            b1  = f"{row.get('B1', float('nan')):.4f}"          if np.isfinite(row.get("B1", float("nan")))          else "---"
            tfm = f"{row.get('B-greedy-TFM', float('nan')):.4f}" if np.isfinite(row.get("B-greedy-TFM", float("nan"))) else "---"
            ben = f"{row.get('benefit', float('nan')):.4f}"      if np.isfinite(row.get("benefit", float("nan")))      else "---"
            ds_col = ds_tex if first else ""
            lines.append(f"  {ds_col} & {rate_label} & {b0} & {b1} & {tfm} & {ben} \\\\")
            first = False
        lines.append(r"  \addlinespace")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    output_dir: Optional[str] = None,
    seed: int = 42,
    max_pipelines: int = 20,
) -> None:
    out_dir = Path(output_dir) if output_dir else (
        Path(__file__).parents[1] / "outputs" / "paper_ready" / "c4_error_sensitivity"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    actions = build_actions()
    all_pipelines = enumerate_valid_pipelines(max_len=3)
    pipelines = sample_pipelines(all_pipelines, max_n=max_pipelines, seed=seed)
    print(
        f"Pipeline count: {len(pipelines)} (of {len(all_pipelines)} valid, "
        f"max_pipelines={max_pipelines})"
    )
    print(f"Datasets: {REPRESENTATIVE_DATASETS}")
    print(f"MCAR rates: {MCAR_RATES}")

    all_results: List[Dict] = []
    dataset_timing: List[Dict] = []   # one row per (dataset, mcar_rate)
    t0_total = time.time()

    for ds_name in REPRESENTATIVE_DATASETS:
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

        for rate in MCAR_RATES:
            t0 = time.time()

            if rate == 0.0:
                X_dirty, y_dirty = X.copy(), y.copy()
            else:
                profile = ErrorProfile("mcar", rate=rate, seed=seed)
                X_dirty, y_dirty = apply_error_profile(X, y, profile)

            miss_pct = float(X_dirty.isna().mean().mean())
            print(
                f"  rate={rate:.0%}  missing={miss_pct:.2%}  rows={len(X_dirty)} … ",
                end="", flush=True,
            )

            base_info = {
                "dataset": ds_name, "mcar_rate": rate,
                "n_rows_dirty": len(X_dirty), "n_cols": n_cols,
                "actual_missing_rate": round(miss_pct, 4),
            }

            # B0 — no cleaning
            t_b0 = time.time()
            acc_b0, ece_b0 = evaluate_with_tabpfn(X_dirty, y_dirty, seed=seed)
            t_b0 = round(time.time() - t_b0, 3)
            all_results.append({**base_info, "baseline": "B0",
                                 "tabpfn_acc": acc_b0, "ece": ece_b0,
                                 "time_search_s": 0.0, "time_eval_s": t_b0,
                                 "n_pipelines": 0, "best_pipeline": "no_op"})

            # B1 — mean + minmax
            t_b1 = time.time()
            X_b1 = apply_b1_baseline(X_dirty)
            acc_b1, ece_b1 = evaluate_with_tabpfn(X_b1, y_dirty, seed=seed)
            t_b1 = round(time.time() - t_b1, 3)
            all_results.append({**base_info, "baseline": "B1",
                                 "tabpfn_acc": acc_b1, "ece": ece_b1,
                                 "time_search_s": 0.0, "time_eval_s": t_b1,
                                 "n_pipelines": 0,
                                 "best_pipeline": "impute(mean) → scale(minmax)"})

            # B-greedy-TFM — Option 2: shared cleaning + TabPFN cache
            tfm_reward = TFMAwareReward(
                weight_accuracy=0.50, weight_retention=0.35, weight_quality=0.15,
                drift_penalty_coeff=0.05, eval_model="tabpfn",
                eval_metric=spec.eval_metric,
            )
            t_search = time.time()
            cleaning_cache = build_cleaning_cache(X_dirty, y_dirty, actions, pipelines)
            tabpfn_cache   = build_tabpfn_cache(cleaning_cache, y_dirty, seed)
            best_tfm = best_from_tfm_cache(
                cleaning_cache, tabpfn_cache, X_dirty, len(X_dirty), tfm_reward
            )
            t_tfm_search = round(time.time() - t_search, 3)
            # Final result: cache lookup (no extra TabPFN call)
            acc_tfm, ece_tfm = tabpfn_cache.get(best_tfm, (float("nan"), float("nan")))
            t_tfm_eval = 0.0   # cache hit
            all_results.append({**base_info, "baseline": "B-greedy-TFM",
                                 "tabpfn_acc": acc_tfm, "ece": ece_tfm,
                                 "time_search_s": t_tfm_search, "time_eval_s": t_tfm_eval,
                                 "n_pipelines": len(pipelines),
                                 "best_pipeline": pipeline_label(best_tfm)})

            elapsed = round(time.time() - t0, 2)
            print(
                f"B0={acc_b0:.3f} B1={acc_b1:.3f} TFM={acc_tfm:.3f} "
                f"benefit={acc_tfm - acc_b1:+.3f}  ({elapsed:.1f}s)"
            )

            dataset_timing.append({
                "dataset": ds_name, "mcar_rate": rate,
                "n_rows_dirty": len(X_dirty), "n_cols": n_cols,
                "actual_missing_rate": round(miss_pct, 4),
                "n_pipelines": len(pipelines),
                "time_b0_eval_s": t_b0,
                "time_b1_eval_s": t_b1,
                "time_tfm_search_s": t_tfm_search,
                "time_tfm_eval_s": t_tfm_eval,
                "time_profile_s": elapsed,
            })

        t_ds = round(time.time() - t0_dataset, 2)
        print(f"  ↳ Dataset total: {t_ds:.1f}s")

    # ── Save ─────────────────────────────────────────────────────────────────
    if not all_results:
        print("\nNo results to save.")
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_dir / "results.csv", index=False)

    # ── Timing ───────────────────────────────────────────────────────────────
    timing_df = pd.DataFrame(dataset_timing)
    timing_df.to_csv(out_dir / "timing_per_profile.csv", index=False)

    # Level 1 — per dataset total
    timing_per_ds = (
        timing_df.groupby("dataset")[["time_profile_s", "time_tfm_search_s"]]
        .agg(total_s=("time_profile_s", "sum"),
             total_tfm_search_s=("time_tfm_search_s", "sum"))
        .reset_index()
    )
    timing_per_ds.to_csv(out_dir / "timing_per_dataset.csv", index=False)

    # Level 2 — per mcar rate (mean across datasets)
    timing_per_rate = (
        timing_df.groupby("mcar_rate")[[
            "time_b0_eval_s", "time_b1_eval_s",
            "time_tfm_search_s", "time_tfm_eval_s", "time_profile_s"
        ]].mean().reset_index()
    )
    timing_per_rate.to_csv(out_dir / "timing_per_mcar_rate.csv", index=False)

    print(f"\n{'─'*60}")
    print("Timing summary — per dataset (total seconds):")
    print(timing_per_ds.to_string(index=False, float_format="{:.1f}".format))
    print("\nTiming summary — mean per MCAR rate:")
    print(timing_per_rate.to_string(index=False, float_format="{:.2f}".format))

    # Monotonicity analysis
    print(f"\n{'='*60}")
    print("C4 — Monotonicity of benefit (B-greedy-TFM acc − B1 acc) vs. MCAR rate:")
    pivot = results_df.pivot_table(
        index=["dataset", "mcar_rate"], columns="baseline", values="tabpfn_acc"
    ).reset_index()
    if "B1" in pivot.columns and "B-greedy-TFM" in pivot.columns:
        pivot["benefit"] = pivot["B-greedy-TFM"] - pivot["B1"]
        n_monotone = 0
        for ds_name in REPRESENTATIVE_DATASETS:
            ds_subset = pivot[pivot["dataset"] == ds_name].sort_values("mcar_rate")
            rates_  = ds_subset["mcar_rate"].values
            benefit = ds_subset["benefit"].dropna().values
            rates_aligned = rates_[:len(benefit)]
            if len(benefit) >= 3:
                rho, pval = spearmanr(rates_aligned, benefit)
                monotone = rho > 0.8
                if monotone:
                    n_monotone += 1
                print(f"  {ds_name:20s}  Spearman ρ={rho:.3f}  p={pval:.3f}  monotone={monotone}")
            else:
                print(f"  {ds_name:20s}  Not enough data points")
        print(f"\n  Monotone on {n_monotone}/{len(REPRESENTATIVE_DATASETS)} datasets "
              f"(target: ≥4/5)")

    # LaTeX table
    latex = make_latex_table_c4(results_df)
    (out_dir / "c4_sensitivity.tex").write_text(latex)

    print(f"\nResults saved to {out_dir}/")
    print(f"Total time: {time.time() - t0_total:.1f}s")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="C4 error sensitivity sweep — requires tabpfn>=2.0"
    )
    parser.add_argument(
        "--output-dir", default=None, metavar="PATH",
        help="Directory for output files (default: outputs/paper_ready/c4_error_sensitivity/)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed (default: 42)",
    )
    parser.add_argument(
        "--max-pipelines", type=int, default=20, metavar="N",
        help=(
            "Max pipeline candidates for greedy search (default: 20; full set: 302). "
            "Set to 0 to use all pipelines."
        ),
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir, seed=args.seed, max_pipelines=args.max_pipelines)
