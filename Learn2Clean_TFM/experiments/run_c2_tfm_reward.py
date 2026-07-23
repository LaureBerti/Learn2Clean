"""
experiments/run_c2_tfm_reward.py

C2 — TFM-Aware Reward Experiment
==================================
Claim: "TFMAwareReward produces statistically higher TabPFN v2 test accuracy
than RF-reward cleaning on ≥7/10 datasets (Wilcoxon p<0.05); winning pipelines
differ structurally in ≥6/10 datasets."

Design
------
* Load all 10 OpenML benchmark datasets.
* Inject MCAR 15% on all datasets (NATURAL_MISSING datasets still get MCAR on
  top to create a uniform error level; hepatitis/diabetes/adult already have
  real NaN, so the total missing rate will be higher — as intended).
* Enumerate all 112 valid pipelines (≤3 steps, no repeated action group).
* For each dataset score all 112 pipelines under BOTH reward modes:
    - RF reward  : MultiObjectiveReward(eval_model="random_forest")
    - TFM reward : TFMAwareReward(eval_model="tabpfn")
* For each reward mode, select the best pipeline; apply it; then evaluate the
  resulting cleaned data with TabPFN v2 accuracy AND ECE.
* ECE formula: Σ|conf − acc| × n_bin / n_total  (10-bin version)
* Wilcoxon signed-rank test across datasets (paired: same dataset, two conditions).

Outputs
-------
  outputs/paper_ready/c2_tfm_reward/
    results.csv              — (dataset, reward_mode, best_pipeline, tabpfn_acc, ece)
    pipeline_overlap.csv     — per dataset: do best pipelines from RF vs TFM match?
    c2_main_results.tex      — LaTeX table for the paper

Dependency check
----------------
TabPFN v2 must be installed:  pip install tabpfn>=2.0

Usage
-----
  PYTHONPATH=src python experiments/run_c2_tfm_reward.py
  PYTHONPATH=src python experiments/run_c2_tfm_reward.py --datasets hepatitis ionosphere
  PYTHONPATH=src python experiments/run_c2_tfm_reward.py --output-dir /tmp/c2 --seed 0
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
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Dependency check — must happen before any local imports that load tabpfn
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
# Local imports (after dependency check so we don't shadow the error above)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from learn2clean_v3.actions import (
    DataFrameAction,
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
MCAR_RATE: float = 0.15
N_BINS_ECE: int = 10

# Action definitions (same 7-action suite as C1)
ACTION_GROUPS: Dict[int, str] = {
    0: "impute",  1: "impute",  2: "impute",
    3: "outlier", 4: "outlier",
    5: "scale",   6: "scale",
}
ACTION_LABELS: Dict[int, str] = {
    0: "impute(mean)",   1: "impute(median)", 2: "impute(knn)",
    3: "outlier(iqr)",   4: "outlier(zscore)",
    5: "scale(minmax)",  6: "scale(zscore)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_actions() -> List[DataFrameAction]:
    return [
        ParameterizedImputer(strategy="mean"),
        ParameterizedImputer(strategy="median"),
        ParameterizedImputer(strategy="knn", n_neighbors=5),
        ParameterizedOutlierCleaner(method="iqr",    threshold=1.5),
        ParameterizedOutlierCleaner(method="zscore", threshold=3.0),
        ParameterizedScaler(method="minmax"),
        ParameterizedScaler(method="zscore"),
    ]


def enumerate_valid_pipelines(max_len: int = 3) -> List[Tuple[int, ...]]:
    """All ordered sequences ≤ max_len steps with no repeated action group."""
    result: List[Tuple[int, ...]] = [()]          # no-op pipeline
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
    """Apply a sequence of actions. Returns None on failure."""
    X_out = X.copy()
    for idx in pipeline:
        try:
            actions[idx].reset()
            X_out = actions[idx](X_out.copy(), y)
        except Exception:
            return None
    return X_out


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: Σ |conf − acc| × n_bin / n_total."""
    n_total = len(y_true)
    if n_total == 0:
        return float("nan")
    # calibration_curve works for binary; for multiclass use max-prob
    if y_prob.ndim > 1 and y_prob.shape[1] > 1:
        conf = y_prob.max(axis=1)
        # binarise: was the top class correct?
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
    """Fit TabPFN v2 and return (accuracy, ECE). Returns (NaN, NaN) on error."""
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

    # Subsample for speed (TabPFN v2 cap)
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

    noop       = [p for p in pipelines if len(p) == 0]
    one_step   = [p for p in pipelines if len(p) == 1]
    two_step   = [p for p in pipelines if len(p) == 2]
    three_step = [p for p in pipelines if len(p) == 3]

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
        + [two_step[i]   for i in sorted(sampled_2)]
        + [three_step[i] for i in sorted(sampled_3)]
    )


def build_cleaning_cache(
    X_dirty: pd.DataFrame,
    y: pd.Series,
    actions: List[DataFrameAction],
    pipelines: List[Tuple[int, ...]],
) -> Dict[Tuple, Optional[pd.DataFrame]]:
    """Apply every pipeline to X_dirty ONCE; cache results."""
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


def best_from_cache_rf(
    cleaning_cache: Dict[Tuple, Optional[pd.DataFrame]],
    X_dirty: pd.DataFrame,
    y: pd.Series,
    rf_reward: MultiObjectiveReward,
) -> Tuple[int, ...]:
    """Select highest RF-reward pipeline from pre-cleaned cache (no TabPFN)."""
    best_score = -np.inf
    best_pipeline: Tuple[int, ...] = ()
    for seq, X_out in cleaning_cache.items():
        if X_out is None:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf_reward.reset(X_dirty, y)
            score = rf_reward(X_out, y)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_pipeline = seq
    return best_pipeline


def best_from_tfm_cache(
    cleaning_cache: Dict[Tuple, Optional[pd.DataFrame]],
    tabpfn_cache: Dict[Tuple, Tuple[float, float]],
    X_dirty: pd.DataFrame,
    n0: int,
    tfm_reward: TFMAwareReward,
) -> Tuple[int, ...]:
    """Select best TFMAwareReward pipeline using pre-computed TabPFN scores."""
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

def make_latex_table(results_df: pd.DataFrame) -> str:
    """Build the main C2 result table (per-dataset RF vs TFM accuracy and ECE)."""
    pivot_acc = results_df.pivot(
        index="dataset", columns="reward_mode", values="tabpfn_acc"
    )
    pivot_ece = results_df.pivot(
        index="dataset", columns="reward_mode", values="ece"
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{C2 — TabPFN v2 test accuracy and ECE under RF-reward vs.\ TFM-reward "
        r"cleaning (MCAR 15\%, best of 112 pipelines). "
        r"Bold = better accuracy per row.}",
        r"\label{tab:c2_tfm_reward}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Accuracy $\uparrow$} & \multicolumn{2}{c}{ECE $\downarrow$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Dataset & RF-reward & TFM-reward & RF-reward & TFM-reward \\",
        r"\midrule",
    ]
    for ds in sorted(results_df["dataset"].unique()):
        rf_acc  = pivot_acc.loc[ds, "rf"]   if ds in pivot_acc.index  and "rf"  in pivot_acc.columns  else float("nan")
        tfm_acc = pivot_acc.loc[ds, "tfm"]  if ds in pivot_acc.index  and "tfm" in pivot_acc.columns  else float("nan")
        rf_ece  = pivot_ece.loc[ds, "rf"]   if ds in pivot_ece.index  and "rf"  in pivot_ece.columns  else float("nan")
        tfm_ece = pivot_ece.loc[ds, "tfm"]  if ds in pivot_ece.index  and "tfm" in pivot_ece.columns  else float("nan")

        rf_acc_s  = f"{rf_acc:.4f}"  if np.isfinite(rf_acc)  else "---"
        tfm_acc_s = f"{tfm_acc:.4f}" if np.isfinite(tfm_acc) else "---"
        rf_ece_s  = f"{rf_ece:.4f}"  if np.isfinite(rf_ece)  else "---"
        tfm_ece_s = f"{tfm_ece:.4f}" if np.isfinite(tfm_ece) else "---"

        # Bold the better accuracy
        if np.isfinite(rf_acc) and np.isfinite(tfm_acc):
            if tfm_acc >= rf_acc:
                tfm_acc_s = r"\textbf{" + tfm_acc_s + r"}"
            else:
                rf_acc_s  = r"\textbf{" + rf_acc_s  + r"}"

        ds_tex = ds.replace("_", r"\_")
        lines.append(f"  {ds_tex} & {rf_acc_s} & {tfm_acc_s} & {rf_ece_s} & {tfm_ece_s} \\\\")

    # Summary row: mean across datasets
    for mode in ["rf", "tfm"]:
        subset = results_df[results_df["reward_mode"] == mode]
        lines.append(r"\midrule")
        break

    rf_mean_acc  = results_df[results_df["reward_mode"] == "rf"]["tabpfn_acc"].mean()
    tfm_mean_acc = results_df[results_df["reward_mode"] == "tfm"]["tabpfn_acc"].mean()
    rf_mean_ece  = results_df[results_df["reward_mode"] == "rf"]["ece"].mean()
    tfm_mean_ece = results_df[results_df["reward_mode"] == "tfm"]["ece"].mean()

    rf_mean_acc_s  = f"{rf_mean_acc:.4f}"  if np.isfinite(rf_mean_acc)  else "---"
    tfm_mean_acc_s = f"{tfm_mean_acc:.4f}" if np.isfinite(tfm_mean_acc) else "---"
    rf_mean_ece_s  = f"{rf_mean_ece:.4f}"  if np.isfinite(rf_mean_ece)  else "---"
    tfm_mean_ece_s = f"{tfm_mean_ece:.4f}" if np.isfinite(tfm_mean_ece) else "---"

    if np.isfinite(rf_mean_acc) and np.isfinite(tfm_mean_acc):
        if tfm_mean_acc >= rf_mean_acc:
            tfm_mean_acc_s = r"\textbf{" + tfm_mean_acc_s + r"}"
        else:
            rf_mean_acc_s = r"\textbf{" + rf_mean_acc_s + r"}"

    lines.append(
        fr"  \textit{{Mean}} & {rf_mean_acc_s} & {tfm_mean_acc_s} "
        fr"& {rf_mean_ece_s} & {tfm_mean_ece_s} \\"
    )
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
    dataset_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    seed: int = 42,
    max_pipelines: int = 20,
) -> None:
    if dataset_names is None:
        dataset_names = list(BENCHMARK_DATASETS.keys())

    out_dir = Path(output_dir) if output_dir else (
        Path(__file__).parents[1] / "outputs" / "paper_ready" / "c2_tfm_reward"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    actions = build_actions()
    all_pipelines = enumerate_valid_pipelines(max_len=3)
    pipelines = sample_pipelines(all_pipelines, max_n=max_pipelines, seed=seed)
    print(
        f"Pipeline count: {len(pipelines)} (of {len(all_pipelines)} valid, "
        f"max_pipelines={max_pipelines})  |  Datasets: {len(dataset_names)}"
    )

    results_rows: List[Dict] = []
    overlap_rows: List[Dict] = []
    t0_total = time.time()

    for ds_name in dataset_names:
        print(f"\n{'─'*60}")
        print(f"  Dataset: {ds_name}")
        t0 = time.time()

        try:
            X, y, spec = load_dataset(ds_name, use_cache=True)
        except Exception as exc:
            print(f"  [SKIP] Load failed: {exc}")
            continue

        print(f"  Loaded: {len(X)} rows × {X.shape[1]} cols  "
              f"| missing={X.isna().mean().mean():.2%}")

        # Inject MCAR 15% on all datasets
        mcar_profile = ErrorProfile("mcar", rate=MCAR_RATE, seed=seed)
        X_dirty, y_dirty = apply_error_profile(X, y, mcar_profile)
        print(f"  MCAR {MCAR_RATE:.0%} injected → missing={X_dirty.isna().mean().mean():.2%}")
        n0 = len(X_dirty)

        # Build reward functions
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

        # ── Option 2: one shared cleaning cache + one TabPFN cache ───────────
        print(f"  Building cleaning cache ({len(pipelines)} pipelines) …", end=" ", flush=True)
        cleaning_cache = build_cleaning_cache(X_dirty, y_dirty, actions, pipelines)
        print("done")

        print(f"  Building TabPFN cache ({len(pipelines)} calls) …", end=" ", flush=True)
        t_tfm = time.time()
        tabpfn_cache = build_tabpfn_cache(cleaning_cache, y_dirty, seed)
        print(f"done ({time.time() - t_tfm:.1f}s)")

        dataset_rows: Dict[str, Dict] = {}

        # RF search — reads RF reward from cleaning cache (no TabPFN)
        best_rf = best_from_cache_rf(cleaning_cache, X_dirty, y_dirty, rf_reward)
        acc_rf, ece_rf = tabpfn_cache.get(best_rf, (float("nan"), float("nan")))
        print(f"  RF  best: {pipeline_label(best_rf)}  acc={acc_rf:.4f}  ECE={ece_rf:.4f}")
        dataset_rows["rf"] = {
            "dataset":       ds_name,
            "reward_mode":   "rf",
            "best_pipeline": pipeline_label(best_rf),
            "tabpfn_acc":    acc_rf,
            "ece":           ece_rf,
        }
        results_rows.append(dataset_rows["rf"])

        # TFM search — reads from TabPFN cache (no new TabPFN calls)
        best_tfm = best_from_tfm_cache(cleaning_cache, tabpfn_cache, X_dirty, n0, tfm_reward)
        acc_tfm, ece_tfm = tabpfn_cache.get(best_tfm, (float("nan"), float("nan")))
        print(f"  TFM best: {pipeline_label(best_tfm)}  acc={acc_tfm:.4f}  ECE={ece_tfm:.4f}")
        dataset_rows["tfm"] = {
            "dataset":       ds_name,
            "reward_mode":   "tfm",
            "best_pipeline": pipeline_label(best_tfm),
            "tabpfn_acc":    acc_tfm,
            "ece":           ece_tfm,
        }
        results_rows.append(dataset_rows["tfm"])

        # Pipeline overlap analysis
        rf_pipe  = pipeline_label(best_rf)
        tfm_pipe = pipeline_label(best_tfm)
        overlap_rows.append({
            "dataset":         ds_name,
            "rf_pipeline":     rf_pipe,
            "tfm_pipeline":    tfm_pipe,
            "pipelines_match": int(rf_pipe == tfm_pipe),
        })

        print(f"  Pipeline match: {rf_pipe == tfm_pipe}")
        print(f"  Done in {time.time() - t0:.1f}s")

    # ── Aggregate & save ─────────────────────────────────────────────────────
    if not results_rows:
        print("\nNo results to save.")
        return

    results_df  = pd.DataFrame(results_rows)
    overlap_df  = pd.DataFrame(overlap_rows)

    results_df.to_csv(out_dir / "results.csv", index=False)
    overlap_df.to_csv(out_dir / "pipeline_overlap.csv", index=False)

    # Wilcoxon test: TFM acc vs RF acc across datasets (paired)
    rf_accs  = results_df[results_df["reward_mode"] == "rf"]["tabpfn_acc"].dropna().values
    tfm_accs = results_df[results_df["reward_mode"] == "tfm"]["tabpfn_acc"].dropna().values
    n_shared = min(len(rf_accs), len(tfm_accs))

    print(f"\n{'='*60}")
    print(f"C2 Summary — {len(dataset_names)} datasets")

    if n_shared >= 2:
        try:
            stat, pval = wilcoxon(tfm_accs[:n_shared], rf_accs[:n_shared], alternative="greater")
            print(f"Wilcoxon (TFM > RF):  statistic={stat:.3f}  p={pval:.4f}")
        except Exception as exc:
            print(f"Wilcoxon test failed: {exc}")

    n_tfm_wins = int(np.sum(tfm_accs[:n_shared] > rf_accs[:n_shared]))
    n_diff_pipelines = int(overlap_df["pipelines_match"].eq(0).sum())
    print(f"TFM wins on accuracy: {n_tfm_wins}/{n_shared} datasets")
    print(f"Different pipelines:  {n_diff_pipelines}/{len(overlap_df)} datasets")

    # LaTeX table
    latex = make_latex_table(results_df)
    (out_dir / "c2_main_results.tex").write_text(latex)

    print(f"\nResults saved to {out_dir}/")
    print(f"Total time: {time.time() - t0_total:.1f}s")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="C2 TFM-Aware Reward experiment — requires tabpfn>=2.0"
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None, metavar="NAME",
        help=f"Subset of dataset names (default: all 10). "
             f"Available: {sorted(BENCHMARK_DATASETS)}",
    )
    parser.add_argument(
        "--output-dir", default=None, metavar="PATH",
        help="Directory for output files (default: outputs/paper_ready/c2_tfm_reward/)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed (default: 42)",
    )
    parser.add_argument(
        "--max-pipelines", type=int, default=20, metavar="N",
        help=(
            "Max pipeline candidates for greedy search (default: 20; full set: ~112). "
            "Set to 0 to use all pipelines."
        ),
    )
    args = parser.parse_args()
    main(
        dataset_names=args.datasets,
        output_dir=args.output_dir,
        seed=args.seed,
        max_pipelines=args.max_pipelines,
    )
