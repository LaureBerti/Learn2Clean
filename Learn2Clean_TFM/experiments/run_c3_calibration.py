
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

try:
    import tabpfn as _tabpfn_check
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

NATURAL_MISSING: frozenset = frozenset({"hepatitis", "diabetes", "adult"})

MCAR_RATES: List[float] = [0.0, 0.05, 0.15, 0.30]

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
    5: "scale",   6: "scale",   8: "scale",
    7: "dedup",
}
ACTION_LABELS: Dict[int, str] = {
    0: "impute(mean)",   1: "impute(median)", 2: "impute(knn)",
    3: "outlier(iqr)",   4: "outlier(zscore)",
    5: "scale(minmax)",  6: "scale(zscore)",  8: "scale(quantile)",
    7: "dedup(first)",
}



def build_actions() -> List[DataFrameAction]:
    return [
        ParameterizedImputer(strategy="mean"),
        ParameterizedImputer(strategy="median"),
        ParameterizedImputer(strategy="knn", n_neighbors=5),
        ParameterizedOutlierCleaner(method="iqr",    threshold=1.5),
        ParameterizedOutlierCleaner(method="zscore", threshold=3.0),
        ParameterizedScaler(method="minmax"),
        ParameterizedScaler(method="zscore"),
        ParameterizedDeduplicator(keep="first", subset="all"),
        ParameterizedScaler(method="quantile"),
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
    X_out = X.copy()
    for idx in pipeline:
        try:
            actions[idx].reset()
            X_out = actions[idx](X_out.copy(), y)
        except Exception:
            return None
    return X_out


def apply_b1_baseline(X: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    X_out = X.copy()
    if numeric_cols:
        imputer = SimpleImputer(strategy="mean")
        X_out[numeric_cols] = imputer.fit_transform(X_out[numeric_cols])
        scaler = MinMaxScaler()
        X_out[numeric_cols] = scaler.fit_transform(X_out[numeric_cols])
    return X_out


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
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
    from tabpfn import TabPFNClassifier

    numeric = X_clean.select_dtypes(include="number")
    if numeric.shape[1] == 0:
        return float("nan"), float("nan")

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



def make_latex_table_c3(results_df: pd.DataFrame) -> str:
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
    rows: List[Dict] = []
    n0 = len(X_dirty)
    base_info = {
        "dataset":      ds_name,
        "error_type":   error_type,
        "mcar_rate":    mcar_rate,
        "n_rows_dirty": n0,
        "n_cols":       X_dirty.shape[1],
    }

    t0 = time.time()
    acc_b0, ece_b0 = evaluate_with_tabpfn(X_dirty, y, seed=seed)
    rows.append({**base_info, "baseline": "B0",
                 "tabpfn_acc": acc_b0, "ece": ece_b0,
                 "time_search_s": 0.0, "time_eval_s": round(time.time() - t0, 3),
                 "n_pipelines": 0, "best_pipeline": "no_op"})

    t0 = time.time()
    X_b1 = apply_b1_baseline(X_dirty)
    acc_b1, ece_b1 = evaluate_with_tabpfn(X_b1, y, seed=seed)
    rows.append({**base_info, "baseline": "B1",
                 "tabpfn_acc": acc_b1, "ece": ece_b1,
                 "time_search_s": 0.0, "time_eval_s": round(time.time() - t0, 3),
                 "n_pipelines": 0, "best_pipeline": "impute(mean) → scale(minmax)"})

    t_clean = time.time()
    cleaning_cache = build_cleaning_cache(X_dirty, y, actions, pipelines)
    t_clean_total  = round(time.time() - t_clean, 3)

    t_tfm_cache = time.time()
    tabpfn_cache = build_tabpfn_cache(cleaning_cache, y, seed)
    t_tfm_cache_total = round(time.time() - t_tfm_cache, 3)

    t_rf = time.time()
    best_rf = best_from_cache_rf(cleaning_cache, X_dirty, y, rf_reward)
    t_rf_search = round(time.time() - t_rf, 3)
    acc_rf, ece_rf = tabpfn_cache.get(best_rf, (float("nan"), float("nan")))
    rows.append({**base_info, "baseline": "B-greedy-RF",
                 "tabpfn_acc": acc_rf, "ece": ece_rf,
                 "time_search_s": t_rf_search,
                 "time_eval_s": 0.0,
                 "n_pipelines": len(pipelines), "best_pipeline": pipeline_label(best_rf)})

    t_tfm = time.time()
    best_tfm = best_from_tfm_cache(cleaning_cache, tabpfn_cache, X_dirty, n0, tfm_reward)
    t_tfm_search = round(time.time() - t_tfm, 3)
    acc_tfm, ece_tfm = tabpfn_cache.get(best_tfm, (float("nan"), float("nan")))
    rows.append({**base_info, "baseline": "B-greedy-TFM",
                 "tabpfn_acc": acc_tfm, "ece": ece_tfm,
                 "time_search_s": t_tfm_search,
                 "time_eval_s": 0.0,
                 "n_pipelines": len(pipelines), "best_pipeline": pipeline_label(best_tfm)})

    rows[0]["time_clean_cache_s"] = t_clean_total
    rows[0]["time_tabpfn_cache_s"] = t_tfm_cache_total

    return rows



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
    dataset_timing: List[Dict] = []
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

        for prof in ERROR_TYPE_PROFILES:
            if prof.error_type == "mcar":
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

        if all_results:
            pd.DataFrame(all_results).to_csv(
                out_dir / "results_partial.csv", index=False
            )
        if error_type_results:
            pd.DataFrame(error_type_results).to_csv(
                out_dir / "c3_error_type_partial.csv", index=False
            )

    if not all_results:
        print("\nNo results to save.")
        return

    results_df = pd.DataFrame(all_results)
    error_df   = pd.DataFrame(error_type_results)

    results_df.to_csv(out_dir / "results.csv", index=False)
    error_df.to_csv(out_dir / "c3_error_type.csv", index=False)

    timing_df = pd.DataFrame(dataset_timing)
    timing_df.to_csv(out_dir / "timing_per_profile.csv", index=False)

    timing_per_ds = (
        timing_df.groupby("dataset")[["time_profile_s", "time_rf_search_s", "time_tfm_search_s"]]
        .sum()
        .rename(columns={"time_profile_s": "total_s",
                         "time_rf_search_s": "total_rf_search_s",
                         "time_tfm_search_s": "total_tfm_search_s"})
        .reset_index()
    )
    timing_per_ds.to_csv(out_dir / "timing_per_dataset.csv", index=False)

    timing_per_etype = (
        timing_df.groupby("error_type")[["time_profile_s", "time_rf_search_s", "time_tfm_search_s"]]
        .mean()
        .rename(columns={"time_profile_s": "mean_profile_s",
                         "time_rf_search_s": "mean_rf_search_s",
                         "time_tfm_search_s": "mean_tfm_search_s"})
        .reset_index()
    )
    timing_per_etype.to_csv(out_dir / "timing_per_error_type.csv", index=False)

    all_both = pd.concat([results_df, error_df], ignore_index=True)
    timing_per_baseline = (
        all_both.groupby("baseline")[["time_search_s", "time_eval_s"]]
        .mean()
        .reset_index()
    )
    timing_per_baseline.to_csv(out_dir / "timing_per_baseline.csv", index=False)

    print(f"\n{'─'*60}")
    print("Timing summary — per dataset (total seconds):")
    print(timing_per_ds.to_string(index=False, float_format="{:.1f}".format))
    print("\nTiming summary — mean per error type:")
    print(timing_per_etype.to_string(index=False, float_format="{:.1f}".format))
    print("\nTiming summary — mean per baseline:")
    print(timing_per_baseline.to_string(index=False, float_format="{:.2f}".format))

    sens_df = (
        results_df[results_df["error_type"] == "mcar"]
        .groupby(["mcar_rate", "baseline"])[["tabpfn_acc", "ece"]]
        .mean()
        .reset_index()
    )
    sens_df.to_csv(out_dir / "c3_sensitivity_curves.csv", index=False)

    latex = make_latex_table_c3(results_df)
    (out_dir / "c3_calibration.tex").write_text(latex)

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
