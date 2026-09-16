
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from learn2clean_v3.actions import (
    DataFrameAction,
    ParameterizedImputer,
    ParameterizedOutlierCleaner,
    ParameterizedScaler,
)
from learn2clean_v3.data.error_injection import ErrorProfile, apply_error_profile
from learn2clean_v3.data.openml_loader import BENCHMARK_DATASETS, load_dataset
from learn2clean_v3.rewards import (
    AccuracyReward,
    BaseReward,
    CompletenessRetentionReward,
    DataDistortionPenaltyReward,
    DriftPenaltyReward,
    IncrementalGainReward,
    MultiObjectiveReward,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

NATURAL_MISSING: set = {"hepatitis", "diabetes", "adult"}

MCAR_PROFILE = ErrorProfile("mcar", rate=0.15, seed=42)

OUT_DIR = Path(__file__).parents[1] / "outputs" / "paper_ready" / "c1_reward_benchmark"


ACTION_GROUPS: Dict[int, str] = {
    0: "impute", 1: "impute", 2: "impute",
    3: "outlier", 4: "outlier",
    5: "scale",  6: "scale",
}

ACTION_LABELS: Dict[int, str] = {
    0: "impute(mean)",
    1: "impute(median)",
    2: "impute(knn)",
    3: "outlier(iqr)",
    4: "outlier(zscore)",
    5: "scale(minmax)",
    6: "scale(zscore)",
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
    ]



def enumerate_valid_pipelines(max_len: int = 3) -> List[Tuple[int, ...]]:
    from itertools import permutations
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



def build_reward_functions(eval_metric: str = "f1") -> List[BaseReward]:
    m = eval_metric
    base_multi = MultiObjectiveReward(
        weight_accuracy=0.5, weight_retention=0.3, weight_quality=0.2,
        drift_penalty_coeff=0.1, eval_cv_folds=1, eval_metric=m,
    )
    return [
        CompletenessRetentionReward(),
        AccuracyReward(eval_model="random_forest", eval_metric=m),
        MultiObjectiveReward(
            weight_accuracy=0.5, weight_retention=0.3, weight_quality=0.2,
            drift_penalty_coeff=0.1, eval_cv_folds=1, eval_metric=m,
        ),
        DriftPenaltyReward(drift_coeff=0.4, eval_model="random_forest", eval_metric=m),
        IncrementalGainReward(base_reward=base_multi),
        DataDistortionPenaltyReward(
            weight_wasserstein=0.30, weight_js=0.25, weight_correlation=0.20,
            weight_variance=0.15, weight_skewness=0.10,
        ).set_name("DataDistortionPenalty(dist)"),
        DataDistortionPenaltyReward(
            weight_accuracy=0.30, weight_wasserstein=0.25,
            weight_js=0.20, weight_correlation=0.15,
            weight_variance=0.05, weight_skewness=0.05,
            eval_cv_folds=1,
        ).set_name("DataDistortionPenalty(acc+dist)"),
    ]



def score_all_pipelines(
    X: pd.DataFrame,
    y: pd.Series,
    actions: List[DataFrameAction],
    reward_fns: List[BaseReward],
    pipelines: List[Tuple[int, ...]],
) -> pd.DataFrame:
    rows = []
    for seq in pipelines:
        X_clean = X.copy()
        ok = True
        for idx in seq:
            try:
                actions[idx].reset()
                X_clean = actions[idx](X_clean.copy(), y)
            except Exception:
                ok = False
                break
        if not ok:
            continue

        row: Dict = {
            "pipeline": pipeline_label(seq),
            "steps":    len(seq),
            "n_rows":   len(X_clean),
        }
        for rf in reward_fns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rf.reset(X, y)
                score = rf(X_clean, y)
            row[rf.name] = round(float(score) if np.isfinite(score) else 0.0, 4)
        rows.append(row)
    return pd.DataFrame(rows)



def make_latex_table(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{C1 — Mean best-pipeline score per reward function across 10 OpenML "
        r"datasets (MCAR 15\%, 112 valid pipelines, RF downstream evaluator). "
        r"Bold = best mean.}",
        r"\label{tab:c1_reward_compare}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Reward Function & Mean & Std & Max \\",
        r"\midrule",
    ]
    best_mean = summary["mean"].max()
    for _, row in summary.iterrows():
        name = row["reward_fn"].replace("_", r"\_")
        mean_str = f"{row['mean']:.4f}"
        if abs(row["mean"] - best_mean) < 1e-6:
            mean_str = r"\textbf{" + mean_str + r"}"
        lines.append(
            f"  {name} & {mean_str} & {row['std']:.4f} & {row['max']:.4f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)



def main(dataset_names: Optional[List[str]] = None) -> None:
    if dataset_names is None:
        dataset_names = list(BENCHMARK_DATASETS.keys())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    actions = build_actions()
    pipelines = enumerate_valid_pipelines(max_len=3)
    print(f"Pipeline count: {len(pipelines)}  |  Datasets: {len(dataset_names)}")

    all_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    t0_total = time.time()

    for ds_name in dataset_names:
        print(f"\n{'─'*60}")
        print(f"  Dataset: {ds_name}")
        t0 = time.time()

        try:
            X_clean, y, spec = load_dataset(ds_name, use_cache=True)
        except Exception as exc:
            print(f"  [SKIP] Load failed: {exc}")
            continue

        print(f"  Loaded: {len(X_clean)} rows × {X_clean.shape[1]} cols  "
              f"| missing={X_clean.isna().mean().mean():.2%}  "
              f"| metric={spec.eval_metric}")

        if ds_name not in NATURAL_MISSING:
            X_dirty, y = apply_error_profile(X_clean, y, MCAR_PROFILE)
            print(f"  Injected MCAR 15% → missing={X_dirty.isna().mean().mean():.2%}")
        else:
            X_dirty = X_clean
            print(f"  Natural missing retained (no injection)")

        reward_fns = build_reward_functions(eval_metric=spec.eval_metric)

        print(f"  Scoring {len(pipelines)} pipelines × {len(reward_fns)} reward fns …")
        scored = score_all_pipelines(X_dirty, y, actions, reward_fns, pipelines)

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

        for _, row in scored.iterrows():
            for rf in reward_fns:
                all_rows.append({
                    "dataset":    ds_name,
                    "pipeline":   row["pipeline"],
                    "steps":      row["steps"],
                    "n_rows":     row["n_rows"],
                    "reward_fn":  rf.name,
                    "score":      row.get(rf.name, np.nan),
                })

        for rf in reward_fns:
            col = rf.name
            if col not in scored.columns:
                continue
            best_row = scored.nlargest(1, col).iloc[0]
            summary_rows.append({
                "dataset":       ds_name,
                "reward_fn":     col,
                "best_pipeline": best_row["pipeline"],
                "steps":         int(best_row["steps"]),
                "n_rows":        int(best_row["n_rows"]),
                "best_score":    best_row[col],
            })

        for rf in reward_fns:
            col = rf.name
            if col not in scored.columns:
                continue
            best = scored.nlargest(1, col).iloc[0]
            print(f"    {col:40s}  best={best[col]:.4f}  pipeline: {best['pipeline']}")

    if not summary_rows:
        print("\nNo results to aggregate.")
        return

    all_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summary_rows)

    rf_perf = (
        summary_df.groupby("reward_fn")["best_score"]
        .agg(["mean", "std", "max"])
        .sort_values("mean", ascending=False)
        .reset_index()
        .rename(columns={"reward_fn": "reward_fn", "mean": "mean", "std": "std", "max": "max"})
    )
    rf_perf["std"] = rf_perf["std"].fillna(0.0)

    print(f"\n{'='*60}")
    print("C1 — Mean best score per reward function (across datasets):")
    print(rf_perf.to_string(index=False, float_format="{:.4f}".format))

    all_df.to_csv(OUT_DIR / "all_results.csv", index=False)
    summary_df.to_csv(OUT_DIR / "summary_by_dataset.csv", index=False)
    rf_perf.to_csv(OUT_DIR / "summary_by_reward.csv", index=False)

    latex = make_latex_table(rf_perf)
    (OUT_DIR / "c1_reward_compare.tex").write_text(latex)

    total = time.time() - t0_total
    print(f"\nResults saved to {OUT_DIR}/")
    print(f"Total time: {total:.1f}s")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C1 reward taxonomy benchmark")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        metavar="NAME",
        help="Subset of dataset names (default: all 10). "
             f"Available: {sorted(BENCHMARK_DATASETS)}",
    )
    args = parser.parse_args()
    main(dataset_names=args.datasets)
