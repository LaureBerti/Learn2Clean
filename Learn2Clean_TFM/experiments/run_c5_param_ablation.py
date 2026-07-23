"""
experiments/run_c5_param_ablation.py

C5 — Parameterized Action Ablation
====================================
Claim: Parameterized actions (continuous k, threshold, scaler type) improve
the best-found pipeline reward vs. discrete-only on the same reward function
and datasets.

Two modes
---------
  discrete  — Fixed default sub-parameters for each action:
                  KNN imputer:         k = 5
                  IQR outlier cleaner: threshold = 1.5
                  z-score outlier:     threshold = 3.0
                  Scalers / mean / median: no sub-params (unchanged)

  param     — Expanded action set: every sub-parameter variant is treated as a
                  distinct action. The greedy oracle picks the best across all
                  candidates at every step:
                  KNN imputer:         k ∈ {3, 5, 7, 10}
                  IQR outlier cleaner: threshold ∈ {1.0, 1.5, 2.0, 2.5, 3.0}
                  z-score outlier:     threshold ∈ {2.0, 2.5, 3.0, 3.5}
                  Scalers / mean / median: no sub-params

Evaluation
----------
Greedy oracle (score every valid pipeline ≤ 3 steps, pick highest) using
MultiObjectiveReward with RF downstream evaluator (no TabPFN).

All 10 OpenML benchmark datasets. MCAR 15% injected on datasets without
natural missing values (hepatitis, diabetes, adult are exempt).

Outputs
-------
  outputs/paper_ready/c5_param_ablation/results.csv
      One row per (dataset, mode): best_score, best_pipeline, pipeline_count

  outputs/paper_ready/c5_param_ablation/c5_param_ablation.tex
      LaTeX table ready for the paper (Table C5).

Usage
-----
  PYTHONPATH=src python experiments/run_c5_param_ablation.py
  PYTHONPATH=src python experiments/run_c5_param_ablation.py --datasets hepatitis ionosphere
  PYTHONPATH=src python experiments/run_c5_param_ablation.py --mode discrete
  PYTHONPATH=src python experiments/run_c5_param_ablation.py --mode param
  PYTHONPATH=src python experiments/run_c5_param_ablation.py --mode both
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from itertools import permutations
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure src/ is importable when run as a plain script.
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Datasets that already carry natural missing values — skip MCAR injection.
NATURAL_MISSING: frozenset = frozenset({"hepatitis", "diabetes", "adult"})

# Error injection profile for datasets without natural missing.
MCAR_PROFILE = ErrorProfile(error_type="mcar", rate=0.15, seed=42)

# Output directory
OUT_DIR = (
    Path(__file__).parents[1] / "outputs" / "paper_ready" / "c5_param_ablation"
)

# ---------------------------------------------------------------------------
# ActionVariant: lightweight descriptor used when building the action table
# ---------------------------------------------------------------------------

class ActionVariant(NamedTuple):
    """A single action with a fixed set of sub-parameters."""
    action: DataFrameAction   # callable, stateless until fit
    group: str                # "impute" | "outlier" | "scale"
    label: str                # human-readable identifier


# ---------------------------------------------------------------------------
# Action suite builders
# ---------------------------------------------------------------------------

def build_discrete_suite() -> List[ActionVariant]:
    """
    7 fixed-parameter actions — the 'discrete-only' baseline.

    KNN k=5, IQR threshold=1.5, z-score threshold=3.0.
    Scalers and non-KNN imputers carry no sub-params.
    """
    return [
        ActionVariant(ParameterizedImputer(strategy="mean"),            "impute",  "impute(mean)"),
        ActionVariant(ParameterizedImputer(strategy="median"),          "impute",  "impute(median)"),
        ActionVariant(ParameterizedImputer(strategy="knn", n_neighbors=5), "impute", "impute(knn,k=5)"),
        ActionVariant(ParameterizedOutlierCleaner(method="iqr",    threshold=1.5), "outlier", "outlier(iqr,t=1.5)"),
        ActionVariant(ParameterizedOutlierCleaner(method="zscore", threshold=3.0), "outlier", "outlier(zscore,t=3.0)"),
        ActionVariant(ParameterizedScaler(method="minmax"),             "scale",   "scale(minmax)"),
        ActionVariant(ParameterizedScaler(method="zscore"),             "scale",   "scale(zscore)"),
    ]


def build_param_suite() -> List[ActionVariant]:
    """
    Expanded action suite — the 'parameterized' mode.

    Every sub-parameter variant is a distinct action:
      mean/median imputers  — unchanged (no sub-params)
      KNN imputer           — k ∈ {3, 5, 7, 10}
      IQR outlier cleaner   — threshold ∈ {1.0, 1.5, 2.0, 2.5, 3.0}
      z-score outlier       — threshold ∈ {2.0, 2.5, 3.0, 3.5}
      scalers               — unchanged (method is already the discrete dimension)

    The greedy oracle will select whichever variant yields the highest reward
    at each pipeline position.
    """
    variants: List[ActionVariant] = []

    # Non-parameterizable imputers (no expansion needed)
    variants.append(ActionVariant(ParameterizedImputer(strategy="mean"),   "impute", "impute(mean)"))
    variants.append(ActionVariant(ParameterizedImputer(strategy="median"), "impute", "impute(median)"))

    # KNN imputer — expanded over k
    for k in (3, 5, 7, 10):
        variants.append(
            ActionVariant(
                ParameterizedImputer(strategy="knn", n_neighbors=k),
                "impute",
                f"impute(knn,k={k})",
            )
        )

    # IQR outlier cleaner — expanded over threshold
    for t in (1.0, 1.5, 2.0, 2.5, 3.0):
        variants.append(
            ActionVariant(
                ParameterizedOutlierCleaner(method="iqr", threshold=t),
                "outlier",
                f"outlier(iqr,t={t})",
            )
        )

    # z-score outlier cleaner — expanded over threshold
    for t in (2.0, 2.5, 3.0, 3.5):
        variants.append(
            ActionVariant(
                ParameterizedOutlierCleaner(method="zscore", threshold=t),
                "outlier",
                f"outlier(zscore,t={t})",
            )
        )

    # Scalers — no sub-param expansion
    variants.append(ActionVariant(ParameterizedScaler(method="minmax"), "scale", "scale(minmax)"))
    variants.append(ActionVariant(ParameterizedScaler(method="zscore"), "scale", "scale(zscore)"))

    return variants


# ---------------------------------------------------------------------------
# Pipeline enumeration
# ---------------------------------------------------------------------------

def enumerate_valid_pipelines(
    suite: List[ActionVariant],
    max_len: int = 3,
) -> List[Tuple[int, ...]]:
    """
    All ordered sequences ≤ max_len steps with no repeated action *group*.

    Each index in the returned tuple refers to a position in ``suite``.
    Two variants from the same group (e.g. impute(knn,k=3) and impute(knn,k=5))
    belong to the same group and therefore cannot both appear in the same pipeline.
    """
    n = len(suite)
    result: List[Tuple[int, ...]] = [()]  # include the no-op pipeline

    for length in range(1, max_len + 1):
        for seq in permutations(range(n), length):
            groups_used = [suite[i].group for i in seq]
            if len(groups_used) == len(set(groups_used)):
                result.append(seq)

    return result


def pipeline_label(seq: Tuple[int, ...], suite: List[ActionVariant]) -> str:
    if not seq:
        return "no_op"
    return " → ".join(suite[i].label for i in seq)


# ---------------------------------------------------------------------------
# Reward builder (MultiObjectiveReward, RF, no TabPFN)
# ---------------------------------------------------------------------------

def build_reward(eval_metric: str = "f1") -> MultiObjectiveReward:
    return MultiObjectiveReward(
        weight_accuracy=0.5,
        weight_retention=0.3,
        weight_quality=0.2,
        drift_penalty_coeff=0.1,
        eval_cv_folds=1,
        eval_metric=eval_metric,
    )


# ---------------------------------------------------------------------------
# Core scoring loop (mirrors run_c1_reward_benchmark.py)
# ---------------------------------------------------------------------------

def score_all_pipelines(
    X: pd.DataFrame,
    y: pd.Series,
    suite: List[ActionVariant],
    reward_fn: MultiObjectiveReward,
    pipelines: List[Tuple[int, ...]],
) -> pd.DataFrame:
    """
    Apply every pipeline and score with the given reward function.

    Returns a DataFrame with columns:
        pipeline, steps, n_rows, score
    """
    rows = []
    for seq in pipelines:
        X_clean = X.copy()
        ok = True
        for idx in seq:
            action = suite[idx].action
            try:
                action.reset()
                X_clean = action(X_clean.copy(), y)
            except Exception:
                ok = False
                break
        if not ok:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reward_fn.reset(X, y)
            score = reward_fn(X_clean, y)

        rows.append({
            "pipeline": pipeline_label(seq, suite),
            "steps":    len(seq),
            "n_rows":   int(len(X_clean)),
            "score":    round(float(score) if np.isfinite(score) else 0.0, 6),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["pipeline", "steps", "n_rows", "score"]
    )


# ---------------------------------------------------------------------------
# LaTeX table builder
# ---------------------------------------------------------------------------

def make_latex_table(results: pd.DataFrame) -> str:
    """
    Build a LaTeX table comparing discrete vs. parameterized best scores
    across datasets.

    ``results`` must have columns:
        dataset, mode, best_score, best_pipeline, pipeline_count
    """
    # Pivot to wide format: one row per dataset, columns per mode
    discrete_df = results[results["mode"] == "discrete"].set_index("dataset")
    param_df    = results[results["mode"] == "param"].set_index("dataset")

    datasets = sorted(set(discrete_df.index) | set(param_df.index))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{C5 — Best-found pipeline reward: discrete-only vs.\ parameterized "
        r"actions across 10 OpenML datasets (MCAR 15\%, MultiObjectiveReward, RF "
        r"evaluator). $\uparrow$ = parameterized wins; $\rightarrow$ = tie.}",
        r"\label{tab:c5_param_ablation}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Dataset & Discrete & Parameterized & $\Delta$ & Winner \\",
        r"\midrule",
    ]

    wins_param    = 0
    wins_discrete = 0
    ties          = 0
    deltas        = []

    for ds in datasets:
        d_score = discrete_df.loc[ds, "best_score"] if ds in discrete_df.index else float("nan")
        p_score = param_df.loc[ds, "best_score"]    if ds in param_df.index    else float("nan")
        delta   = p_score - d_score if (np.isfinite(d_score) and np.isfinite(p_score)) else float("nan")
        deltas.append(delta)

        if np.isfinite(delta):
            if delta > 1e-4:
                winner = r"$\uparrow$ param"
                wins_param += 1
            elif delta < -1e-4:
                winner = r"$\downarrow$ discrete"
                wins_discrete += 1
            else:
                winner = r"$\rightarrow$ tie"
                ties += 1
        else:
            winner = "—"

        ds_tex   = ds.replace("_", r"\_")
        d_str    = f"{d_score:.4f}" if np.isfinite(d_score) else "—"
        p_str    = f"{p_score:.4f}" if np.isfinite(p_score) else "—"
        delta_str = f"{delta:+.4f}" if np.isfinite(delta) else "—"

        # Bold the higher value
        if np.isfinite(delta) and delta > 1e-4:
            p_str = r"\textbf{" + p_str + r"}"
        elif np.isfinite(delta) and delta < -1e-4:
            d_str = r"\textbf{" + d_str + r"}"

        lines.append(f"  {ds_tex} & {d_str} & {p_str} & {delta_str} & {winner} \\\\")

    # Summary row
    mean_delta = float(np.nanmean(deltas)) if any(np.isfinite(d) for d in deltas) else float("nan")
    mean_delta_str = f"{mean_delta:+.4f}" if np.isfinite(mean_delta) else "—"

    lines += [
        r"\midrule",
        f"  \\textit{{Mean}} & & & {mean_delta_str} & "
        f"param wins {wins_param}/{len(datasets)} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-mode runner
# ---------------------------------------------------------------------------

def run_mode(
    mode: str,
    suite: List[ActionVariant],
    dataset_names: List[str],
) -> List[Dict]:
    """
    Score every valid pipeline for the given suite and datasets.

    Returns a list of result dicts (one per dataset).
    """
    pipelines = enumerate_valid_pipelines(suite, max_len=3)
    print(
        f"\n[{mode.upper()}] suite size={len(suite)} actions  "
        f"| valid pipelines={len(pipelines)}"
    )

    rows = []
    for ds_name in dataset_names:
        print(f"  ├─ {ds_name} …", end=" ", flush=True)
        t0 = time.time()

        try:
            X_raw, y, spec = load_dataset(ds_name, use_cache=True)
        except Exception as exc:
            print(f"SKIP (load error: {exc})")
            continue

        # Error injection
        if ds_name not in NATURAL_MISSING:
            X_work, y = apply_error_profile(X_raw, y, MCAR_PROFILE)
        else:
            X_work = X_raw

        reward_fn = build_reward(eval_metric=spec.eval_metric)

        scored = score_all_pipelines(X_work, y, suite, reward_fn, pipelines)

        if scored.empty:
            print("SKIP (no valid pipeline scored)")
            continue

        best_row = scored.nlargest(1, "score").iloc[0]
        elapsed  = time.time() - t0
        print(
            f"best={best_row['score']:.4f}  "
            f"pipeline='{best_row['pipeline']}'  "
            f"({elapsed:.1f}s)"
        )

        rows.append({
            "dataset":        ds_name,
            "mode":           mode,
            "best_score":     round(float(best_row["score"]), 6),
            "best_pipeline":  best_row["pipeline"],
            "pipeline_count": len(scored),
            "eval_metric":    spec.eval_metric,
        })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    dataset_names: Optional[List[str]] = None,
    modes: Optional[List[str]] = None,
) -> None:
    if dataset_names is None:
        dataset_names = list(BENCHMARK_DATASETS.keys())
    if modes is None:
        modes = ["discrete", "param"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0_total = time.time()
    all_rows: List[Dict] = []

    suite_map = {
        "discrete": build_discrete_suite,
        "param":    build_param_suite,
    }

    for mode in modes:
        suite = suite_map[mode]()
        rows  = run_mode(mode, suite, dataset_names)
        all_rows.extend(rows)

    if not all_rows:
        print("\nNo results collected — nothing to save.")
        return

    results_df = pd.DataFrame(all_rows)

    # ── Print comparison summary ─────────────────────────────────────────────
    if {"discrete", "param"}.issubset(set(results_df["mode"].unique())):
        print(f"\n{'='*70}")
        print("C5 Summary — Parameterized vs. Discrete (best pipeline reward):")
        pivot = results_df.pivot(index="dataset", columns="mode", values="best_score")
        pivot.columns.name = None
        pivot = pivot.reset_index()
        if "discrete" in pivot.columns and "param" in pivot.columns:
            pivot["delta"]  = pivot["param"] - pivot["discrete"]
            pivot["winner"] = pivot["delta"].apply(
                lambda d: "param" if d > 1e-4 else ("discrete" if d < -1e-4 else "tie")
            )
        print(pivot.to_string(index=False, float_format="{:.4f}".format))

        wins = (pivot["winner"] == "param").sum() if "winner" in pivot.columns else 0
        n    = len(pivot)
        print(f"\nParam wins: {wins}/{n} datasets")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults → {csv_path}")

    # ── Build and save LaTeX table ───────────────────────────────────────────
    if {"discrete", "param"}.issubset(set(results_df["mode"].unique())):
        latex   = make_latex_table(results_df)
        tex_path = OUT_DIR / "c5_param_ablation.tex"
        tex_path.write_text(latex, encoding="utf-8")
        print(f"LaTeX  → {tex_path}")

    total = time.time() - t0_total
    print(f"Total time: {total:.1f}s")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "C5 — Parameterized vs. discrete action ablation. "
            "Compares best-found pipeline reward under fixed vs. expanded sub-parameters."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        metavar="NAME",
        help=(
            "Subset of dataset names to run (default: all 10). "
            f"Available: {sorted(BENCHMARK_DATASETS)}"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["discrete", "param", "both"],
        default="both",
        help=(
            "Which mode(s) to run. "
            "'discrete' = fixed default sub-params; "
            "'param' = expanded sub-parameter variants; "
            "'both' = run both and compare (default)."
        ),
    )
    args = parser.parse_args()

    modes: List[str]
    if args.mode == "both":
        modes = ["discrete", "param"]
    else:
        modes = [args.mode]

    main(dataset_names=args.datasets, modes=modes)
