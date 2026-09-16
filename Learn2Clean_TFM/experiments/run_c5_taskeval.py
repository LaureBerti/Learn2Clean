from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import run_c2_tfm_reward_nested as C2
from learn2clean_v3.actions import (
    ParameterizedImputer,
    ParameterizedOutlierCleaner,
    ParameterizedScaler,
)

def build_discrete() -> Tuple[list, Dict[int, str], Dict[int, str]]:
    actions = [
        ParameterizedImputer(strategy="mean"),
        ParameterizedImputer(strategy="median"),
        ParameterizedImputer(strategy="knn", n_neighbors=5),
        ParameterizedOutlierCleaner(method="iqr", threshold=1.5),
        ParameterizedOutlierCleaner(method="zscore", threshold=3.0),
        ParameterizedScaler(method="minmax"),
        ParameterizedScaler(method="zscore"),
    ]
    labels = {0: "impute(mean)", 1: "impute(median)", 2: "impute(knn,k=5)",
              3: "outlier(iqr,t=1.5)", 4: "outlier(zscore,t=3.0)",
              5: "scale(minmax)", 6: "scale(zscore)"}
    groups = {0: "impute", 1: "impute", 2: "impute", 3: "outlier", 4: "outlier",
              5: "scale", 6: "scale"}
    return actions, labels, groups


def build_param() -> Tuple[list, Dict[int, str], Dict[int, str]]:
    actions, labels, groups = [], {}, {}
    i = 0
    actions.append(ParameterizedImputer(strategy="mean"));   labels[i] = "impute(mean)";   groups[i] = "impute"; i += 1
    actions.append(ParameterizedImputer(strategy="median")); labels[i] = "impute(median)"; groups[i] = "impute"; i += 1
    for k in (3, 5, 7, 10):
        actions.append(ParameterizedImputer(strategy="knn", n_neighbors=k))
        labels[i] = f"impute(knn,k={k})"; groups[i] = "impute"; i += 1
    for t in (1.0, 1.5, 2.0, 2.5, 3.0):
        actions.append(ParameterizedOutlierCleaner(method="iqr", threshold=t))
        labels[i] = f"outlier(iqr,t={t})"; groups[i] = "outlier"; i += 1
    for t in (2.0, 2.5, 3.0, 3.5):
        actions.append(ParameterizedOutlierCleaner(method="zscore", threshold=t))
        labels[i] = f"outlier(zscore,t={t})"; groups[i] = "outlier"; i += 1
    actions.append(ParameterizedScaler(method="minmax")); labels[i] = "scale(minmax)"; groups[i] = "scale"; i += 1
    actions.append(ParameterizedScaler(method="zscore")); labels[i] = "scale(zscore)"; groups[i] = "scale"; i += 1
    return actions, labels, groups


from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def make_prepare_test(labels: Dict[int, str]):
    def prepare_test_like_train(X_sel: pd.DataFrame, X_test: pd.DataFrame,
                                pipeline: Tuple[int, ...]) -> pd.DataFrame:
        sel = X_sel.select_dtypes(include="number").copy()
        test = X_test.select_dtypes(include="number").copy()
        shared = [c for c in sel.columns if c in test.columns]
        sel, test = sel[shared], test[shared]
        for idx in pipeline:
            label = labels.get(idx, "")
            if label.startswith("impute"):
                if "knn" in label:
                    m = re.search(r"k=(\d+)", label)
                    imp = KNNImputer(n_neighbors=int(m.group(1)) if m else 5)
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
        return test
    return prepare_test_like_train


def run_mode(mode: str, ds_names: List[str], seeds: Tuple[int, ...],
             exhaustive: bool, max_pipelines: int) -> List[Dict]:
    actions, labels, groups = (build_discrete() if mode == "discrete" else build_param())
    C2.ACTION_LABELS = labels
    C2.ACTION_GROUPS = groups
    C2.prepare_test_like_train = make_prepare_test(labels)
    all_pipelines = C2.enumerate_valid_pipelines(max_len=3)
    print(f"\n[{mode.upper()}] {len(actions)} actions | {len(all_pipelines)} valid pipelines")

    rows: List[Dict] = []
    for ds in ds_names:
        for seed in seeds:
            pipelines = all_pipelines if exhaustive else C2.sample_pipelines(all_pipelines, max_pipelines, seed)
            t = time.time()
            r = C2.run_one(ds, seed, pipelines, actions)
            if r is None:
                continue
            rows.append({"dataset": ds, "seed": seed, "mode": mode,
                         "acc": r["rf_acc"], "ece": r["rf_ece"], "f1": r["rf_f1"],
                         "pipeline": r["rf_pipeline"]})
            print(f"  {ds:<18} seed{seed}  acc={r['rf_acc']:.4f}  '{r['rf_pipeline']}'  ({time.time()-t:.0f}s)", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--sample", action="store_true", help="sample pipelines instead of exhaustive")
    ap.add_argument("--max-pipelines", type=int, default=200)
    a = ap.parse_args()

    ds_names = a.datasets or list(C2.BENCHMARK_DATASETS.keys())
    seeds = tuple(a.seeds)
    out_dir = Path(a.output_dir) if a.output_dir else (
        Path(__file__).parents[1] / "outputs" / "paper_ready" / "c5_taskeval")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows: List[Dict] = []
    for mode in ("discrete", "param"):
        rows.extend(run_mode(mode, ds_names, seeds, not a.sample, a.max_pipelines))
        pd.DataFrame(rows).to_csv(out_dir / "results_per_seed.csv", index=False)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No results."); return
    df.to_csv(out_dir / "results_per_seed.csv", index=False)

    piv = df.groupby(["dataset", "mode"]).acc.mean().unstack("mode")
    piv["delta"] = piv.get("param") - piv.get("discrete")
    piv.to_csv(out_dir / "results_aggregated.csv")
    print("\n=== C5 TASK-LEVEL (TabPFN v2 test accuracy, RF-reward selection) ===")
    print(piv.round(4).to_string())
    d = piv["delta"].dropna()
    print(f"\nmean Δ(param − discrete) TabPFN acc = {d.mean():+.4f} over {len(d)} datasets; "
          f"param wins {int((d > 0).sum())}/{len(d)}")
    if len(d) >= 3:
        try:
            stat, p = wilcoxon(piv['param'].dropna().values, piv['discrete'].dropna().values)
            print(f"paired Wilcoxon (param vs discrete test acc): stat={stat:.3f} p={p:.4f}")
        except Exception as e:
            print(f"Wilcoxon failed: {e}")
    print(f"Total wall-clock: {(time.time()-t0)/60:.1f} min\nSaved → {out_dir}")


if __name__ == "__main__":
    main()
