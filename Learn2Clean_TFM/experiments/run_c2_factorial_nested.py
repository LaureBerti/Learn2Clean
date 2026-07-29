"""
experiments/run_c2_factorial_nested.py

FAIR 2x2 factorial RF-vs-TFM comparison (addresses the confound that the
submitted RF-reward and TFM-reward used DIFFERENT weights AND a different functional
form, so "RF vs TFM" conflated the eval model with the weighting).

Two factors, fully crossed, held-out protocol (same outer/inner split + TabPFN final eval as D1):
  Factor A — weight config:
      rfw  : w_acc=0.50 w_ret=0.30 w_qual=0.20  retention^1.0  drift=0.10
      tfmw : w_acc=0.50 w_ret=0.35 w_qual=0.15  retention^2.0  drift=0.05
  Factor B — selection eval model (the accuracy proxy in the reward):
      rf      : RandomForest inner-val accuracy
      tabpfn  : TabPFN v2  inner-val accuracy

CRITICAL: the scoring function is UNIFIED — identical code path, identical inner-val
protocol (one inner split), identical drift definition (Wasserstein to the dirty
pre-cleaning data, faithful to the submitted reward). Within a weight config the ONLY
thing that changes across the two cells is the eval model; across weight configs only
(weights, retention power, drift coeff) change. Final accuracy/ECE for every cell is
reported with TabPFN v2 on the untouched outer test (TabPFN is the deployment model
regardless of which proxy selected the pipeline).

This lets us estimate, held-out protocol and with multiple seeds:
  * main effect of the eval model (does selecting with TabPFN beat selecting with RF?),
  * main effect of the weights,
  * their interaction (does TabPFN-eval help specifically under the TFM weighting?).

Usage
-----
  PYTHONPATH=src:experiments python experiments/run_c2_factorial_nested.py --seeds 42 1 2 3 4 5 6 7
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import run_c2_tfm_reward_nested as G
from learn2clean_v3.data.error_injection import ErrorProfile, apply_error_profile
from learn2clean_v3.data.openml_loader import BENCHMARK_DATASETS, load_dataset

WEIGHT_CONFIGS: Dict[str, Dict[str, float]] = {
    "rfw":  {"w_acc": 0.50, "w_ret": 0.30, "w_qual": 0.20, "power": 1.0, "drift": 0.10},
    "tfmw": {"w_acc": 0.50, "w_ret": 0.35, "w_qual": 0.15, "power": 2.0, "drift": 0.05},
}
EVAL_MODELS = ["rf", "tabpfn"]


def inner_val_acc(X_clean: pd.DataFrame, y: pd.Series, seed: int, eval_model: str) -> float:
    """UNIFIED selection proxy: one inner train/val split; fit RF or TabPFN; return
    accuracy on the inner-val (never touches the outer test)."""
    X_arr, y_enc, _ = G._encode_align(X_clean, y)
    if len(y_enc) < 20 or len(np.unique(y_enc)) < 2:
        return float("nan")
    try:
        Xtr, Xval, ytr, yval = train_test_split(
            X_arr, y_enc, test_size=G.INNER_VAL_SIZE, random_state=seed, stratify=y_enc)
    except ValueError:
        return float("nan")
    if len(np.unique(ytr)) < 2:
        return float("nan")
    try:
        if eval_model == "rf":
            clf = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1)
            clf.fit(Xtr, ytr)
            pred = clf.predict(Xval)
        else:
            pred, _ = G._tabpfn_fit_predict(Xtr, ytr, Xval, seed)
        return float(accuracy_score(yval, pred))
    except Exception:
        return float("nan")


def drift_to_dirty(X_clean: pd.DataFrame, ref_cols: Dict[str, np.ndarray]) -> float:
    """Mean column-wise normalised Wasserstein-1 distance from cleaned data to the
    DIRTY pre-cleaning marginals (faithful to the submitted _drift_score)."""
    numeric = X_clean.select_dtypes(include="number")
    dists: List[float] = []
    for col in numeric.columns:
        if col not in ref_cols:
            continue
        cur = numeric[col].dropna().values
        ref = ref_cols[col]
        if len(cur) < 2 or len(ref) < 2:
            continue
        try:
            w = float(stats.wasserstein_distance(cur, ref))
            dists.append(min(w / (float(np.std(ref)) or 1.0), 5.0))
        except Exception:
            continue
    return float(np.mean(dists)) if dists else 0.0


def select(X_sel, y_sel, pipelines, actions, seed, cfg, eval_model, ref_cols) -> Tuple[int, ...]:
    n0 = len(X_sel)
    best, best_score = (), -np.inf
    for seq in pipelines:
        X_clean = G.apply_pipeline(X_sel, y_sel, seq, actions)
        if X_clean is None or len(X_clean) == 0:
            continue
        acc = inner_val_acc(X_clean, y_sel, seed, eval_model)
        if not np.isfinite(acc):
            continue
        ret = (len(X_clean) / n0) ** cfg["power"]
        miss = float(X_clean.isna().mean().mean())
        dup = float(X_clean.duplicated().sum()) / max(len(X_clean), 1)
        qual = (1.0 - miss) * (1.0 - dup)
        drift = drift_to_dirty(X_clean, ref_cols)
        score = cfg["w_acc"] * acc + cfg["w_ret"] * ret + cfg["w_qual"] * qual - cfg["drift"] * drift
        if score > best_score:
            best_score, best = score, seq
    return best


def run_one(ds_name, seed, pipelines, actions) -> Optional[Dict]:
    try:
        X, y, _ = load_dataset(ds_name, use_cache=True)
    except Exception as exc:
        print(f"  [SKIP] {ds_name}: {exc}"); return None
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

    # dirty reference for drift = pre-cleaning D_sel marginals
    ref_cols = {c: X_sel[c].dropna().values.astype(float)
                for c in X_sel.select_dtypes(include="number").columns}

    row: Dict = {"dataset": ds_name, "seed": seed}
    for wname, cfg in WEIGHT_CONFIGS.items():
        for em in EVAL_MODELS:
            best = select(X_sel, y_sel, pipelines, actions, seed, cfg, em, ref_cols)
            X_clean = G.apply_pipeline(X_sel, y_sel, best, actions)
            X_test_prep = G.prepare_test_like_train(X_sel, X_test, best)
            acc, ece = G.final_test_tabpfn(X_clean, y_sel, X_test_prep, y_test, seed)
            cell = f"{wname}_{em}"
            row[f"{cell}_acc"], row[f"{cell}_ece"] = acc, ece
            row[f"{cell}_pipe"] = G.pipeline_label(best)
    return row


def summarize(df: pd.DataFrame) -> None:
    cells = [f"{w}_{e}" for w in WEIGHT_CONFIGS for e in EVAL_MODELS]
    print(f"\n{'='*64}\n2x2 FACTORIAL (held-out protocol) — final TabPFN acc, mean over runs")
    means = {c: df[f"{c}_acc"].dropna().mean() for c in cells}
    for c in cells:
        print(f"  {c:12} acc={means[c]:.4f}")
    # main effects
    rf_eval  = np.nanmean([df["rfw_rf_acc"], df["tfmw_rf_acc"]])
    tab_eval = np.nanmean([df["rfw_tabpfn_acc"], df["tfmw_tabpfn_acc"]])
    rfw  = np.nanmean([df["rfw_rf_acc"], df["rfw_tabpfn_acc"]])
    tfmw = np.nanmean([df["tfmw_rf_acc"], df["tfmw_tabpfn_acc"]])
    print(f"\n  MAIN EFFECT eval model : TabPFN-eval={tab_eval:.4f}  RF-eval={rf_eval:.4f}  "
          f"Δ={tab_eval-rf_eval:+.4f}")
    print(f"  MAIN EFFECT weights    : tfmw={tfmw:.4f}  rfw={rfw:.4f}  Δ={tfmw-rfw:+.4f}")
    inter = (means["tfmw_tabpfn"] - means["tfmw_rf"]) - (means["rfw_tabpfn"] - means["rfw_rf"])
    print(f"  INTERACTION (TabPFN help under tfmw vs rfw): {inter:+.4f}")


def main(dataset_names=None, output_dir=None, seeds=(42,), max_pipelines=20) -> None:
    if dataset_names is None:
        dataset_names = list(BENCHMARK_DATASETS.keys())
    out_dir = Path(output_dir) if output_dir else (
        Path(__file__).parents[1] / "outputs" / "paper_ready" / "c2_factorial_nested")
    out_dir.mkdir(parents=True, exist_ok=True)

    actions = G.build_actions()
    all_pipelines = G.enumerate_valid_pipelines(max_len=3)
    rows: List[Dict] = []
    t0 = time.time()
    for ds in dataset_names:
        for seed in seeds:
            pipelines = G.sample_pipelines(all_pipelines, max_pipelines, seed)
            print(f"\n── {ds} | seed {seed} ──", flush=True)
            ts = time.time()
            r = run_one(ds, seed, pipelines, actions)
            if r is not None:
                rows.append(r)
                print(f"   rfw_rf={r['rfw_rf_acc']:.4f} rfw_tab={r['rfw_tabpfn_acc']:.4f} "
                      f"tfmw_rf={r['tfmw_rf_acc']:.4f} tfmw_tab={r['tfmw_tabpfn_acc']:.4f} "
                      f"({time.time()-ts:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(out_dir / "factorial_per_run.csv", index=False)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "factorial_per_run.csv", index=False)
        summarize(df)
        print(f"\nWall-clock: {(time.time()-t0)/60:.1f} min → {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--max-pipelines", type=int, default=20)
    a = ap.parse_args()
    main(a.datasets, a.output_dir, tuple(a.seeds), a.max_pipelines)
