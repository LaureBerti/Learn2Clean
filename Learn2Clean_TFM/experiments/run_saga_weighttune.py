"""

SAGA comparison — does PER-DATASET weight tuning (and the drift term) change the verdict?
Builds a per-pipeline pool on each SAGA dataset (inner-val acc + retention/Q/W1 + sacred-test
LogReg/TabPFN accuracy), then evaluates, per dataset:

  fixed         : the a-priori R7 weights (0.50, 0.35, 0.15), power=2
  ORACLE        : weights chosen to MAXIMISE the sacred-test accuracy  (UPPER BOUND — NOT
                  deployable; this is selection-on-test, reported only to show the ceiling)
  leakfree-tune : weights chosen on a held-out inner tuning fold, reported on the sacred test
each × drift ∈ {off (λ=0), on (λ=0.05)}.

Why the oracle matters: if even test-optimal per-dataset weights don't beat SAGA, weight
tuning cannot rescue us. And note the leak-free taxonomy showed accuracy-tuning collapses to
the over-pruning corner (the worst reward), so deployable tuning is expected to regress.

Usage
-----
  PYTHONPATH=src:experiments python experiments/run_saga_weighttune.py --seeds 42 1 2 3 4
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

import run_c2_tfm_reward_nested as G
import run_weight_robustness as W
from run_saga_comparison import DATASETS, SAGA_PUBLISHED, load_saga

LAMBDAS = [0.0, 0.05]
POWERS = [1.0, 2.0]
FIXED = (0.50, 0.35, 0.15)


def drift_to_dirty(Xc, ref):
    num = Xc.select_dtypes(include="number"); d = []
    for c in num.columns:
        if c not in ref: continue
        cur, r = num[c].dropna().values, ref[c]
        if len(cur) >= 2 and len(r) >= 2:
            try: d.append(min(float(stats.wasserstein_distance(cur, r)) / (float(np.std(r)) or 1.0), 5.0))
            except Exception: pass
    return float(np.mean(d)) if d else 0.0


def build_pool(ds, seed, pipelines, actions, tune_frac=0.25) -> List[Dict]:
    loaded = load_saga(ds)
    if loaded is None: return []
    X, y = loaded
    if len(X) > G.SUBSAMPLE_CAP:
        X, _, y, _ = train_test_split(X, y, train_size=G.SUBSAMPLE_CAP, random_state=seed, stratify=y)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    try:
        X_sel, X_te, y_sel, y_te = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
    except ValueError:
        X_sel, X_te, y_sel, y_te = train_test_split(X, y, test_size=0.30, random_state=seed)
    X_sel, y_sel = X_sel.reset_index(drop=True), y_sel.reset_index(drop=True)
    X_te, y_te = X_te.reset_index(drop=True), y_te.reset_index(drop=True)
    ref = {c: X_sel[c].dropna().values.astype(float) for c in X_sel.select_dtypes(include="number").columns}
    n0 = len(X_sel)

    rows = []
    for seq in pipelines:
        Xc = G.apply_pipeline(X_sel, y_sel, seq, actions)
        if Xc is None or len(Xc) == 0: continue
        acc_in = W.inner_val_acc(Xc, y_sel, seed, "tabpfn")          # selection signal
        # held-out tuning-fold accuracy (different seed → different inner split) for leak-free tuning
        acc_tune = W.inner_val_acc(Xc, y_sel, seed + 1000, "tabpfn")
        if not (np.isfinite(acc_in) or np.isfinite(acc_tune)): continue
        miss = float(Xc.isna().mean().mean()); dup = float(Xc.duplicated().sum()) / max(len(Xc), 1)
        # sacred-test metrics
        Xtp = G.prepare_test_like_train(X_sel, X_te, seq)
        Xtr, ytr, le = G._encode_align(Xc, y_sel)
        shared = [c for c in Xc.select_dtypes(include="number").columns if c in Xtp.columns]
        try:
            yte = le.transform(np.asarray(y_te)); Xte = Xtp[shared].values.astype(float)
            panel = W.panel_on_test(Xtr, ytr, Xte, yte, seed) if (len(np.unique(ytr)) >= 2 and len(yte)) else {}
        except Exception:
            panel = {}
        rows.append({"dataset": ds, "seed": seed, "acc_in": acc_in, "acc_tune": acc_tune,
                     "ret": len(Xc) / n0, "Q": (1 - miss) * (1 - dup), "W1": drift_to_dirty(Xc, ref),
                     "test_logreg": panel.get("logreg", {}).get("acc", np.nan),
                     "test_tabpfn": panel.get("tabpfn", {}).get("acc", np.nan)})
    return rows


def pick(pool: pd.DataFrame, w, lam, power, by) -> float:
    """Select pipeline = argmax score(by), return its test_logreg accuracy."""
    a = pool[by].values
    s = w[0]*np.nan_to_num(a, nan=-1) + w[1]*(pool["ret"].values**power) + w[2]*pool["Q"].values - lam*pool["W1"].values
    i = int(np.argmax(s))
    return float(pool.iloc[i]["test_logreg"])


def main(seeds, max_pipelines, step, output_dir=None) -> None:
    out_dir = Path(output_dir) if output_dir else (Path(__file__).parents[1] / "outputs" / "paper_ready" / "saga_weighttune")
    out_dir.mkdir(parents=True, exist_ok=True)
    actions = G.build_actions(); allp = G.enumerate_valid_pipelines(max_len=3)
    grid = W.simplex_grid(step)
    pool_rows = []
    for ds in DATASETS:
        for seed in seeds:
            pls = G.sample_pipelines(allp, max_pipelines, seed)
            pool_rows += build_pool(ds, seed, pls, actions)
            pd.DataFrame(pool_rows).to_csv(out_dir / "saga_wt_pool.csv", index=False)
            print(f"  pooled {ds} s{seed}", flush=True)
    pool = pd.DataFrame(pool_rows)

    # Per dataset: fixed / oracle / leakfree-tune, × drift off/on. Average test_logreg over seeds.
    res = []
    for ds, g_ds in pool.groupby("dataset"):
        row = {"dataset": ds}
        for lam in LAMBDAS:
            tag = "drift" if lam > 0 else "nodrift"
            fixed, oracle, lftune = [], [], []
            for seed, g in g_ds.groupby("seed"):
                g = g.reset_index(drop=True)
                fixed.append(pick(g, FIXED, lam, 2.0, "acc_in"))
                # ORACLE: choose (w,power) maximising TEST accuracy (upper bound, not deployable)
                oracle.append(max(_score_to_testacc(g, w, lam, p, "test_logreg") for w, p in product(grid, POWERS)))
                # leak-free tune: pick (w,power) maximising held-out TUNING-fold accuracy, report on test
                besti = max((_score_to_testacc(g, w, lam, p, "acc_tune", return_idx=True) for w, p in product(grid, POWERS)),
                            key=lambda i: g.iloc[i]["acc_tune"])
                lftune.append(float(g.iloc[besti]["test_logreg"]))
            row[f"fixed_{tag}"] = np.nanmean(fixed)
            row[f"oracle_{tag}"] = np.nanmean(oracle)
            row[f"lftune_{tag}"] = np.nanmean(lftune)
        res.append(row)
    R = pd.DataFrame(res); R.to_csv(out_dir / "saga_weighttune_results.csv", index=False)
    print("\n=== SAGA weight-tuning (test LogReg acc; ORACLE = test-optimal upper bound) ===")
    print(f"{'dataset':14}{'SAGA':>6}{'fixed':>8}{'fix+dr':>8}{'lf-tune':>8}{'lf+dr':>8}{'ORACLE':>8}{'orc+dr':>8}")
    for _, r in R.iterrows():
        s = SAGA_PUBLISHED.get(r["dataset"], (None, None, None))[1]
        ss = " N/A" if s is None else f"{s:.2f}"
        print(f"{r['dataset']:14}{ss:>6}{r['fixed_nodrift']:>8.3f}{r['fixed_drift']:>8.3f}"
              f"{r['lftune_nodrift']:>8.3f}{r['lftune_drift']:>8.3f}{r['oracle_nodrift']:>8.3f}{r['oracle_drift']:>8.3f}")
    print(f"\nSaved → {out_dir}\nNOTE: ORACLE columns are upper bounds (weights chosen on the test set) — NOT deployable.")


def _score_to_testacc(g, w, lam, power, by, return_idx=False):
    a = g[by].values
    s = w[0]*np.nan_to_num(a, nan=-1) + w[1]*(g["ret"].values**power) + w[2]*g["Q"].values - lam*g["W1"].values
    i = int(np.argmax(s))
    return i if return_idx else float(g.iloc[i]["test_logreg"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2, 3, 4])
    ap.add_argument("--max-pipelines", type=int, default=18)
    ap.add_argument("--step", type=float, default=0.2)
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    main(tuple(a.seeds), a.max_pipelines, a.step, a.output_dir)
