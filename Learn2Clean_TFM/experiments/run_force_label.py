"""
experiments/run_force_label.py

force_label follow-up — disentangles "label cleaning doesn't help" from "label cleaning
helps but the noisy validation can't SELECT it." Motivated by the sweep finding: under
label noise the inner-val accuracy signal is itself corrupted, so the reward never picks
the mislabel-removal operator (label_used≈0). Here we BYPASS the reward and FORCE label
cleaning, then measure downstream test accuracy on the clean test labels.

Held-out protocol: label noise injected on training labels (y_sel) only; test labels are ground
truth. Dose-response over label-noise rate, all 13 datasets (OpenML 10 + SAGA 3).

Arms:
  no_clean    : raw noisy D_sel → TabPFN
  R7          : TabPFN-reward over base ops (reward-selected; does NOT pick label cleaning)
  force_lblO  : FORCE LabelCleaner only (remove likely-mislabeled rows), then TabPFN
  force_lbl   : FORCE LabelCleaner, then R7-select a base pipeline on the cleaned set
Key gap: force_lbl(O) − no_clean under label noise ⇒ does removing in-context label noise
help TabPFN, independent of whether the reward could find it?

Usage
-----
  PYTHONPATH=src:experiments python experiments/run_force_label.py --datasets EEG Titanic --seeds 42 1 2
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import run_c2_tfm_reward_nested as G
from run_corruption_sweep import (BASE, BASE_GROUP, LabelCleaner, enum_pipes, eval_arm,
                                  inject_label_noise, load_any, select)
from sklearn.model_selection import train_test_split
from learn2clean_v3.data.openml_loader import BENCHMARK_DATASETS
from run_saga_comparison import DATASETS as SAGA_DS

RATES = [0.0, 0.10, 0.20, 0.35]


def run_one(ds, rate, seed, base_pipes) -> Optional[Dict]:
    loaded = load_any(ds)
    if loaded is None:
        return None
    X, y = loaded
    if len(X) > G.SUBSAMPLE_CAP:
        X, _, y, _ = train_test_split(X, y, train_size=G.SUBSAMPLE_CAP, random_state=seed, stratify=y)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    try:
        X_sel, X_test, y_sel, y_test = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
    except ValueError:
        X_sel, X_test, y_sel, y_test = train_test_split(X, y, test_size=0.30, random_state=seed)
    X_sel, y_sel = X_sel.reset_index(drop=True), y_sel.reset_index(drop=True)
    X_test, y_test = X_test.reset_index(drop=True), y_test.reset_index(drop=True)
    if rate > 0:
        y_sel = inject_label_noise(y_sel, rate, seed)

    row = {"dataset": ds, "rate": rate, "seed": seed}
    # no_clean + reward-selected R7
    row.update({f"no_clean_{k}": v for k, v in
                {"tabpfn": eval_arm(X_sel, y_sel, X_test, y_test, (), BASE, seed)["tabpfn"]}.items()})
    best_r7 = select(X_sel, y_sel, base_pipes, BASE, seed, "tabpfn")
    row["R7_tabpfn"] = eval_arm(X_sel, y_sel, X_test, y_test, best_r7, BASE, seed)["tabpfn"]

    # FORCE label cleaning (bypass reward)
    X_lc = LabelCleaner()(X_sel, y_sel)
    try:
        y_lc = y_sel.loc[X_lc.index]
    except Exception:
        y_lc = y_sel.iloc[: len(X_lc)]
    X_lc, y_lc = X_lc.reset_index(drop=True), pd.Series(np.asarray(y_lc)).reset_index(drop=True)
    row["frac_removed"] = 1.0 - len(X_lc) / max(len(X_sel), 1)
    row["force_lblO_tabpfn"] = eval_arm(X_lc, y_lc, X_test, y_test, (), BASE, seed)["tabpfn"]
    best_fl = select(X_lc, y_lc, base_pipes, BASE, seed, "tabpfn")
    row["force_lbl_tabpfn"] = eval_arm(X_lc, y_lc, X_test, y_test, best_fl, BASE, seed)["tabpfn"]
    return row


def main(datasets, seeds, max_pipelines, output_dir=None) -> None:
    out_dir = Path(output_dir) if output_dir else (Path(__file__).parents[1] / "outputs" / "paper_ready" / "force_label")
    out_dir.mkdir(parents=True, exist_ok=True)
    base_pipes = G.sample_pipelines(enum_pipes(BASE_GROUP), max_pipelines, 42)
    rows: List[Dict] = []; t0 = time.time()
    for ds in datasets:
        for rate in RATES:
            for seed in seeds:
                r = run_one(ds, rate, seed, base_pipes)
                if r:
                    rows.append(r)
                pd.DataFrame(rows).to_csv(out_dir / "force_label_per_run.csv", index=False)
            sub = [x for x in rows if x["dataset"] == ds and x["rate"] == rate]
            if sub:
                nc = np.nanmean([x["no_clean_tabpfn"] for x in sub])
                flo = np.nanmean([x["force_lblO_tabpfn"] for x in sub])
                print(f"  {ds:14} label@{rate}: no_clean={nc:.4f} force_lblOnly={flo:.4f} "
                      f"Δ={flo-nc:+.4f} removed={np.nanmean([x['frac_removed'] for x in sub]):.1%}", flush=True)
    df = pd.DataFrame(rows)
    if len(df):
        agg = df.groupby("rate").agg(no_clean=("no_clean_tabpfn", "mean"), R7=("R7_tabpfn", "mean"),
                                     force_lblO=("force_lblO_tabpfn", "mean"), force_lbl=("force_lbl_tabpfn", "mean"),
                                     frac_removed=("frac_removed", "mean")).reset_index()
        agg["gap_forcelbl"] = agg["force_lblO"] - agg["no_clean"]
        agg.to_csv(out_dir / "force_label_by_rate.csv", index=False)
        print("\n=== FORCE-LABEL by noise rate (TabPFN acc; all datasets) ===")
        print(agg.to_string(index=False))
        print("\nKEY: gap_forcelbl>0 under noise ⇒ forced label-cleaning helps (reward just can't select it).")
    print(f"\nWall-clock {(time.time()-t0)/60:.1f} min → {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(BENCHMARK_DATASETS.keys()) + list(SAGA_DS.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2, 3, 4])
    ap.add_argument("--max-pipelines", type=int, default=12)
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    main(a.datasets, tuple(a.seeds), a.max_pipelines, a.output_dir)
