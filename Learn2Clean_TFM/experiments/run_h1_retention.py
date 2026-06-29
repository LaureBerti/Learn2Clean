"""

H1 — does R7's retention/context term open a GENUINE leak-free gap?
Mechanism: TabPFN is in-context, so deleting rows hurts it more than RF. R7 penalises
row deletion quadratically (context_penalty_power=2); R3 linearly. This only matters when
cleaning actually REMOVES rows — so we inject OUTLIERS (not MCAR) at increasing rates so
outlier-removal pipelines shrink the context, on SMALL datasets (every row matters for ICL),
with NO subsampling (full cleaned set is the TabPFN context).

Pre-registered prediction: as the outlier rate rises, R3 (linear retention) over-prunes rows,
R7 (quadratic retention) keeps more context, and R7's final TabPFN accuracy pulls ahead of R3.
If the gap stays flat/zero, the retention term does NOT rescue R7 — also a clean result.

Three arms to DECOMPOSE any gap (matched everything except the named factor):
  R3       : RF estimator,    retention^1, drift 0.10   (paper baseline)
  R7lin    : TabPFN estimator, retention^1, drift 0.05   (isolates estimator)
  R7       : TabPFN estimator, retention^2, drift 0.05   (paper R7; adds the retention term)
  → R7 − R7lin isolates the RETENTION term; R7lin − R3 isolates the ESTIMATOR.
All evaluated leak-free (sacred outer test), final accuracy with TabPFN; retention of the
selected pipeline is logged to confirm the mechanism (does R7 keep more rows?).

Usage
-----
  PYTHONPATH=src:experiments python experiments/run_h1_retention.py --seeds 42 1 2 3 4 5 6 7
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

import run_c2_tfm_reward_nested as G
import run_weight_robustness as W
from learn2clean_v3.data.error_injection import ErrorProfile, apply_error_profile
from learn2clean_v3.data.openml_loader import load_dataset

# Small datasets where retention/context binds (all < 1024 train rows → no cap active).
SMALL = ["hepatitis", "heart_statlog", "ionosphere", "blood_transfusion", "diabetes"]
OUTLIER_RATES = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
ARMS = {  # name: (estimator, retention_power, drift_coeff)
    "R3":    ("rf",     1.0, 0.10),
    "R7lin": ("tabpfn", 1.0, 0.05),
    "R7":    ("tabpfn", 2.0, 0.05),
}


def drift_to_dirty(Xc, ref_cols):
    num = Xc.select_dtypes(include="number"); d = []
    for c in num.columns:
        if c not in ref_cols: continue
        cur, ref = num[c].dropna().values, ref_cols[c]
        if len(cur) < 2 or len(ref) < 2: continue
        try: d.append(min(float(stats.wasserstein_distance(cur, ref)) / (float(np.std(ref)) or 1.0), 5.0))
        except Exception: pass
    return float(np.mean(d)) if d else 0.0


def select(X_sel, y_sel, pipelines, actions, seed, estimator, power, drift_coeff, ref_cols) -> Tuple:
    """Return (best_seq, retention_of_best)."""
    n0 = len(X_sel); best, best_s, best_ret = (), -np.inf, 1.0
    for seq in pipelines:
        Xc = G.apply_pipeline(X_sel, y_sel, seq, actions)
        if Xc is None or len(Xc) == 0: continue
        acc = W.inner_val_acc(Xc, y_sel, seed, estimator)
        if not np.isfinite(acc): continue
        ret_raw = len(Xc) / n0
        miss = float(Xc.isna().mean().mean()); dup = float(Xc.duplicated().sum()) / max(len(Xc), 1)
        qual = (1 - miss) * (1 - dup)
        score = 0.50 * acc + 0.35 * (ret_raw ** power) + 0.15 * qual - drift_coeff * drift_to_dirty(Xc, ref_cols)
        if score > best_s:
            best_s, best, best_ret = score, seq, ret_raw
    return best, best_ret


def run_one(ds, rate, seed, pipelines, actions) -> Optional[Dict]:
    try:
        X, y, _ = load_dataset(ds, use_cache=True)
    except Exception:
        return None
    # NO subsampling (small datasets) — inject OUTLIERS at the swept rate (row-removal trigger)
    if rate > 0:
        Xd, yd = apply_error_profile(X, y, ErrorProfile("outlier", rate=rate, k=5.0))
    else:
        Xd, yd = X.copy(), y.copy()
    try:
        X_sel, X_test, y_sel, y_test = train_test_split(Xd, yd, test_size=0.30, random_state=seed, stratify=yd)
    except ValueError:
        X_sel, X_test, y_sel, y_test = train_test_split(Xd, yd, test_size=0.30, random_state=seed)
    X_sel, y_sel = X_sel.reset_index(drop=True), y_sel.reset_index(drop=True)
    X_test, y_test = X_test.reset_index(drop=True), y_test.reset_index(drop=True)
    ref_cols = {c: X_sel[c].dropna().values.astype(float) for c in X_sel.select_dtypes(include="number").columns}

    row = {"dataset": ds, "rate": rate, "seed": seed}
    for name, (est, power, dc) in ARMS.items():
        best, ret = select(X_sel, y_sel, pipelines, actions, seed, est, power, dc, ref_cols)
        Xc = G.apply_pipeline(X_sel, y_sel, best, actions)
        Xtp = G.prepare_test_like_train(X_sel, X_test, best)
        acc, ece = G.final_test_tabpfn(Xc, y_sel, Xtp, y_test, seed)
        row[f"{name}_acc"], row[f"{name}_ret"] = acc, ret
    row["gap_R7_R3"] = row["R7_acc"] - row["R3_acc"]
    row["gap_retention_term"] = row["R7_acc"] - row["R7lin_acc"]   # isolates retention
    row["gap_estimator"] = row["R7lin_acc"] - row["R3_acc"]        # isolates estimator
    return row


def main(seeds, max_pipelines, output_dir=None) -> None:
    out_dir = Path(output_dir) if output_dir else (Path(__file__).parents[1] / "outputs" / "paper_ready" / "h1_retention")
    out_dir.mkdir(parents=True, exist_ok=True)
    actions = G.build_actions(); all_p = G.enumerate_valid_pipelines(max_len=3)
    rows: List[Dict] = []; t0 = time.time()
    for ds in SMALL:
        for rate in OUTLIER_RATES:
            for seed in seeds:
                pls = G.sample_pipelines(all_p, max_pipelines, seed)
                r = run_one(ds, rate, seed, pls, actions)
                if r: rows.append(r)
                pd.DataFrame(rows).to_csv(out_dir / "h1_per_run.csv", index=False)
            sub = [x for x in rows if x["dataset"] == ds and x["rate"] == rate]
            if sub:
                g = np.nanmean([x["gap_R7_R3"] for x in sub]); rt = np.nanmean([x["R7_ret"] - x["R3_ret"] for x in sub])
                print(f"  {ds:16} rate={rate:.2f}: gap(R7-R3)={g:+.4f}  Δretention(R7-R3)={rt:+.3f}", flush=True)
    df = pd.DataFrame(rows)
    # H1 verdict: does gap grow with rate? regress gap on rate.
    print("\n=== H1 — gap(R7-R3) vs outlier rate (pooled over small datasets, 8 seeds) ===")
    by_rate = df.groupby("rate").agg(gap_R7_R3=("gap_R7_R3","mean"), gap_retention=("gap_retention_term","mean"),
                                     gap_estimator=("gap_estimator","mean"),
                                     dRet=("R7_ret","mean")).reset_index()
    by_rate["R3_ret"] = df.groupby("rate")["R3_ret"].mean().values
    print(by_rate.to_string(index=False))
    if df["rate"].nunique() > 2:
        sl = np.polyfit(df["rate"], df["gap_R7_R3"].fillna(0), 1)[0]
        rho, p = stats.spearmanr(df["rate"], df["gap_R7_R3"].fillna(0))
        print(f"\ngap(R7-R3) vs rate: slope={sl:+.4f}/unit-rate | Spearman ρ={rho:+.3f} p={p:.3f}")
        print("PREDICTION: positive slope ⇒ retention term opens a genuine gap as corruption rises.")
    by_rate.to_csv(out_dir / "h1_by_rate.csv", index=False)
    print(f"\nWall-clock {(time.time()-t0)/60:.1f} min → {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--max-pipelines", type=int, default=18)
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    main(tuple(a.seeds), a.max_pipelines, a.output_dir)
