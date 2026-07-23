"""
experiments/run_scalability.py

S — R7 scalability (docs/protocols/r7_scalability_protocol.md). Measures how each REWARD
TERM scales with rows n and columns p, and the R7-vs-R3 per-call speedup. Synthetic data
(make_classification) with a fixed corruption profile held constant across sizes, so scaling
is isolated from task difficulty. Timing: perf_counter, warmup-excluded repeats, median+IQR.

Terms timed per reward call (context cap c=512):
  R7.accuracy  : TabPFN v2 fit+predict on ≤c subsampled context   → predicted ~flat in n (HS1)
  R3.accuracy  : RandomForest fit on full n (NOT cap-able)        → grows ~n log n (HS4)
  W1.drift     : per-column Wasserstein-1                          → O(p·n log n) (HS2)
  quality      : missing-rate + exact dup-rate                    → O(n·p)
Speedup S(n)=T_R3/T_R7 should grow with n (HS4). Fitted log-log exponent a per term (+R²).

Experiments: S1 row scaling (sweep n, p=50), S2 column scaling (sweep p, n=fixed).

Usage
-----
  PYTHONPATH=src:experiments python experiments/run_scalability.py
  PYTHONPATH=src:experiments python experiments/run_scalability.py --n-grid 1000 5000 20000 100000 --repeats 5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

import run_c2_tfm_reward_nested as G

CONTEXT_CAP = 512


def _inject(X: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = X.copy()
    # fixed corruption profile: 10% missing cells + 5% outliers (held constant across sizes)
    n, p = X.shape
    mask = rng.random((n, p)) < 0.10
    X[mask] = np.nan
    om = rng.random((n, p)) < 0.05
    X[om] = X[om] if not om.any() else (np.nanmean(X) + 5 * np.nanstd(X) * rng.standard_normal(om.sum()))
    return X


def _median_iqr(times: List[float]) -> tuple:
    a = np.array(times)
    return float(np.median(a)), float(np.percentile(a, 75) - np.percentile(a, 25))


def time_terms(n: int, p: int, seed: int, repeats: int) -> Dict[str, float]:
    X, y = make_classification(n_samples=n, n_features=p, n_informative=max(2, p // 2),
                               n_classes=2, random_state=seed)
    X = _inject(X, seed)
    Xdf = pd.DataFrame(X, columns=[f"c{i}" for i in range(p)])
    ref = {c: pd.Series(X[:, i]).dropna().values for i, c in enumerate(Xdf.columns)}
    # impute for the model-based terms (TabPFN handles NaN; RF needs imputation)
    Ximp = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)

    def t_tabpfn():
        idx = np.random.default_rng(seed).choice(n, size=min(CONTEXT_CAP, n), replace=False)
        Xtr, ytr = Ximp[idx], y[idx]
        try:
            G._tabpfn_fit_predict(Xtr, ytr, Ximp[:min(256, n)], seed)
        except Exception:
            pass

    def t_rf():
        RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1).fit(Ximp, y)

    def t_w1():
        for i, c in enumerate(Xdf.columns):
            cur = Xdf[c].dropna().values
            if len(cur) >= 2 and len(ref[c]) >= 2:
                stats.wasserstein_distance(cur[:200000], ref[c][:200000])

    def t_quality():
        float(Xdf.isna().mean().mean()); float(Xdf.duplicated().sum())

    out = {}
    for name, fn in [("R7_acc_tabpfn", t_tabpfn), ("R3_acc_rf", t_rf), ("W1_drift", t_w1), ("quality", t_quality)]:
        fn()  # warmup (excluded)
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        med, iqr = _median_iqr(ts)
        out[name], out[name + "_iqr"] = med, iqr
    return out


def fit_exponent(ns: np.ndarray, lat: np.ndarray) -> tuple:
    m = (lat > 0) & np.isfinite(lat)
    if m.sum() < 2:
        return float("nan"), float("nan")
    a, b = np.polyfit(np.log(ns[m]), np.log(lat[m]), 1)
    pred = a * np.log(ns[m]) + b
    ss = 1 - np.sum((np.log(lat[m]) - pred) ** 2) / np.sum((np.log(lat[m]) - np.log(lat[m]).mean()) ** 2)
    return float(a), float(ss)


def main(n_grid, p_grid, p_fixed, n_fixed, repeats, seeds, output_dir=None) -> None:
    out_dir = Path(output_dir) if output_dir else (Path(__file__).parents[1] / "outputs" / "paper_ready" / "scalability")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    print("=== S1 row scaling (p=%d) ===" % p_fixed, flush=True)
    for n in n_grid:
        for seed in seeds:
            r = time_terms(n, p_fixed, seed, repeats); r.update({"exp": "S1_rows", "n": n, "p": p_fixed, "seed": seed})
            rows.append(r); pd.DataFrame(rows).to_csv(out_dir / "scalability_raw.csv", index=False)
        agg = pd.DataFrame([x for x in rows if x["exp"] == "S1_rows" and x["n"] == n])
        sp = agg["R3_acc_rf"].median() / max(agg["R7_acc_tabpfn"].median(), 1e-9)
        print(f"  n={n:>7}: tabpfn={agg['R7_acc_tabpfn'].median()*1e3:7.1f}ms rf={agg['R3_acc_rf'].median()*1e3:8.1f}ms "
              f"W1={agg['W1_drift'].median()*1e3:7.1f}ms  speedup(R3/R7)={sp:5.1f}×", flush=True)
    print("=== S2 column scaling (n=%d) ===" % n_fixed, flush=True)
    for p in p_grid:
        for seed in seeds:
            r = time_terms(n_fixed, p, seed, repeats); r.update({"exp": "S2_cols", "n": n_fixed, "p": p, "seed": seed})
            rows.append(r); pd.DataFrame(rows).to_csv(out_dir / "scalability_raw.csv", index=False)

    df = pd.DataFrame(rows)
    # Fitted exponents (S1, vs n)
    s1 = df[df.exp == "S1_rows"].groupby("n").median(numeric_only=True)
    print("\n=== FITTED SCALING EXPONENTS (log-log slope a, R²) over n ===")
    expo = {}
    for term in ["R7_acc_tabpfn", "R3_acc_rf", "W1_drift", "quality"]:
        a, r2 = fit_exponent(s1.index.values.astype(float), s1[term].values)
        expo[term] = (a, r2)
        print(f"  {term:16} a={a:+.3f}  R²={r2:.3f}   (HS: tabpfn≈0, rf≈1, W1≈1)")
    pd.DataFrame([{"term": k, "exponent_a": v[0], "r2": v[1]} for k, v in expo.items()]).to_csv(
        out_dir / "scalability_exponents.csv", index=False)
    print(f"\nSaved → {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-grid", nargs="+", type=int, default=[1000, 5000, 20000, 100000])
    ap.add_argument("--p-grid", nargs="+", type=int, default=[10, 50, 100, 200])
    ap.add_argument("--p-fixed", type=int, default=50)
    ap.add_argument("--n-fixed", type=int, default=20000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    main(a.n_grid, a.p_grid, a.p_fixed, a.n_fixed, a.repeats, tuple(a.seeds), a.output_dir)
