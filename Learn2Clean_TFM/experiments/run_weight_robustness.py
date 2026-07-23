"""
experiments/run_weight_robustness.py

WR — Weight-robustness: fair R7 (TFMAwareReward) vs R3 (RF-reward) over the WHOLE weight
simplex + weight-free Pareto. Implements docs/protocols/weight_robustness_protocol.md.
Held-out protocol (sacred outer test / inner-val selection); MATCHED — the ONLY difference between
R3 and R7 is the accuracy estimator (RandomForest vs TabPFN v2).

Design (complete-metrics, single cache → all analyses free):
  Per (dataset, seed), for EVERY candidate pipeline, record ONE pool row:
    inner-val:   acc_rf_inner, acc_tab_inner               (selection signals)
    components:  retention, Q, W1_drift
    test panel:  for each evaluator E in {tabpfn, logreg, mlp, xgb} and the neutral
                 aggregate (logreg+mlp+xgb, i.e. excluding TabPFN's own family):
                   E_acc, E_f1, E_precision, E_recall, E_ece     on the sacred test
  → outputs/paper_ready/weight_robustness/pool_metrics.csv

Then pure analysis over the pool (no extra model calls):
  * Simplex sweep: per (weights, λ): R3 = argmax score with acc_rf_inner,
    R7 = argmax with acc_tab_inner; gap = test-metric(R7 pipe) − test-metric(R3 pipe).
    Win/tie/loss fractions over the simplex per metric; named-point readout
    (R7=(.50,.35,.15), R3=(.50,.30,.20)); HW1 concentration corr(|gap|, w_acc).
  * Pareto (weight-free): per cell, global front over the pool on
    (neutral_f1 ↑, retention ↑, W1 ↓, neutral_ece ↓); dominance rate of R7- vs
    R3-selected points; hypervolume (pymoo if available).
  → wr_per_config.csv, wr_summary.txt, wr_pareto.csv

Usage
-----
  PYTHONPATH=src:experiments python experiments/run_weight_robustness.py --seeds 42 1 2 3 4 5 6 7
"""

from __future__ import annotations

import argparse
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import run_c2_tfm_reward_nested as G
from learn2clean_v3.data.error_injection import ErrorProfile, apply_error_profile
from learn2clean_v3.data.openml_loader import BENCHMARK_DATASETS, load_dataset

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
try:
    from pymoo.indicators.hv import HV
    _HAS_HV = True
except Exception:
    _HAS_HV = False

LAMBDAS = [0.0, 0.025, 0.05, 0.10, 0.20]
NAMED = {"R7": (0.50, 0.35, 0.15), "R3": (0.50, 0.30, 0.20)}
TIE_EPS = 0.005
EVALUATORS = ["tabpfn", "logreg", "mlp"] + (["xgb"] if _HAS_XGB else [])
NEUTRAL = [e for e in EVALUATORS if e != "tabpfn"]
METRICS = ["acc", "f1", "precision", "recall", "ece"]


def simplex_grid(step: float = 0.1) -> List[Tuple[float, float, float]]:
    k = round(1 / step)
    return [(i / k, j / k, (k - i - j) / k) for i in range(k + 1) for j in range(k + 1 - i)]


# --------------------------------------------------------------------------- #
# Per-pipeline scoring (inner-val components + sacred-test panel)
# --------------------------------------------------------------------------- #
def inner_val_acc(X_clean, y, seed, eval_model) -> float:
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
            clf = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1).fit(Xtr, ytr)
            pred = clf.predict(Xval)
        else:
            pred, _ = G._tabpfn_fit_predict(Xtr, ytr, Xval, seed)
        return float(accuracy_score(yval, pred))
    except Exception:
        return float("nan")


def drift_to_dirty(X_clean, ref_cols) -> float:
    numeric = X_clean.select_dtypes(include="number")
    d = []
    for col in numeric.columns:
        if col not in ref_cols:
            continue
        cur, ref = numeric[col].dropna().values, ref_cols[col]
        if len(cur) < 2 or len(ref) < 2:
            continue
        try:
            d.append(min(float(stats.wasserstein_distance(cur, ref)) / (float(np.std(ref)) or 1.0), 5.0))
        except Exception:
            pass
    return float(np.mean(d)) if d else 0.0


def _clf_metrics(yte, pred, prob) -> Dict[str, float]:
    avg = "binary" if len(np.unique(yte)) == 2 else "macro"
    return {"acc": float(accuracy_score(yte, pred)),
            "f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
            "precision": float(precision_score(yte, pred, average=avg, zero_division=0)),
            "recall": float(recall_score(yte, pred, average=avg, zero_division=0)),
            "ece": G.compute_ece(yte, prob) if prob is not None else float("nan")}


def _nan_metrics() -> Dict[str, float]:
    return {m: float("nan") for m in METRICS}


def panel_on_test(Xtr, ytr, Xte, yte, seed) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    try:
        p, pr = G._tabpfn_fit_predict(Xtr, ytr, Xte, seed); out["tabpfn"] = _clf_metrics(yte, p, pr)
    except Exception:
        out["tabpfn"] = _nan_metrics()
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    for name, mk in [("logreg", lambda: LogisticRegression(max_iter=500, random_state=seed)),
                     ("mlp", lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=seed))]:
        try:
            clf = mk().fit(Xtr_s, ytr)
            pr = clf.predict_proba(Xte_s) if hasattr(clf, "predict_proba") else None
            out[name] = _clf_metrics(yte, clf.predict(Xte_s), pr)
        except Exception:
            out[name] = _nan_metrics()
    if _HAS_XGB:
        try:
            clf = XGBClassifier(n_estimators=100, max_depth=4, random_state=seed,
                                eval_metric="logloss", verbosity=0).fit(Xtr, ytr)
            out["xgb"] = _clf_metrics(yte, clf.predict(Xte), clf.predict_proba(Xte))
        except Exception:
            out["xgb"] = _nan_metrics()
    out["neutral"] = {m: float(np.nanmean([out[e][m] for e in NEUTRAL])) for m in METRICS}
    return out


def build_pool(ds_name, seed, pipelines, actions) -> List[Dict]:
    """One row per pipeline: inner components + full test panel (all metrics)."""
    try:
        X, y, _ = load_dataset(ds_name, use_cache=True)
    except Exception:
        return []
    if len(X) > G.SUBSAMPLE_CAP:
        X, _, y, _ = train_test_split(X, y, train_size=G.SUBSAMPLE_CAP, random_state=seed, stratify=y)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    Xd, yd = apply_error_profile(X, y, ErrorProfile("mcar", rate=G.MCAR_RATE, seed=seed))
    try:
        X_sel, X_test, y_sel, y_test = train_test_split(Xd, yd, test_size=G.OUTER_TEST_SIZE,
                                                        random_state=seed, stratify=yd)
    except ValueError:
        X_sel, X_test, y_sel, y_test = train_test_split(Xd, yd, test_size=G.OUTER_TEST_SIZE, random_state=seed)
    X_sel, y_sel = X_sel.reset_index(drop=True), y_sel.reset_index(drop=True)
    X_test, y_test = X_test.reset_index(drop=True), y_test.reset_index(drop=True)
    ref_cols = {c: X_sel[c].dropna().values.astype(float)
                for c in X_sel.select_dtypes(include="number").columns}
    n0 = len(X_sel)

    rows = []
    for seq in pipelines:
        Xc = G.apply_pipeline(X_sel, y_sel, seq, actions)
        if Xc is None or len(Xc) == 0:
            continue
        acc_rf = inner_val_acc(Xc, y_sel, seed, "rf")
        acc_tab = inner_val_acc(Xc, y_sel, seed, "tabpfn")
        if not (np.isfinite(acc_rf) or np.isfinite(acc_tab)):
            continue
        miss = float(Xc.isna().mean().mean())
        dup = float(Xc.duplicated().sum()) / max(len(Xc), 1)
        row = {"dataset": ds_name, "seed": seed, "pipe": G.pipeline_label(seq),
               "acc_rf_inner": acc_rf, "acc_tab_inner": acc_tab,
               "retention": len(Xc) / n0, "Q": (1 - miss) * (1 - dup),
               "W1": drift_to_dirty(Xc, ref_cols)}
        # test panel
        Xtp = G.prepare_test_like_train(X_sel, X_test, seq)
        Xtr, ytr, le = G._encode_align(Xc, y_sel)
        shared = [c for c in Xc.select_dtypes(include="number").columns if c in Xtp.columns]
        try:
            yte = le.transform(np.asarray(y_test)); Xte = Xtp[shared].values.astype(float)
            panel = panel_on_test(Xtr, ytr, Xte, yte, seed) if (len(np.unique(ytr)) >= 2 and len(yte)) else None
        except Exception:
            panel = None
        for fam in EVALUATORS + ["neutral"]:
            fm = (panel or {}).get(fam, _nan_metrics())
            for m in METRICS:
                row[f"{fam}_{m}"] = fm.get(m, float("nan"))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Analysis over the pool (sweep + Pareto) — pure arithmetic, no model calls
# --------------------------------------------------------------------------- #
def _argmax(pool: pd.DataFrame, w, lam, acc_col) -> Optional[int]:
    a = pool[acc_col].values
    score = w[0] * a + w[1] * pool["retention"].values + w[2] * pool["Q"].values - lam * pool["W1"].values
    score = np.where(np.isfinite(a), score, -np.inf)
    return int(np.argmax(score)) if np.isfinite(score).any() else None


def sweep(pool_df: pd.DataFrame, weights) -> pd.DataFrame:
    rows = []
    for (ds, seed), pool in pool_df.groupby(["dataset", "seed"]):
        pool = pool.reset_index(drop=True)
        for w, lam in product(weights, LAMBDAS):
            i3, i7 = _argmax(pool, w, lam, "acc_rf_inner"), _argmax(pool, w, lam, "acc_tab_inner")
            if i3 is None or i7 is None:
                continue
            r3, r7 = pool.iloc[i3], pool.iloc[i7]
            row = {"dataset": ds, "seed": seed, "w_acc": w[0], "w_ret": w[1], "w_qual": w[2],
                   "lambda": lam, "same_pipe": int(r3["pipe"] == r7["pipe"])}
            for fam in ["neutral", "tabpfn"]:
                for m in METRICS:
                    row[f"gap_{fam}_{m}"] = r7[f"{fam}_{m}"] - r3[f"{fam}_{m}"]
            rows.append(row)
    return pd.DataFrame(rows)


def _pareto_mask(M: np.ndarray) -> np.ndarray:
    """M: maximize all columns. Return boolean mask of non-dominated rows."""
    n = len(M); keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i != j and np.all(M[j] >= M[i]) and np.any(M[j] > M[i]):
                keep[i] = False; break
    return keep


def pareto(pool_df: pd.DataFrame, weights) -> pd.DataFrame:
    """Per cell: global front over the pool on (neutral_f1↑, retention↑, −W1↑, −neutral_ece↑);
    dominance of R7- vs R3-selected pipelines (selection at the named points + a mid weight)."""
    rows = []
    probe = list(NAMED.values()) + [(1/3, 1/3, 1/3)]
    for (ds, seed), pool in pool_df.groupby(["dataset", "seed"]):
        pool = pool.reset_index(drop=True)
        obj = np.column_stack([pool["neutral_f1"].values, pool["retention"].values,
                               -pool["W1"].values, -pool["neutral_ece"].values])
        valid = np.all(np.isfinite(obj), axis=1)
        if valid.sum() < 2:
            continue
        front = np.zeros(len(pool), bool); front[np.where(valid)[0][_pareto_mask(obj[valid])]] = True
        hv = float("nan")
        if _HAS_HV:
            ref = obj[valid].min(axis=0) - 0.01
            try:
                hv = float(HV(ref_point=-ref)(-obj[valid][_pareto_mask(obj[valid])]))
            except Exception:
                pass
        for w in probe:
            for lam in [0.05]:
                i3, i7 = _argmax(pool, w, lam, "acc_rf_inner"), _argmax(pool, w, lam, "acc_tab_inner")
                if i3 is None or i7 is None:
                    continue
                rows.append({"dataset": ds, "seed": seed, "w": str(w), "lambda": lam,
                             "r3_on_front": int(front[i3]), "r7_on_front": int(front[i7]),
                             "front_size": int(front.sum()), "hv": hv})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, par: pd.DataFrame, out_dir: Path) -> str:
    L = ["WEIGHT-ROBUSTNESS SUMMARY (held-out protocol, matched; only estimator differs)\n"]
    for fam in ["neutral", "tabpfn"]:
        for m in ["acc", "f1"]:
            g = df[f"gap_{fam}_{m}"].dropna()
            if not len(g):
                continue
            win = float((g > TIE_EPS).mean()); loss = float((g < -TIE_EPS).mean())
            L.append(f"[{fam} {m}] {len(g)} configs: R7>R3 {win:.0%} | tie {1-win-loss:.0%} | "
                     f"R7<R3 {loss:.0%} | mean {g.mean():+.4f}")
    for name, (wa, wr, wq) in NAMED.items():
        sub = df[np.isclose(df.w_acc, wa) & np.isclose(df.w_ret, wr) & np.isclose(df.w_qual, wq)]
        if len(sub):
            L.append(f"named {name}=({wa},{wr},{wq}): neutral-acc gap {sub['gap_neutral_acc'].mean():+.4f} (n={len(sub)})")
    gg = df.dropna(subset=["gap_neutral_acc"])
    if len(gg) > 10:
        L.append(f"HW1 concentration corr(|gap|,w_acc) = {np.corrcoef(gg.w_acc, gg.gap_neutral_acc.abs())[0,1]:+.3f}")
    if len(par):
        L.append(f"\n[Pareto] R7 on front {par.r7_on_front.mean():.0%} of probes | "
                 f"R3 on front {par.r3_on_front.mean():.0%} | HV available={_HAS_HV}")
    txt = "\n".join(L)
    (out_dir / "wr_summary.txt").write_text(txt)
    return txt


# --------------------------------------------------------------------------- #
def main(dataset_names=None, output_dir=None, seeds=(42,), max_pipelines=18, step=0.1) -> None:
    if dataset_names is None:
        dataset_names = list(BENCHMARK_DATASETS.keys())
    out_dir = Path(output_dir) if output_dir else (
        Path(__file__).parents[1] / "outputs" / "paper_ready" / "weight_robustness")
    out_dir.mkdir(parents=True, exist_ok=True)

    actions = G.build_actions()
    all_pipelines = G.enumerate_valid_pipelines(max_len=3)
    weights = simplex_grid(step)
    print(f"evaluators={EVALUATORS} | simplex={len(weights)}×λ{len(LAMBDAS)} | xgb={_HAS_XGB} hv={_HAS_HV}", flush=True)

    pool_rows: List[Dict] = []
    t0 = time.time()
    for ds in dataset_names:
        for seed in seeds:
            pipelines = G.sample_pipelines(all_pipelines, max_pipelines, seed)
            ts = time.time()
            rows = build_pool(ds, seed, pipelines, actions)
            pool_rows.extend(rows)
            pd.DataFrame(pool_rows).to_csv(out_dir / "pool_metrics.csv", index=False)
            print(f"  {ds:16} s{seed}: pool={len(rows)} pipelines ({time.time()-ts:.0f}s)", flush=True)

    if not pool_rows:
        print("No pool rows."); return
    pool_df = pd.DataFrame(pool_rows)
    cfg = sweep(pool_df, weights); cfg.to_csv(out_dir / "wr_per_config.csv", index=False)
    par = pareto(pool_df, weights); par.to_csv(out_dir / "wr_pareto.csv", index=False)
    print("\n" + summarize(cfg, par, out_dir))
    print(f"\nWall-clock: {(time.time()-t0)/60:.1f} min → {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--max-pipelines", type=int, default=18)
    ap.add_argument("--step", type=float, default=0.1)
    a = ap.parse_args()
    main(a.datasets, a.output_dir, tuple(a.seeds), a.max_pipelines, a.step)
