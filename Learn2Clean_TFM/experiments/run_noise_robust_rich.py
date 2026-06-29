"""

Noise-robust R7 variants on the RICH operator pool (+ label-clean toggle), under label noise,
8 seeds, imbalanced datasets, accuracy AND macro-F1 on a SACRED CLEAN test.

EFFICIENT DESIGN: enumerate a SHARED set of candidate pipelines once; for each candidate fit TabPFN
ONCE on the cleaned train and read off ALL proba-based signals (acc/F1/conf/margin on noisy + clean
val) plus one RF fit and the (cheap) prior-distance. Every reward arm then = argmax of its own signal
over the SAME candidate set (a fair comparison), and winners are deployed on TabPFN to the clean test.
(The earlier per-arm greedy was ~10x slower; `iterrf` imputer dropped as the heaviest, marginal op.)

Arms (signal in parens), all deployed on TabPFN, scored on CLEAN test:
  R3 (rf-acc) · R7acc (TabPFN acc) · R7F1 (TabPFN macro-F1) · R7conf (confidence, label-free)
  · R7margin (margin, label-free) · R7prior (-(M2+M3) prior-distance, label-free)
  · oracle (acc on TRUE val) · oracleF1 (F1 on TRUE val)   [ceilings]

Usage: PYTHONPATH=src:experiments python experiments/run_noise_robust_rich.py \
          --datasets blood_transfusion credit_g hepatitis --rates 0.0 0.1 0.2 0.35 --seeds 42 1 2 3 4 5 6 7
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
import run_saga_richops as R
import run_c2_tfm_reward_nested as G
import run_prior_distance as PD
from run_corruption_sweep import inject_label_noise

R.IMPUTE = [x for x in R.IMPUTE if x != "iterrf"]          # drop the heaviest (RandomForest) imputer
ARMS = {"R3": "rf", "R7acc": "acc_noisy", "R7F1": "f1_noisy", "R7conf": "conf", "R7margin": "margin",
        "R7prior": "prior", "oracle": "acc_clean", "oracleF1": "f1_clean"}
K_CANDIDATES = 16


def labelclean(tr, ytr2, seed):
    try:
        yv = ytr2.values
        if len(np.unique(yv)) < 2 or len(tr) < 30:
            return tr, ytr2
        proba = cross_val_predict(LogisticRegression(max_iter=400), tr.values, yv, cv=3, method="predict_proba")
        classes = np.unique(yv); pred = classes[proba.argmax(1)]; conf = proba.max(1)
        keep = ~((pred != yv) & (conf > 0.70))
        if keep.sum() >= max(20, int(0.5 * len(tr))):
            return tr.iloc[keep], ytr2.iloc[keep]
    except Exception:
        pass
    return tr, ytr2


def apply_rich(X_sel, y_sel, X_other, pipe, do_lbl, seed):
    out = R.apply_pipeline(X_sel, y_sel, X_other, pipe, seed)
    if out is None:
        return None
    tr, ytr2, te = out
    if do_lbl:
        tr, ytr2 = labelclean(tr.reset_index(drop=True), ytr2.reset_index(drop=True), seed)
    return tr, ytr2, te


def default_pipe():
    return tuple(None if None in o else o[0] for _, o in R.groups_for(True))


def sample_candidates(seed):
    pool = R.enumerate_pool(True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), min(K_CANDIDATES, len(pool)), replace=False)
    pipes = [default_pipe()] + [pool[i] for i in idx]
    cands = []
    for p in pipes:
        cands += [(p, False), (p, True)]
    return cands


def eval_val(cand, X_sel, y_sel, X_val, y_vn, y_vc, seed):
    """One cleaned-train fit → all selection signals on inner-val. None if invalid."""
    out = apply_rich(X_sel, y_sel, X_val, cand[0], cand[1], seed)
    if out is None or len(out[0]) == 0:
        return None
    tr, ytr2, val = out
    if len(np.unique(ytr2.values)) < 2 or tr.shape[1] == 0:
        return None
    d = {}
    try:
        Z = PD.standardize(tr.select_dtypes(include="number")); d["prior"] = -(PD.m_mmd(Z, seed) + PD.m_marg_w1(Z))
    except Exception:
        d["prior"] = -np.inf
    try:
        clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1).fit(tr.values, ytr2.values)
        d["rf"] = accuracy_score(y_vn.values, clf.predict(val.values))
    except Exception:
        d["rf"] = -np.inf
    try:
        pred, prob = G._tabpfn_fit_predict(tr.values, ytr2.values, val.values, seed)
        d["acc_noisy"] = accuracy_score(y_vn.values, pred); d["acc_clean"] = accuracy_score(y_vc.values, pred)
        d["f1_noisy"] = f1_score(y_vn.values, pred, average="macro"); d["f1_clean"] = f1_score(y_vc.values, pred, average="macro")
        if prob.ndim == 2:
            d["conf"] = float(prob.max(1).mean()); s = np.sort(prob, 1)
            d["margin"] = float((s[:, -1] - s[:, -2]).mean()) if prob.shape[1] >= 2 else d["conf"]
        else:
            d["conf"] = d["margin"] = -np.inf
    except Exception:
        for k in ("acc_noisy", "acc_clean", "f1_noisy", "f1_clean", "conf", "margin"):
            d[k] = -np.inf
    return d


def deploy(cand, X_sel, y_sel, X_test, y_test, seed):
    out = apply_rich(X_sel, y_sel, X_test, cand[0], cand[1], seed)
    if out is None or len(out[0]) == 0:
        return np.nan, np.nan
    tr, ytr2, te = out
    try:
        if len(np.unique(ytr2.values)) < 2 or tr.shape[1] == 0:
            return np.nan, np.nan
        pred, _ = G._tabpfn_fit_predict(tr.values, ytr2.values, te.values, seed)
        return float(accuracy_score(y_test.values, pred)), float(f1_score(y_test.values, pred, average="macro"))
    except Exception:
        return np.nan, np.nan


def run_one(name, rate, seed):
    X, y = R.load_ds(name)
    if len(X) > G.SUBSAMPLE_CAP:
        X, _, y, _ = train_test_split(X, y, train_size=G.SUBSAMPLE_CAP, random_state=0, stratify=y)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    X_sel, X_val, y_sel_c, y_val_c = train_test_split(X_tr, y_tr, test_size=0.25, random_state=seed, stratify=y_tr)
    X_sel, X_val = X_sel.reset_index(drop=True), X_val.reset_index(drop=True)
    y_sel_c, y_val_c = y_sel_c.reset_index(drop=True), y_val_c.reset_index(drop=True)
    X_test, y_test = X_test.reset_index(drop=True), y_test.reset_index(drop=True)
    y_sel = inject_label_noise(y_sel_c, rate, seed); y_val_n = inject_label_noise(y_val_c, rate, seed + 1)

    cands = sample_candidates(seed)
    ev = {c: eval_val(c, X_sel, y_sel, X_val, y_val_n, y_val_c, seed) for c in cands}
    ev = {c: e for c, e in ev.items() if e is not None}
    row = {"dataset": name, "rate": rate, "seed": seed, "n_cand": len(ev)}
    dcache = {}
    if ev:
        for arm, sig in ARMS.items():
            best = max(ev, key=lambda c: ev[c].get(sig, -np.inf))
            if best not in dcache:
                dcache[best] = deploy(best, X_sel, y_sel, X_test, y_test, seed)
            row[f"{arm}_acc"], row[f"{arm}_f1"] = dcache[best]; row[f"{arm}_uselbl"] = int(best[1])
    nc = deploy((default_pipe(), False), X_sel, y_sel, X_test, y_test, seed)
    row["no_clean_acc"], row["no_clean_f1"] = nc
    return row


def main(datasets, rates, seeds, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in datasets:
        for rate in rates:
            for seed in seeds:
                try:
                    r = run_one(ds, rate, seed)
                except Exception as e:
                    r = {"dataset": ds, "rate": rate, "seed": seed, "err": repr(e)[:140]}
                rows.append(r); pd.DataFrame(rows).to_csv(out / "noise_rich_per_run.csv", index=False)
                print(f"  {ds:12} r{rate} s{seed}: R7acc={r.get('R7acc_f1',np.nan):.3f} R7F1={r.get('R7F1_f1',np.nan):.3f} "
                      f"R7conf={r.get('R7conf_f1',np.nan):.3f} R7prior={r.get('R7prior_f1',np.nan):.3f} "
                      f"oracleF1={r.get('oracleF1_f1',np.nan):.3f} (F1)", flush=True)
    df = pd.DataFrame(rows)
    agg = df.groupby("rate").mean(numeric_only=True).reset_index()
    agg.to_csv(out / "noise_rich_by_rate.csv", index=False)
    print("\n=== CLEAN-test macro-F1 by label-noise rate (RICH pool) ===")
    for _, r in agg.iterrows():
        print(f"  rate={r['rate']:.2f}: no_clean={r.no_clean_f1:.3f} R3={r.R3_f1:.3f} R7acc={r.R7acc_f1:.3f} "
              f"R7F1={r.R7F1_f1:.3f} R7conf={r.R7conf_f1:.3f} R7margin={r.R7margin_f1:.3f} "
              f"R7prior={r.R7prior_f1:.3f} oracle={r.oracle_f1:.3f} oracleF1={r.oracleF1_f1:.3f}")
    print("\n  F1 deltas vs references (mean over rates):")
    for a in ("R7F1", "R7conf", "R7margin", "R7prior"):
        print(f"    {a:9}: −R7acc={ (agg[f'{a}_f1']-agg.R7acc_f1).mean():+.4f}   −R7F1={ (agg[f'{a}_f1']-agg.R7F1_f1).mean():+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["blood_transfusion", "credit_g", "hepatitis"])
    ap.add_argument("--rates", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.35])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/paper_ready/noise_rich"))
    a = ap.parse_args()
    main(a.datasets, a.rates, tuple(a.seeds), a.output_dir)
