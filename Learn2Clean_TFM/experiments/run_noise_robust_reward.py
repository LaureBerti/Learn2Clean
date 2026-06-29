"""

Tests the most promising R7 improvement (selectability finding, the paper): under LABEL NOISE the
validation labels are themselves corrupted, so selecting by validation accuracy cannot reward the
label-cleaning operator. We test a LABEL-FREE, noise-robust selection signal — TabPFN predictive
CONFIDENCE on the inner-val — which a clean training context should sharpen, letting it pick the
mislabel-remover that accuracy-based rewards miss.

Selection arms (action pool = BASE 7 ops + LabelCleaner), all DEPLOYED on TabPFN, evaluated on a
SACRED CLEAN test (true labels):
  no_clean : identity
  R3       : argmax RF accuracy on (noisy) inner-val
  R7acc    : argmax TabPFN accuracy on (noisy) inner-val          <- current reward
  R7conf   : argmax TabPFN mean-confidence on inner-val (LABEL-FREE)  <- proposed noise-robust reward
  oracle   : argmax TabPFN accuracy on CLEAN inner-val (true labels)  <- upper bound (headroom)

Hypothesis: under label noise, R7conf > R7acc ≈ R3 and selects LabelCleaner more often, approaching
the oracle — i.e. a noise-robust reward unlocks the one regime where model-awareness should matter.

Usage:  PYTHONPATH=src:experiments python experiments/run_noise_robust_reward.py \
            --datasets hepatitis diabetes ionosphere credit_g --rates 0.0 0.1 0.2 0.35 --seeds 42 1 2
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
import run_c2_tfm_reward_nested as G
from run_corruption_sweep import LabelCleaner, inject_label_noise, BASE, EXT, EXT_GROUP, enum_pipes

CLEAN_VAL = {}   # cache true-label inner-val per (dataset,seed) for the oracle arm


def _tab_proba(Xc, y, X_val, seed):
    """TabPFN fit on cleaned (Xc,y); return (pred,le), mean_confidence, mean_margin on X_val."""
    Xtr, ytr, le = G._encode_align(Xc, y)
    shared = [c for c in Xc.select_dtypes(include="number").columns]
    Xv = X_val[[c for c in shared if c in X_val.columns]].values.astype(float)
    if len(np.unique(ytr)) < 2 or Xv.shape[1] == 0:
        return None, np.nan, np.nan
    pred, prob = G._tabpfn_fit_predict(Xtr, ytr, Xv, seed)
    if prob.ndim != 2:
        return (pred, le), np.nan, np.nan
    conf = float(np.mean(np.max(prob, axis=1)))
    srt = np.sort(prob, axis=1)
    margin = float(np.mean(srt[:, -1] - srt[:, -2])) if prob.shape[1] >= 2 else conf
    return (pred, le), conf, margin


def _score(seq, X_sel, y_sel, X_val, y_val_noisy, y_val_clean, seed, kind):
    """Selection score of pipeline seq under the chosen signal."""
    Xc = G.apply_pipeline(X_sel, y_sel, seq, EXT)
    if Xc is None or len(Xc) == 0:
        return -np.inf
    yc = y_sel.loc[Xc.index]
    if kind == "rf":
        Xtr, ytr, le = G._encode_align(Xc, yc)
        shared = [c for c in Xc.select_dtypes(include="number").columns if c in X_val.columns]
        if len(np.unique(ytr)) < 2 or not shared:
            return -np.inf
        try:
            clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1).fit(Xtr, ytr)
            return accuracy_score(le.transform(y_val_noisy.values), clf.predict(X_val[shared].values.astype(float)))
        except Exception:
            return -np.inf
    out, conf = _tab_proba(Xc, yc, X_val, seed)
    if out is None:
        return -np.inf
    pred, le = out
    if kind == "conf":
        return conf                                              # LABEL-FREE
    ref = y_val_clean if kind == "acc_clean" else y_val_noisy    # accuracy vs noisy or clean labels
    try:
        return accuracy_score(le.transform(ref.values), pred)
    except Exception:
        return -np.inf


def select(pipes, X_sel, y_sel, X_val, y_vn, y_vc, seed, kind):
    best, bs = (), -np.inf
    for seq in pipes:
        s = _score(seq, X_sel, y_sel, X_val, y_vn, y_vc, seed, kind)
        if s > bs:
            bs, best = s, seq
    return best


def deploy(seq, X_sel, y_sel, X_test, y_test_clean, seed):
    """Apply seq to train, fit TabPFN, score on the CLEAN sacred test."""
    Xc = G.apply_pipeline(X_sel, y_sel, seq, EXT)
    if Xc is None or len(Xc) == 0:
        return np.nan, np.nan
    yc = y_sel.loc[Xc.index]
    Xtp = G.prepare_test_like_train(X_sel, X_test, seq)
    Xtr, ytr, le = G._encode_align(Xc, yc)
    shared = [c for c in Xc.select_dtypes(include="number").columns if c in Xtp.columns]
    try:
        if len(np.unique(ytr)) < 2 or not shared:
            return np.nan, np.nan
        pred, _ = G._tabpfn_fit_predict(Xtr, ytr, Xtp[shared].values.astype(float), seed)
        yt = le.transform(y_test_clean.values)
        return float(accuracy_score(yt, pred)), float(f1_score(yt, pred, average="macro"))
    except Exception:
        return np.nan, np.nan


def run_one(name, rate, seed, pipes):
    sys.path.insert(0, str(ROOT / "src"))
    from learn2clean_v3.data.openml_loader import load_dataset
    X, y, _ = load_dataset(name, use_cache=True); y = pd.Series(y).astype(str)
    if len(X) > G.SUBSAMPLE_CAP:
        X, _, y, _ = train_test_split(X, y, train_size=G.SUBSAMPLE_CAP, random_state=0, stratify=y)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    # sacred CLEAN test (true labels); training pool then gets label noise
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    X_tr, y_tr = X_tr.reset_index(drop=True), y_tr.reset_index(drop=True)
    X_test, y_test = X_test.reset_index(drop=True), y_test.reset_index(drop=True)
    X_sel, X_val, y_sel_c, y_val_c = train_test_split(X_tr, y_tr, test_size=0.25, random_state=seed, stratify=y_tr)
    X_sel, X_val = X_sel.reset_index(drop=True), X_val.reset_index(drop=True)
    y_sel_c, y_val_c = y_sel_c.reset_index(drop=True), y_val_c.reset_index(drop=True)
    # inject label noise on the (selection) training + its validation — the realistic selectability case
    y_sel = inject_label_noise(y_sel_c, rate, seed)
    y_val_n = inject_label_noise(y_val_c, rate, seed + 1)

    def uses_label(seq):
        return int(any(EXT_GROUP[i] == "label" for i in seq))

    row = {"dataset": name, "rate": rate, "seed": seed}
    arms = {"R3": "rf", "R7acc": "acc_noisy", "R7conf": "conf", "oracle": "acc_clean"}
    for arm, kind in arms.items():
        seq = () if False else select(pipes, X_sel, y_sel, X_val, y_val_n, y_val_c, seed, kind)
        acc, f1 = deploy(seq, X_sel, y_sel, X_test, y_test, seed)
        row[f"{arm}_acc"], row[f"{arm}_f1"], row[f"{arm}_uselbl"] = acc, f1, uses_label(seq)
    nc_acc, nc_f1 = deploy((), X_sel, y_sel, X_test, y_test, seed)
    row["no_clean_acc"], row["no_clean_f1"] = nc_acc, nc_f1
    return row


def main(datasets, rates, seeds, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    pipes = G.sample_pipelines(enum_pipes(EXT_GROUP), 24, 42)
    rows = []
    for ds in datasets:
        for rate in rates:
            for seed in seeds:
                try:
                    r = run_one(ds, rate, seed, pipes)
                except Exception as e:
                    r = {"dataset": ds, "rate": rate, "seed": seed, "err": repr(e)[:140]}
                rows.append(r); pd.DataFrame(rows).to_csv(out / "noise_robust_per_run.csv", index=False)
                print(f"  {ds:12} r{rate} s{seed}: R3={r.get('R3_acc',np.nan):.3f} R7acc={r.get('R7acc_acc',np.nan):.3f} "
                      f"R7conf={r.get('R7conf_acc',np.nan):.3f} oracle={r.get('oracle_acc',np.nan):.3f} "
                      f"| useLbl R7conf={r.get('R7conf_uselbl','?')}", flush=True)
    df = pd.DataFrame(rows)
    agg = df.groupby("rate").mean(numeric_only=True).reset_index()
    agg.to_csv(out / "noise_robust_by_rate.csv", index=False)
    print("\n=== CLEAN-test accuracy by label-noise rate (mean over ds×seeds) ===")
    for _, r in agg.iterrows():
        print(f"  rate={r['rate']:.2f}: no_clean={r.get('no_clean_acc',np.nan):.4f}  R3={r.R3_acc:.4f}  "
              f"R7acc={r.R7acc_acc:.4f}  R7conf={r.R7conf_acc:.4f}  oracle={r.oracle_acc:.4f}  "
              f"| LabelCleaner-selected: R7conf={r.R7conf_uselbl:.2f} R7acc={r.R7acc_uselbl:.2f} R3={r.R3_uselbl:.2f}")
    print(f"\nKEY: mean(R7conf − R7acc) = {(agg.R7conf_acc-agg.R7acc_acc).mean():+.4f}  "
          f"(does the label-free reward beat accuracy-selection under noise?)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["hepatitis", "diabetes", "ionosphere", "credit_g"])
    ap.add_argument("--rates", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.35])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/paper_ready/noise_robust"))
    a = ap.parse_args()
    main(a.datasets, a.rates, tuple(a.seeds), a.output_dir)
