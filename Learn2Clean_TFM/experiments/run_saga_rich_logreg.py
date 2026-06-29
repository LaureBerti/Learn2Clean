"""

CLEAN operator-richness-vs-SAGA test with the MODEL HELD CONSTANT (LogReg, matching SAGA's mLogReg).
For each SAGA dataset: greedy-select a cleaning pipeline from the BASE (7-op) pool and from the RICH
(13,440-pipeline) pool, selecting AND evaluating with LogReg, leak-free, 8 seeds. Answers: do rich
operators close the gap to SAGA's published Table-5 accuracy when we don't change the downstream model?

Reuses run_saga_richops.apply_pipeline / groups_for / load_ds (all rich operators), but swaps the
TabPFN evaluator for standardized multinomial LogReg both in selection and on the sacred test.

Usage:  PYTHONPATH=src:experiments python experiments/run_saga_rich_logreg.py --seeds 42 1 2 3 4 5 6 7
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
import run_saga_richops as R
import run_c2_tfm_reward_nested as G

SAGA_PUBLISHED = {"EEG": 0.68, "Titanic": 0.82, "AnimalShelter": 0.86}   # paper Table 5 (mLogReg acc)


def lr_acc(tr, ytr, te, yte):
    sc = StandardScaler().fit(tr)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(sc.transform(tr), ytr)
    return float(accuracy_score(yte, clf.predict(sc.transform(te))))


def inner_lr(pipe, Xi_tr, yi_tr, Xi_val, yi_val, seed):
    out = R.apply_pipeline(Xi_tr, yi_tr, Xi_val, pipe, seed)
    if out is None:
        return -1.0
    tr, ytr2, val = out
    try:
        if len(np.unique(ytr2.values)) < 2:
            return -1.0
        return lr_acc(tr.values, ytr2.values, val.values, yi_val.values)
    except Exception:
        return -1.0


def select_lr(Xtr_out, ytr_out, rich, seed, passes=2):
    Xi_tr, Xi_val, yi_tr, yi_val = train_test_split(
        Xtr_out, ytr_out, test_size=0.25, random_state=seed,
        stratify=ytr_out if ytr_out.value_counts().min() >= 2 else None)
    groups = R.groups_for(rich)
    cur = tuple(None if None in opts else opts[0] for _, opts in groups)
    best = inner_lr(cur, Xi_tr, yi_tr, Xi_val, yi_val, seed)
    for _ in range(passes):
        improved = False
        for gi, (_, opts) in enumerate(groups):
            for opt in opts:
                if opt == cur[gi]:
                    continue
                trial = cur[:gi] + (opt,) + cur[gi + 1:]
                s = inner_lr(trial, Xi_tr, yi_tr, Xi_val, yi_val, seed)
                if s > best:
                    best, cur, improved = s, trial, True
        if not improved:
            break
    return cur


def eval_lr(Xtr_out, ytr_out, Xte, yte, pipe, seed):
    out = R.apply_pipeline(Xtr_out, ytr_out, Xte, pipe, seed)
    if out is None:
        return np.nan
    tr, ytr2, te = out
    try:
        if len(np.unique(ytr2.values)) < 2:
            return np.nan
        return lr_acc(tr.values, ytr2.values, te.values, yte.values)
    except Exception:
        return np.nan


def run_one(name, seed):
    X, y = R.load_ds(name)
    if len(X) > G.SUBSAMPLE_CAP:
        X, _, y, _ = train_test_split(X, y, train_size=G.SUBSAMPLE_CAP, random_state=0,
                                      stratify=y if y.value_counts().min() >= 2 else None)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    Xd = R.mcar(X, 0.15, seed)
    Xtr, Xte, ytr, yte = train_test_split(Xd, y, test_size=0.20, random_state=seed,
                                          stratify=y if y.value_counts().min() >= 2 else None)
    Xtr, ytr = Xtr.reset_index(drop=True), ytr.reset_index(drop=True)
    Xte, yte = Xte.reset_index(drop=True), yte.reset_index(drop=True)
    bp = select_lr(Xtr, ytr, False, seed); rp = select_lr(Xtr, ytr, True, seed)
    return {"dataset": name, "seed": seed,
            "base_lr": eval_lr(Xtr, ytr, Xte, yte, bp, seed),
            "rich_lr": eval_lr(Xtr, ytr, Xte, yte, rp, seed)}


def main(datasets, seeds, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in datasets:
        for seed in seeds:
            try:
                r = run_one(ds, seed)
            except Exception as e:
                r = {"dataset": ds, "seed": seed, "base_lr": np.nan, "rich_lr": np.nan, "err": repr(e)[:120]}
            rows.append(r); pd.DataFrame(rows).to_csv(out / "saga_rich_logreg_per_run.csv", index=False)
            print(f"  {ds:14} s{seed}: base_lr={r.get('base_lr',np.nan):.4f} rich_lr={r.get('rich_lr',np.nan):.4f}", flush=True)
    df = pd.DataFrame(rows)
    agg = df.groupby("dataset").agg(base_lr=("base_lr", "mean"), base_sd=("base_lr", "std"),
                                    rich_lr=("rich_lr", "mean"), rich_sd=("rich_lr", "std")).reset_index()
    agg["saga"] = agg.dataset.map(SAGA_PUBLISHED)
    agg["rich_minus_base"] = agg.rich_lr - agg.base_lr
    agg["rich_vs_saga"] = agg.rich_lr - agg.saga
    agg.to_csv(out / "saga_rich_logreg_aggregated.csv", index=False)
    print(f"\n=== Operator richness vs SAGA, MODEL HELD CONSTANT (LogReg), {len(seeds)} seeds ===")
    print(agg.round(4).to_string(index=False))
    print(f"\nmean rich−base = {agg.rich_minus_base.mean():+.4f}  (do rich operators help, model fixed?)")
    print(f"mean rich−SAGA = {agg.rich_vs_saga.mean():+.4f}  (does rich close the SAGA gap?)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["EEG", "Titanic", "AnimalShelter"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/paper_ready/saga_rich_logreg"))
    a = ap.parse_args()
    main(a.datasets, tuple(a.seeds), a.output_dir)
