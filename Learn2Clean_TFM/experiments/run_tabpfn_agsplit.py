"""
experiments/run_tabpfn_agsplit.py

Byte-identical companion to `run_automl_baselines.py` for verdict ⑫: run OUR cleaning+TabPFN on the
EXACT same data preparation AutoGluon/Auto-sklearn saw — same load, same 8000-row cap (random_state=0),
same MCAR-15%, same 70/30 stratified split (random_state=seed) — so ours-vs-AutoGluon is on identical
splits. Selection is held-out protocol R7 (TabPFN inner-val accuracy, base operator pool) on the 70% train
only; the 30% test is scored once with TabPFN.

Usage: PYTHONPATH=src:experiments python experiments/run_tabpfn_agsplit.py --seeds 42 1 2
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
import run_saga_richops as R   # load_ds, apply_pipeline, groups_for, select_pipeline, test_metrics

SAGA = {"EEG", "AnimalShelter", "Titanic"}
OPENML = ["hepatitis", "heart_statlog", "ionosphere", "blood_transfusion", "diabetes",
          "credit_g", "kr_vs_kp", "phoneme", "adult", "bank_marketing"]


def mcar(X, rate, seed):
    rng = np.random.default_rng(seed); X = X.copy()
    for c in X.select_dtypes(include="number").columns:
        X.loc[rng.random(len(X)) < rate, c] = np.nan
    return X


def run_one(name, seed):
    X, y = R.load_ds(name)
    if len(X) > 8000:                                            # EXACT AutoML cap
        X, _, y, _ = train_test_split(X, y, train_size=8000, random_state=0, stratify=y)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    Xd = mcar(X, 0.15, seed)                                     # EXACT AutoML corruption
    try:
        Xtr, Xte, ytr, yte = train_test_split(Xd, y, test_size=0.3, random_state=seed, stratify=y)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(Xd, y, test_size=0.3, random_state=seed)
    Xtr, ytr = Xtr.reset_index(drop=True), ytr.reset_index(drop=True)
    Xte, yte = Xte.reset_index(drop=True), yte.reset_index(drop=True)
    # held-out protocol selection on the 70% train (base pool) under BOTH objectives, deploy TabPFN on 30% test.
    # Report matched-metric: accuracy from R7acc-selection, macro-F1 from R7F1-selection (verdict ⑭).
    pa, _ = R.select_pipeline(Xtr, ytr, False, seed, "tabpfn", "acc")
    pf, _ = R.select_pipeline(Xtr, ytr, False, seed, "tabpfn", "f1")
    ma = R.test_metrics(Xtr, ytr, Xte, yte, pa, seed)
    mf = R.test_metrics(Xtr, ytr, Xte, yte, pf, seed)
    return {"dataset": name, "seed": seed,
            "acc": ma["acc"], "f1": mf["f1"],              # matched-metric headline
            "R7acc_acc": ma["acc"], "R7acc_f1": ma["f1"],
            "R7f1_acc": mf["acc"], "R7f1_f1": mf["f1"]}


def main(datasets, seeds, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in datasets:
        for seed in seeds:
            try:
                r = run_one(ds, seed)
            except Exception as e:
                r = {"dataset": ds, "seed": seed, "acc": np.nan, "f1": np.nan, "err": repr(e)[:140]}
            rows.append(r); pd.DataFrame(rows).to_csv(out / "tabpfn_agsplit_per_run.csv", index=False)
            print(f"  {ds:14} s{seed}: acc={r.get('acc', np.nan):.4f} f1={r.get('f1', np.nan):.4f}", flush=True)
    df = pd.DataFrame(rows)
    agg = df.groupby("dataset").agg(acc=("acc", "mean"), acc_sd=("acc", "std"), f1=("f1", "mean")).reset_index()
    agg.to_csv(out / "tabpfn_agsplit_aggregated.csv", index=False)
    print("\n=== ours (clean+TabPFN) on AutoGluon's EXACT split ===")
    print(agg.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=OPENML + list(SAGA))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/paper_ready/tabpfn_agsplit"))
    a = ap.parse_args()
    main(a.datasets, tuple(a.seeds), a.output_dir)
