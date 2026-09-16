from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
import run_saga_richops as R

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
    if len(X) > 8000:
        X, _, y, _ = train_test_split(X, y, train_size=8000, random_state=0, stratify=y)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    Xd = mcar(X, 0.15, seed)
    try:
        Xtr, Xte, ytr, yte = train_test_split(Xd, y, test_size=0.3, random_state=seed, stratify=y)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(Xd, y, test_size=0.3, random_state=seed)
    Xtr, ytr = Xtr.reset_index(drop=True), ytr.reset_index(drop=True)
    Xte, yte = Xte.reset_index(drop=True), yte.reset_index(drop=True)
    pa, _ = R.select_pipeline(Xtr, ytr, False, seed, "tabpfn", "acc")
    pf, _ = R.select_pipeline(Xtr, ytr, False, seed, "tabpfn", "f1")
    ma = R.test_metrics(Xtr, ytr, Xte, yte, pa, seed)
    mf = R.test_metrics(Xtr, ytr, Xte, yte, pf, seed)
    return {"dataset": name, "seed": seed,
            "acc": ma["acc"], "f1": mf["f1"],
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
