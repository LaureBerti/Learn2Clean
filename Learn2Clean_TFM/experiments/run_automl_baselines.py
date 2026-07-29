"""
experiments/run_automl_baselines.py

AutoML end-to-end baselines: AutoGluon and Auto-sklearn 2.0, run on
the SAME datasets + MCAR-15% corruption + held-out protocol 70/30 split as our main experiments.
These are END-TO-END systems (their own preprocessing + model), so we report their test
accuracy alongside our cleaning+TabPFN numbers. Self-contained — does NOT import our TabPFN
stack (AutoML pins conflicting sklearn versions), so each runs in its own VM/env.

  --method autogluon   : AutoGluon TabularPredictor (handles NaN + categoricals natively)
  --method autosklearn : Auto-sklearn 2.0 (ordinal-encode categoricals, impute sentinel)

Usage
-----
  PYTHONPATH=src python experiments/run_automl_baselines.py --method autogluon --seeds 42 1 2
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parents[1]
SAGA = {"EEG": ("eeg.csv", "Class", []),
        "AnimalShelter": (["animalshelter_train.csv", "animalshelter_test.csv"], "class",
                          ["AnimalID", "Name", "OutcomeSubtype"]),
        "Titanic": ("titanic.csv", "survived", ["name", "ticket", "cabin", "boat", "body", "home.dest"])}
OPENML = ["hepatitis", "heart_statlog", "ionosphere", "blood_transfusion", "diabetes",
          "credit_g", "kr_vs_kp", "phoneme", "adult", "bank_marketing"]


def load_ds(name):
    if name in SAGA:
        files, tgt, drop = SAGA[name]
        files = [files] if isinstance(files, str) else files
        df = pd.concat([pd.read_csv(ROOT / "data/saga" / f) for f in files], ignore_index=True)
        df = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
        return df.drop(columns=[tgt]), df[tgt].astype(str)
    sys.path.insert(0, str(ROOT / "src"))
    from learn2clean_v3.data.openml_loader import load_dataset
    X, y, _ = load_dataset(name, use_cache=True)
    return X, y.astype(str)


def mcar(X, rate, seed):
    rng = np.random.default_rng(seed); X = X.copy()
    for c in X.select_dtypes(include="number").columns:
        X.loc[rng.random(len(X)) < rate, c] = np.nan
    return X


def fit_predict(method, Xtr, ytr, Xte, tl):
    if method == "autogluon":
        from autogluon.tabular import TabularPredictor
        tr = Xtr.copy(); tr["__label__"] = ytr.values
        p = TabularPredictor(label="__label__", verbosity=0).fit(tr, time_limit=tl, presets="medium_quality")
        return p.predict(Xte).values
    import autosklearn.classification as ASK
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
    cat = Xtr.select_dtypes(include=["object", "category"]).columns.tolist()
    Xtr2, Xte2 = Xtr.copy(), Xte.copy()
    if cat:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xtr2[cat] = enc.fit_transform(Xtr2[cat].astype(str)); Xte2[cat] = enc.transform(Xte2[cat].astype(str))
    Xtr2 = Xtr2.apply(pd.to_numeric, errors="coerce").fillna(-999.0)
    Xte2 = Xte2.apply(pd.to_numeric, errors="coerce").fillna(-999.0)
    le = LabelEncoder(); y = le.fit_transform(ytr)
    clf = ASK.AutoSklearnClassifier(time_left_for_this_task=tl, per_run_time_limit=max(30, tl // 4), memory_limit=8000)
    clf.fit(Xtr2.values, y)
    return le.inverse_transform(clf.predict(Xte2.values))


def main(method, datasets, seeds, time_limit, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in datasets:
        try:
            X, y = load_ds(ds)
        except Exception as e:
            print(f"  [SKIP] {ds}: {e}"); continue
        if len(X) > 8000:
            X, _, y, _ = train_test_split(X, y, train_size=8000, random_state=0, stratify=y)
            X, y = X.reset_index(drop=True), y.reset_index(drop=True)
        for seed in seeds:
            Xd = mcar(X, 0.15, seed)
            try:
                Xtr, Xte, ytr, yte = train_test_split(Xd, y, test_size=0.3, random_state=seed, stratify=y)
            except ValueError:
                Xtr, Xte, ytr, yte = train_test_split(Xd, y, test_size=0.3, random_state=seed)
            t0 = time.time()
            try:
                yp = fit_predict(method, Xtr, ytr, Xte, time_limit)
                acc = accuracy_score(yte.values, yp); f1 = f1_score(yte.values, yp, average="macro")
            except Exception as e:
                print(f"  FAIL {ds} s{seed}: {repr(e)[:160]}"); acc = f1 = float("nan")
            rows.append({"method": method, "dataset": ds, "seed": seed, "acc": acc, "f1": f1, "sec": time.time() - t0})
            pd.DataFrame(rows).to_csv(out / "automl_per_run.csv", index=False)
            print(f"  {method} {ds:14} s{seed}: acc={acc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    if len(df):
        agg = df.groupby("dataset").agg(acc=("acc", "mean"), acc_sd=("acc", "std"), f1=("f1", "mean")).reset_index()
        agg.to_csv(out / "automl_aggregated.csv", index=False)
        print(f"\n=== {method} mean acc by dataset ===\n{agg.to_string(index=False)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["autogluon", "autosklearn"], required=True)
    ap.add_argument("--datasets", nargs="+", default=OPENML + list(SAGA.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--time-limit", type=int, default=120)
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/paper_ready/automl_baselines"))
    a = ap.parse_args()
    main(a.method, a.datasets, tuple(a.seeds), a.time_limit, a.output_dir)
