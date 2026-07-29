"""
experiments/run_saga_comparison.py

SAGA head-to-head on the strong datasets.
Runs OUR held-out protocol cleaning method on the SAGA datasets we could source, evaluated
with LogReg (to match SAGA's multinomial-logreg downstream) AND TabPFN (flagged), and
tabulates our accuracy next to SAGA's PUBLISHED numbers.

Strong datasets (A — run our method):
  EEG          14 numeric, 2-class      (ideal fit; OpenML eeg-eye-state)
  AnimalShelter 8 categorical, 5-class  (encoded; SAGA committed CSVs)
  Titanic      mixed, 2-class           (leakage cols dropped; encoded)
Cited only (C — published, not re-run here): Movie, Nashville, Puma (categorical-heavy /
hard to source the exact version), Cancer, Housing (regression — out of scope).

HONEST CAVEATS baked into the report:
  * We use each dataset's NATURAL state (no MCAR injection), so our "dirty" baseline is
    NOT SAGA's CleanML-corrupted version — the OpenML EEG is largely clean, so the
    comparison is indicative, not a controlled identical-dirt head-to-head.
  * Our cleaning operators are numeric; categorical features are ordinal-encoded first
    (SAGA dummy-codes) — a methodological difference.
  * SAGA uses a 70/30 split; we use 80/20 outer + inner-val. Downstream: LogReg ≈ mLogReg.

Usage
-----
  PYTHONPATH=src:experiments python experiments/run_saga_comparison.py --seeds 42 1 2 3 4
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

import run_c2_tfm_reward_nested as G
import run_weight_robustness as W

DATA = Path(__file__).parents[1] / "data" / "saga"

# Per-dataset config: file(s), target col, columns to drop (IDs / post-outcome leakage / high-card).
DATASETS = {
    "EEG": {"files": ["eeg.csv"], "target": "Class", "drop": []},
    "AnimalShelter": {"files": ["animalshelter_train.csv", "animalshelter_test.csv"],
                      "target": "class", "drop": ["AnimalID", "Name", "OutcomeSubtype"]},
    "Titanic": {"files": ["titanic.csv"], "target": "survived",
                "drop": ["name", "ticket", "cabin", "boat", "body", "home.dest"]},
}

# SAGA paper Table 5 (linear/mLogReg), for the comparison table. R² for cancer/housing.
SAGA_PUBLISHED = {  # dataset: (dirty, saga, learn2clean_v1)
    "Animal": (0.70, 0.86, None), "EEG": (0.65, 0.68, 0.67), "Movie": (0.75, 0.85, 0.76),
    "Nashville": (0.79, 0.80, 0.79), "Puma": (0.54, 0.57, 0.51), "Titanic": (0.78, 0.82, 0.73),
    "Cancer": (0.43, 0.52, 0.61), "Housing": (0.81, 0.87, 0.89),
}


def load_saga(name: str) -> Optional[tuple]:
    cfg = DATASETS[name]
    paths = [DATA / f for f in cfg["files"]]
    if not all(p.exists() for p in paths):
        print(f"  [SKIP] {name}: missing {[str(p) for p in paths if not p.exists()]}")
        return None
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df.drop(columns=[c for c in cfg["drop"] if c in df.columns], errors="ignore")
    y_raw = df[cfg["target"]].astype(str)
    X = df.drop(columns=[cfg["target"]])
    # ordinal-encode object/category columns; keep numerics
    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat] = enc.fit_transform(X[cat].astype(str))
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.Series(LabelEncoder().fit_transform(y_raw), name="y")
    return X.reset_index(drop=True), y.reset_index(drop=True)


def select_r7(X_sel, y_sel, pipelines, actions, seed) -> tuple:
    """Held-out protocol TFM-aware (R7) selection on inner-val only."""
    n0 = len(X_sel); best, best_s = (), -np.inf
    for seq in pipelines:
        Xc = G.apply_pipeline(X_sel, y_sel, seq, actions)
        if Xc is None or len(Xc) == 0:
            continue
        acc = W.inner_val_acc(Xc, y_sel, seed, "tabpfn")
        if not np.isfinite(acc):
            continue
        ret = (len(Xc) / n0) ** 2.0
        miss = float(Xc.isna().mean().mean()); dup = float(Xc.duplicated().sum()) / max(len(Xc), 1)
        s = 0.50 * acc + 0.35 * ret + 0.15 * (1 - miss) * (1 - dup)
        if s > best_s:
            best_s, best = s, seq
    return best


def run_one(name, seed, pipelines, actions) -> Optional[Dict]:
    loaded = load_saga(name)
    if loaded is None:
        return None
    X, y = loaded
    if len(X) > G.SUBSAMPLE_CAP:
        X, _, y, _ = train_test_split(X, y, train_size=G.SUBSAMPLE_CAP, random_state=seed, stratify=y)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    # NATURAL dirty — no injection
    try:
        X_sel, X_test, y_sel, y_test = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
    except ValueError:
        X_sel, X_test, y_sel, y_test = train_test_split(X, y, test_size=0.30, random_state=seed)
    X_sel, y_sel = X_sel.reset_index(drop=True), y_sel.reset_index(drop=True)
    X_test, y_test = X_test.reset_index(drop=True), y_test.reset_index(drop=True)

    best = select_r7(X_sel, y_sel, pipelines, actions, seed)
    Xc = G.apply_pipeline(X_sel, y_sel, best, actions)
    Xtp = G.prepare_test_like_train(X_sel, X_test, best)
    Xtr, ytr, le = G._encode_align(Xc, y_sel)
    shared = [c for c in Xc.select_dtypes(include="number").columns if c in Xtp.columns]
    try:
        yte = le.transform(np.asarray(y_test)); Xte = Xtp[shared].values.astype(float)
        panel = W.panel_on_test(Xtr, ytr, Xte, yte, seed)
    except Exception:
        panel = {}
    return {"dataset": name, "seed": seed, "pipe": G.pipeline_label(best),
            "logreg_acc": panel.get("logreg", {}).get("acc", np.nan),
            "logreg_f1": panel.get("logreg", {}).get("f1", np.nan),
            "tabpfn_acc": panel.get("tabpfn", {}).get("acc", np.nan)}


def main(seeds, max_pipelines, output_dir=None) -> None:
    out_dir = Path(output_dir) if output_dir else (
        Path(__file__).parents[1] / "outputs" / "paper_ready" / "saga_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    actions = G.build_actions()
    all_pipelines = G.enumerate_valid_pipelines(max_len=3)
    rows: List[Dict] = []
    for name in DATASETS:
        for seed in seeds:
            pls = G.sample_pipelines(all_pipelines, max_pipelines, seed)
            r = run_one(name, seed, pls, actions)
            if r:
                rows.append(r)
                print(f"  {name:14} s{seed}: ours(LogReg)={r['logreg_acc']:.4f} "
                      f"ours(TabPFN)={r['tabpfn_acc']:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(out_dir / "saga_comparison_per_seed.csv", index=False)

    if not rows:
        print("No results."); return
    df = pd.DataFrame(rows)
    agg = df.groupby("dataset").agg(ours_logreg=("logreg_acc", "mean"),
                                    ours_logreg_sd=("logreg_acc", "std"),
                                    ours_tabpfn=("tabpfn_acc", "mean")).reset_index()
    agg.to_csv(out_dir / "saga_comparison_aggregated.csv", index=False)
    print("\n=== SAGA COMPARISON (ours = L2C V3, held-out protocol; natural-dirty; LogReg≈mLogReg) ===")
    print(f"{'dataset':14} {'dirty':>6} {'SAGA':>6} {'L2C-v1':>7} {'OURS-LogReg':>12} {'OURS-TabPFN':>12}")
    for _, r in agg.iterrows():
        pub = SAGA_PUBLISHED.get(r["dataset"], (None, None, None))
        d = "  N/A" if pub[0] is None else f"{pub[0]:.2f}"
        s = "  N/A" if pub[1] is None else f"{pub[1]:.2f}"
        l = "   N/A" if pub[2] is None else f"{pub[2]:.2f}"
        print(f"{r['dataset']:14} {d:>6} {s:>6} {l:>7} {r['ours_logreg']:>12.4f} {r['ours_tabpfn']:>12.4f}")
    print(f"\nSaved → {out_dir}\nNOTE: natural-dirty ≠ SAGA's CleanML-dirty; indicative comparison.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2, 3, 4])
    ap.add_argument("--max-pipelines", type=int, default=18)
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    main(tuple(a.seeds), a.max_pipelines, a.output_dir)
