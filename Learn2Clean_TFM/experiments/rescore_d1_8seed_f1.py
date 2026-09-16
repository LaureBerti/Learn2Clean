from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import run_c2_tfm_reward_nested as C2
from run_c2_tfm_reward_nested import (ACTION_LABELS, MCAR_RATE, OUTER_TEST_SIZE,
                                      SUBSAMPLE_CAP, apply_pipeline,
                                      build_actions, compute_ece, _encode_align,
                                      _tabpfn_fit_predict, prepare_test_like_train)
from learn2clean_v3.data.error_injection import ErrorProfile, apply_error_profile
from learn2clean_v3.data.openml_loader import load_dataset

LABEL_TO_IDX = {v: k for k, v in ACTION_LABELS.items()}


def parse_pipeline(label: str):
    label = str(label).strip()
    if label in ("", "no_op", "nan"):
        return ()
    steps = [s.strip() for s in label.replace("->", "→").split("→")]
    seq = []
    for s in steps:
        if s not in LABEL_TO_IDX:
            raise ValueError(f"unknown action label: {s!r} in {label!r}")
        seq.append(LABEL_TO_IDX[s])
    return tuple(seq)


def final_test_allmetrics(X_sel_clean, y_sel, X_test_prepared, y_test, seed):
    Xtr, ytr, le = _encode_align(X_sel_clean, y_sel)
    shared = [c for c in X_sel_clean.select_dtypes(include="number").columns
              if c in X_test_prepared.columns]
    Xte = X_test_prepared[shared].values.astype(float)
    nan = (float("nan"),) * 5
    try:
        yte = le.transform(np.asarray(y_test))
    except Exception:
        return nan
    if len(np.unique(ytr)) < 2 or len(yte) == 0:
        return nan
    try:
        y_pred, y_prob = _tabpfn_fit_predict(Xtr, ytr, Xte, seed)
    except Exception as exc:
        print(f"    final-test TabPFN failed: {exc}")
        return nan
    acc = float(accuracy_score(yte, y_pred))
    ece = compute_ece(yte, y_prob)
    f1 = float(f1_score(yte, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(yte, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(yte, y_pred, average="macro", zero_division=0))
    return acc, ece, f1, prec, rec


def reconstruct_split(ds_name: str, seed: int):
    X, y, spec = load_dataset(ds_name, use_cache=True)
    if len(X) > SUBSAMPLE_CAP:
        Xs, _, ys, _ = train_test_split(X, y, train_size=SUBSAMPLE_CAP,
                                        random_state=seed, stratify=y)
        X, y = Xs.reset_index(drop=True), ys.reset_index(drop=True)
    X_dirty, y_dirty = apply_error_profile(X, y, ErrorProfile("mcar", rate=MCAR_RATE, seed=seed))
    try:
        X_sel, X_test, y_sel, y_test = train_test_split(
            X_dirty, y_dirty, test_size=OUTER_TEST_SIZE, random_state=seed, stratify=y_dirty)
    except ValueError:
        X_sel, X_test, y_sel, y_test = train_test_split(
            X_dirty, y_dirty, test_size=OUTER_TEST_SIZE, random_state=seed)
    return (X_sel.reset_index(drop=True), y_sel.reset_index(drop=True),
            X_test.reset_index(drop=True), y_test.reset_index(drop=True))


def main(in_csv: str, out_csv: str) -> None:
    src = pd.read_csv(in_csv)
    actions = build_actions()
    rows, max_acc_err, max_ece_err = [], 0.0, 0.0
    t0 = time.time()

    for _, r in src.iterrows():
        ds, seed = r["dataset"], int(r["seed"])
        ds_key = ds.replace("_", "-")
        X_sel, y_sel, X_test, y_test = reconstruct_split(ds_key, seed)
        out = {"dataset": ds, "seed": seed,
               "rf_pipeline": r["rf_pipeline"], "tfm_pipeline": r["tfm_pipeline"],
               "pipelines_match": int(r["pipelines_match"])}
        for mode in ("rf", "tfm"):
            seq = parse_pipeline(r[f"{mode}_pipeline"])
            X_clean = apply_pipeline(X_sel, y_sel, seq, actions)
            if X_clean is None:
                for m in ("acc", "ece", "f1", "prec", "rec"):
                    out[f"{mode}_{m}"] = float("nan")
                continue
            X_test_prep = prepare_test_like_train(X_sel, X_test, seq)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                acc, ece, f1, prec, rec = final_test_allmetrics(
                    X_clean, y_sel, X_test_prep, y_test, seed)
            out[f"{mode}_acc"], out[f"{mode}_ece"] = acc, ece
            out[f"{mode}_f1"], out[f"{mode}_prec"], out[f"{mode}_rec"] = f1, prec, rec
            if np.isfinite(acc) and np.isfinite(r.get(f"{mode}_acc", np.nan)):
                max_acc_err = max(max_acc_err, abs(acc - r[f"{mode}_acc"]))
                max_ece_err = max(max_ece_err, abs(ece - r[f"{mode}_ece"]))
        rows.append(out)
        print(f"  {ds:16s} seed {seed:2d}  rf_f1={out['rf_f1']:.4f} tfm_f1={out['tfm_f1']:.4f} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\n{'='*64}")
    print(f"VALIDATION vs recorded d1_8seed: max |Δacc|={max_acc_err:.2e}  "
          f"max |Δece|={max_ece_err:.2e}  (should be ~0)")
    print(f"Saved {len(df)} rows -> {out_csv}  ({(time.time()-t0)/60:.1f} min)")

    agg_rows = []
    for ds, g in df.groupby("dataset"):
        row = {"dataset": ds, "n_seeds": len(g)}
        for mode in ("rf", "tfm"):
            for m in ("acc", "f1", "prec", "rec", "ece"):
                vals = g[f"{mode}_{m}"].dropna().values
                if len(vals):
                    row[f"{mode}_{m}_mean"] = float(np.mean(vals))
                    row[f"{mode}_{m}_ci95"] = (1.96 * float(np.std(vals, ddof=1)) /
                                               np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                else:
                    row[f"{mode}_{m}_mean"] = row[f"{mode}_{m}_ci95"] = float("nan")
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    agg_path = out_csv.replace("per_seed", "aggregated")
    agg.to_csv(agg_path, index=False)
    print(f"Saved aggregated -> {agg_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv",
                    default="outputs/paper_ready/d1_8seed/results_per_seed.csv")
    ap.add_argument("--out", dest="out_csv",
                    default="outputs/paper_ready/d1_8seed/results_allmetrics_per_seed.csv")
    a = ap.parse_args()
    main(a.in_csv, a.out_csv)
