"""
experiments/run_tfm_levers_nested.py

TFM-reward improvement levers under the HELD-OUT PROTOCOL nested 8-seed protocol.
====================================================================
Same outer/inner split contract as run_c2_tfm_reward_nested.py (which it imports
from): outer 20% test is sacred; all selection, threshold tuning and calibration
penalties are computed on D_sel / its inner-val ONLY.

Arms (TFM evaluator = TabPFN v2, standard 7-op pool):
  tfm_acc     baseline: select by inner-val accuracy;              test = argmax
  tfm_f1thr   Lever A : select by inner-val best-threshold macroF1; test = tuned threshold
  tfm_calib   Lever B : select by inner-val (acc - LAMBDA*ECE);     test = argmax
  rf          reference: RF multi-objective reward;                 test = argmax

The three TFM arms share ONE inner-val TabPFN fit per candidate pipeline (acc,
tuned-F1 and ECE are all read off the same validation probabilities), so the
selection cost is ~the same as the single-arm Table 3 run.

Usage
-----
  PYTHONPATH=src:experiments python experiments/run_tfm_levers_nested.py \
      --datasets hepatitis blood_transfusion diabetes credit_g \
      --seeds 42 1 2 3 4 5 6 7 --output-dir outputs/paper_ready/tfm_levers_8seed
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

# Reuse the audited held-out protocol harness helpers verbatim.
import run_c2_tfm_reward_nested as NST
from run_c2_tfm_reward_nested import (
    INNER_VAL_SIZE, MCAR_RATE, OUTER_TEST_SIZE, SUBSAMPLE_CAP,
    apply_pipeline, build_actions, compute_ece, enumerate_valid_pipelines,
    pipeline_label, prepare_test_like_train, sample_pipelines,
    _encode_align, _tabpfn_fit_predict,
)
from learn2clean_v3.data.error_injection import ErrorProfile, apply_error_profile
from learn2clean_v3.data.openml_loader import BENCHMARK_DATASETS, load_dataset
from learn2clean_v3.rewards import MultiObjectiveReward

LAMBDA_ECE = 1.0            # weight of the calibration penalty in tfm_calib / tfm_combo
ARMS = ("tfm_acc", "tfm_f1thr", "tfm_calib", "tfm_combo", "rf", "rf_thr")
# tfm_combo = Lever A (tuned-threshold macro-F1 selection + test threshold)
#           + Lever B (calibration penalty in the selection score).
# rf_thr    = CONTROL: the RF-selected pipeline, but with test-time threshold
#             tuning. Isolates the decision-rule effect from the reward source:
#             rf_thr vs rf  = pure threshold effect;
#             tfm_f1thr vs rf_thr = extra value of TFM F1-selection.
_METRICS = ("acc", "ece", "f1", "prec", "rec")
_NAN5 = (float("nan"),) * 5


# --------------------------------------------------------------------------- #
# Threshold tuning (binary only; multiclass falls back to argmax)
# --------------------------------------------------------------------------- #
def _tune_threshold_f1(y_val: np.ndarray, prob_pos: np.ndarray) -> Tuple[float, float]:
    """Return (threshold*, macroF1*) maximizing macro-F1 on the validation fold."""
    cands = np.unique(np.concatenate([[0.0], np.sort(prob_pos), [1.0]]))
    best_t, best_f1 = 0.5, -1.0
    for t in cands:
        pred = (prob_pos >= t).astype(int)
        f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def _val_scores(y_val: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """acc, tuned-threshold macro-F1, and (acc - LAMBDA*ECE) from one val fit."""
    acc = float(accuracy_score(y_val, y_pred))
    ece = compute_ece(y_val, y_prob)
    if y_prob.ndim > 1 and y_prob.shape[1] == 2:
        _, f1thr = _tune_threshold_f1(y_val, y_prob[:, 1])
    else:
        f1thr = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
    return {"acc": acc, "f1thr": f1thr,
            "calib": acc - LAMBDA_ECE * ece,        # Lever B
            "combo": f1thr - LAMBDA_ECE * ece}      # Lever A + B


# --------------------------------------------------------------------------- #
# Final test-fold metrics with an explicit decision rule
# --------------------------------------------------------------------------- #
def _final_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Tuple[float, ...]:
    return (float(accuracy_score(y_true, y_pred)), compute_ece(y_true, y_prob),
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            float(recall_score(y_true, y_pred, average="macro", zero_division=0)))


def final_test(X_sel_clean, y_sel, X_test_prep, y_test, seed, use_threshold: bool) -> Tuple[float, ...]:
    """Fit TabPFN on cleaned D_sel, evaluate on untouched D_test.
    If use_threshold: tune the decision threshold on a D_sel inner-val (held-out protocol)
    and apply it to the test probabilities; else argmax."""
    _, ytr, le = _encode_align(X_sel_clean, y_sel)
    num = X_sel_clean.select_dtypes(include="number")
    shared = [c for c in num.columns if c in X_test_prep.columns]
    if not shared:
        return _NAN5
    Xtr = num[shared].values.astype(float)          # train aligned to shared cols
    Xte = X_test_prep[shared].values.astype(float)  # test aligned to the same cols
    try:
        yte = le.transform(np.asarray(y_test))
    except Exception:
        return _NAN5
    if len(np.unique(ytr)) < 2 or len(yte) == 0:
        return _NAN5
    try:
        thr = None
        if use_threshold and len(np.unique(ytr)) == 2:
            # tune t* on an inner split of D_sel — never touches the test fold
            Xit, Xiv, yit, yiv = train_test_split(
                Xtr, ytr, test_size=INNER_VAL_SIZE, random_state=seed, stratify=ytr)
            if len(np.unique(yit)) == 2:
                _, vprob = _tabpfn_fit_predict(Xit, yit, Xiv, seed)
                if vprob.ndim > 1 and vprob.shape[1] == 2:
                    thr, _ = _tune_threshold_f1(yiv, vprob[:, 1])
        # refit on full D_sel, predict test
        y_pred, y_prob = _tabpfn_fit_predict(Xtr, ytr, Xte, seed)
        if thr is not None and y_prob.ndim > 1 and y_prob.shape[1] == 2:
            y_pred = (y_prob[:, 1] >= thr).astype(int)
        return _final_metrics(yte, y_pred, y_prob)
    except Exception as exc:
        NST.logger.debug("final_test failed: %s", exc)
        return _NAN5


# --------------------------------------------------------------------------- #
# Selection over the pool (one TabPFN inner-val fit per pipeline, shared by arms)
# --------------------------------------------------------------------------- #
def select_arms(X_sel, y_sel, pipelines, actions, seed, rf_reward) -> Dict[str, Tuple[int, ...]]:
    n0 = len(X_sel)
    w_ret, w_qual, alpha = 0.35, 0.15, 2.0
    w_acc = 0.50
    best = {a: () for a in ARMS}
    score = {a: -np.inf for a in ARMS}

    for seq in pipelines:
        X_clean = apply_pipeline(X_sel, y_sel, seq, actions)
        if X_clean is None or len(X_clean) == 0:
            continue

        # RF reference arm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf_reward.reset(X_sel, y_sel)
            rf_score = rf_reward(X_clean, y_sel)
        if np.isfinite(rf_score) and rf_score > score["rf"]:
            score["rf"], best["rf"] = rf_score, seq

        # TFM arms — one inner-val TabPFN fit, three read-outs
        X_arr, y_enc, _ = _encode_align(X_clean, y_sel)
        if len(y_enc) < 20 or len(np.unique(y_enc)) < 2:
            continue
        try:
            Xtr, Xval, ytr, yval = train_test_split(
                X_arr, y_enc, test_size=INNER_VAL_SIZE, random_state=seed, stratify=y_enc)
            if len(np.unique(ytr)) < 2:
                continue
            y_pred, y_prob = _tabpfn_fit_predict(Xtr, ytr, Xval, seed)
        except Exception:
            continue
        vs = _val_scores(yval, y_pred, y_prob)

        retention = (len(X_clean) / n0) ** alpha
        miss = float(X_clean.isna().mean().mean())
        dup = float(X_clean.duplicated().sum()) / max(len(X_clean), 1)
        quality = (1.0 - miss) * (1.0 - dup)
        common = w_ret * retention + w_qual * quality
        arm_score = {
            "tfm_acc":   w_acc * vs["acc"] + common,
            "tfm_f1thr": w_acc * vs["f1thr"] + common,
            "tfm_calib": w_acc * vs["calib"] + common,
            "tfm_combo": w_acc * vs["combo"] + common,
        }
        for a, s in arm_score.items():
            if s > score[a]:
                score[a], best[a] = s, seq
    best["rf_thr"] = best["rf"]   # control: same pipeline as RF, threshold at test
    return best


def run_one(ds_name: str, seed: int, pipelines, actions) -> Optional[Dict]:
    try:
        X, y, spec = load_dataset(ds_name, use_cache=True)
    except Exception as exc:
        print(f"  [SKIP] {ds_name}: load failed: {exc}")
        return None
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
    X_sel = X_sel.reset_index(drop=True); y_sel = y_sel.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True); y_test = y_test.reset_index(drop=True)

    rf_reward = MultiObjectiveReward(
        weight_accuracy=0.5, weight_retention=0.3, weight_quality=0.2,
        drift_penalty_coeff=0.1, eval_model="random_forest",
        eval_metric=spec.eval_metric, eval_cv_folds=3)

    best = select_arms(X_sel, y_sel, pipelines, actions, seed, rf_reward)
    out = {"dataset": ds_name, "seed": seed}
    for arm in ARMS:
        seq = best[arm]
        out[f"{arm}_pipeline"] = pipeline_label(seq)
        X_clean = apply_pipeline(X_sel, y_sel, seq, actions)
        if X_clean is None:
            for m in _METRICS:
                out[f"{arm}_{m}"] = float("nan")
            continue
        X_test_prep = prepare_test_like_train(X_sel, X_test, seq)
        vals = final_test(X_clean, y_sel, X_test_prep, y_test, seed,
                          use_threshold=(arm in ("tfm_f1thr", "tfm_combo", "rf_thr")))
        for m, v in zip(_METRICS, vals):
            out[f"{arm}_{m}"] = v
    return out


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, g in df.groupby("dataset"):
        row = {"dataset": ds, "n_seeds": len(g)}
        for arm in ARMS:
            for m in _METRICS:
                vals = g[f"{arm}_{m}"].dropna().values
                if len(vals):
                    row[f"{arm}_{m}_mean"] = float(np.mean(vals))
                    row[f"{arm}_{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                else:
                    row[f"{arm}_{m}_mean"] = row[f"{arm}_{m}_std"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main(dataset_names=None, output_dir=None, seeds=(42,), max_pipelines=20) -> None:
    if dataset_names is None:
        dataset_names = ["hepatitis", "blood_transfusion", "diabetes", "credit_g"]
    out_dir = Path(output_dir) if output_dir else (
        Path(__file__).parents[1] / "outputs" / "paper_ready" / "tfm_levers_8seed")
    out_dir.mkdir(parents=True, exist_ok=True)

    actions = build_actions()
    all_pipelines = enumerate_valid_pipelines(max_len=3)
    rows: List[Dict] = []
    t0 = time.time()
    for ds in dataset_names:
        for seed in seeds:
            pipelines = sample_pipelines(all_pipelines, max_pipelines, seed)
            print(f"\n── {ds} | seed {seed} | {len(pipelines)} pipelines ──", flush=True)
            ts = time.time()
            r = run_one(ds, seed, pipelines, actions)
            if r is not None:
                rows.append(r)
                print(f"   base acc={r['tfm_acc_acc']:.3f} f1={r['tfm_acc_f1']:.3f} | "
                      f"f1thr f1={r['tfm_f1thr_f1']:.3f} | calib ece={r['tfm_calib_ece']:.3f} "
                      f"({time.time()-ts:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(out_dir / "results_per_seed.csv", index=False)

    if not rows:
        print("No results."); return
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results_per_seed.csv", index=False)
    aggregate(df).to_csv(out_dir / "results_aggregated.csv", index=False)

    # ── Paired CV comparison: RF vs each TFM arm across all (dataset,seed) folds ──
    # The 8 seeds are 8 independent stratified outer holdouts → repeated-holdout CV.
    from scipy.stats import wilcoxon
    stat_rows = []
    print(f"\n{'='*66}\nPaired CV (RF vs TFM arm) — {len(df)} folds "
          f"({df.dataset.nunique()} ds x {df.seed.nunique()} seeds)")
    for arm in ("tfm_acc", "tfm_f1thr", "tfm_calib", "tfm_combo", "rf_thr"):
        for m in ("acc", "f1", "ece"):
            a = df[f"{arm}_{m}"].values
            b = df[f"rf_{m}"].values
            mask = np.isfinite(a) & np.isfinite(b)
            a, b = a[mask], b[mask]
            if len(a) < 2:
                continue
            delta = a - b                      # arm - RF (ECE: lower better → negative delta = win)
            try:
                p = float(wilcoxon(a, b).pvalue)
            except Exception:
                p = float("nan")
            win = float(np.mean(delta < 0) if m == "ece" else np.mean(delta > 0))
            stat_rows.append({"arm": arm, "metric": m, "n": len(a),
                              "mean_delta_vs_rf": float(delta.mean()),
                              "std_delta": float(delta.std(ddof=1)),
                              "win_rate": win, "wilcoxon_p": p})
            print(f"  {arm:10s} {m:3s}: Δ(arm-RF)={delta.mean():+.4f}±{delta.std(ddof=1):.4f} "
                  f"win={win*100:4.0f}% p={p:.3f}")
    pd.DataFrame(stat_rows).to_csv(out_dir / "rf_vs_arm_cv_stats.csv", index=False)
    print(f"\nDONE {(time.time()-t0)/60:.1f} min → {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--max-pipelines", type=int, default=20)
    a = ap.parse_args()
    main(a.datasets, a.output_dir, tuple(a.seeds), a.max_pipelines)
