"""
experiments/run_d2_prioralign_nested.py

D2 — Prior-aligned drift, held-out protocol (PVLDB revision item M1/M2c, R3-W2).
==================================================================
Reviewer 3 (W2): the implemented Wasserstein "drift" term penalised distance to the
DIRTY pre-cleaning data, not to the TFM prior — so "prior alignment" was not actually
implemented, and drift-heavy rewards degenerate to no-op.

This experiment implements the intended semantics: drift is measured as the mean
column-wise normalised Wasserstein-1 distance between the CLEANED data and a CLEAN
REFERENCE distribution (the pre-injection clean dataset's marginals, which we have
because error injection is seeded from clean OpenML data). Lower drift now means
"closer to the clean/prior distribution", as the formal definition intends.

It reuses the EXACT held-out nested protocol of run_c2_tfm_reward_nested (outer
untouched test / inner-val selection / TabPFN final eval), and adds the prior-aligned
drift term to the TFM selection score:

    tfm_score = w_acc * inner_val_acc
              + w_ret * retention^alpha
              + w_qual * quality
              - drift_coeff * drift_to_CLEAN_reference      <-- the D2 change

We then compare, held-out protocol:
    * RF-reward         (unchanged baseline)
    * TFM-prior-aligned (this experiment)
to see whether the corrected objective recovers a real accuracy/calibration signal
that the (leaky) submitted version only appeared to have.

Usage
-----
  PYTHONPATH=src python experiments/run_d2_prioralign_nested.py --seeds 42 1 2 3 4
  PYTHONPATH=src python experiments/run_d2_prioralign_nested.py --datasets ionosphere --seeds 42 --drift-coeff 0.05
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split

import run_c2_tfm_reward_nested as G
from learn2clean_v3.data.error_injection import ErrorProfile, apply_error_profile
from learn2clean_v3.data.openml_loader import BENCHMARK_DATASETS, load_dataset
from learn2clean_v3.rewards import MultiObjectiveReward


def drift_to_reference(X_clean: pd.DataFrame, ref_cols: Dict[str, np.ndarray]) -> float:
    """Mean column-wise normalised Wasserstein-1 distance from cleaned data to the
    CLEAN reference marginals (the prior proxy). Lower = better aligned."""
    numeric = X_clean.select_dtypes(include="number")
    dists: List[float] = []
    for col in numeric.columns:
        if col not in ref_cols:
            continue
        cur = numeric[col].dropna().values
        ref = ref_cols[col]
        if len(cur) < 2 or len(ref) < 2:
            continue
        try:
            w = float(stats.wasserstein_distance(cur, ref))
            ref_std = float(np.std(ref)) or 1.0
            dists.append(min(w / ref_std, 5.0))
        except Exception:
            continue
    return float(np.mean(dists)) if dists else 0.0


def select_best_prioraligned(
    X_sel, y_sel, pipelines, actions, seed, rf_reward, ref_cols, drift_coeff,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return (best_rf, best_tfm_prioraligned), selected on D_sel only."""
    n0 = len(X_sel)
    best_rf, best_rf_score = (), -np.inf
    best_tfm, best_tfm_score = (), -np.inf
    w_acc, w_ret, w_qual, alpha = 0.50, 0.35, 0.15, 2.0

    for seq in pipelines:
        X_clean = G.apply_pipeline(X_sel, y_sel, seq, actions)
        if X_clean is None or len(X_clean) == 0:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf_reward.reset(X_sel, y_sel)
            rf_score = rf_reward(X_clean, y_sel)
        if np.isfinite(rf_score) and rf_score > best_rf_score:
            best_rf_score, best_rf = rf_score, seq

        acc = G.inner_val_tabpfn_acc(X_clean, y_sel, seed)
        if not np.isfinite(acc):
            continue
        retention = (len(X_clean) / n0) ** alpha
        miss = float(X_clean.isna().mean().mean())
        dup = float(X_clean.duplicated().sum()) / max(len(X_clean), 1)
        quality = (1.0 - miss) * (1.0 - dup)
        drift = drift_to_reference(X_clean, ref_cols)            # <-- D2: clean-referenced
        tfm_score = w_acc * acc + w_ret * retention + w_qual * quality - drift_coeff * drift
        if tfm_score > best_tfm_score:
            best_tfm_score, best_tfm = tfm_score, seq

    return best_rf, best_tfm


def run_one(ds_name, seed, pipelines, actions, drift_coeff) -> Optional[Dict]:
    try:
        X, y, spec = load_dataset(ds_name, use_cache=True)
    except Exception as exc:
        print(f"  [SKIP] {ds_name}: {exc}")
        return None
    if len(X) > G.SUBSAMPLE_CAP:
        X, _, y, _ = train_test_split(X, y, train_size=G.SUBSAMPLE_CAP, random_state=seed, stratify=y)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)

    # CLEAN reference marginals (prior proxy) — from pre-injection data
    ref_cols = {c: X[c].dropna().values.astype(float)
                for c in X.select_dtypes(include="number").columns}

    Xd, yd = apply_error_profile(X, y, ErrorProfile("mcar", rate=G.MCAR_RATE, seed=seed))
    try:
        X_sel, X_test, y_sel, y_test = train_test_split(
            Xd, yd, test_size=G.OUTER_TEST_SIZE, random_state=seed, stratify=yd)
    except ValueError:
        X_sel, X_test, y_sel, y_test = train_test_split(
            Xd, yd, test_size=G.OUTER_TEST_SIZE, random_state=seed)
    X_sel, y_sel = X_sel.reset_index(drop=True), y_sel.reset_index(drop=True)
    X_test, y_test = X_test.reset_index(drop=True), y_test.reset_index(drop=True)

    rf_reward = MultiObjectiveReward(
        weight_accuracy=0.5, weight_retention=0.3, weight_quality=0.2,
        drift_penalty_coeff=0.1, eval_model="random_forest",
        eval_metric=spec.eval_metric, eval_cv_folds=3,
    )
    best_rf, best_tfm = select_best_prioraligned(
        X_sel, y_sel, pipelines, actions, seed, rf_reward, ref_cols, drift_coeff)

    out = {"dataset": ds_name, "seed": seed,
           "rf_pipeline": G.pipeline_label(best_rf),
           "tfm_pipeline": G.pipeline_label(best_tfm),
           "pipelines_match": int(best_rf == best_tfm)}
    for mode, best in (("rf", best_rf), ("tfm", best_tfm)):
        X_clean = G.apply_pipeline(X_sel, y_sel, best, actions)
        if X_clean is None:
            out[f"{mode}_acc"], out[f"{mode}_ece"] = float("nan"), float("nan"); continue
        X_test_prep = G.prepare_test_like_train(X_sel, X_test, best)
        acc, ece = G.final_test_tabpfn(X_clean, y_sel, X_test_prep, y_test, seed)
        out[f"{mode}_acc"], out[f"{mode}_ece"] = acc, ece
    return out


def main(dataset_names=None, output_dir=None, seeds=(42,), max_pipelines=20, drift_coeff=0.05) -> None:
    if dataset_names is None:
        dataset_names = list(BENCHMARK_DATASETS.keys())
    out_dir = Path(output_dir) if output_dir else (
        Path(__file__).parents[1] / "outputs" / "paper_ready" / "d2_prioralign_nested")
    out_dir.mkdir(parents=True, exist_ok=True)

    actions = G.build_actions()
    all_pipelines = G.enumerate_valid_pipelines(max_len=3)
    rows: List[Dict] = []
    t0 = time.time()
    for ds in dataset_names:
        for seed in seeds:
            pipelines = G.sample_pipelines(all_pipelines, max_pipelines, seed)
            print(f"\n── {ds} | seed {seed} | drift_coeff={drift_coeff} ──", flush=True)
            ts = time.time()
            r = run_one(ds, seed, pipelines, actions, drift_coeff)
            if r is not None:
                rows.append(r)
                print(f"   rf acc={r['rf_acc']:.4f}  |  tfm(prior) acc={r['tfm_acc']:.4f} "
                      f"({time.time()-ts:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(out_dir / "results_per_seed.csv", index=False)

    if not rows:
        print("No results."); return
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results_per_seed.csv", index=False)
    agg = G.aggregate(df)
    agg.to_csv(out_dir / "results_aggregated.csv", index=False)

    pivot = agg.set_index("dataset")
    rf = pivot["rf_acc_mean"].dropna(); tfm = pivot["tfm_acc_mean"].dropna()
    shared = rf.index.intersection(tfm.index)
    print(f"\n{'='*60}\nD2 PRIOR-ALIGNED (held-out protocol) — {len(shared)} datasets, drift_coeff={drift_coeff}")
    if len(shared) >= 2:
        try:
            stat, p = wilcoxon(tfm[shared].values, rf[shared].values, alternative="greater")
            print(f"Wilcoxon TFM(prior)>RF: stat={stat:.3f} p={p:.4f}")
        except Exception as exc:
            print(f"Wilcoxon failed: {exc}")
        wins = int((tfm[shared].values > rf[shared].values).sum())
        print(f"TFM wins: {wins}/{len(shared)} | TFM mean={tfm[shared].mean():.4f} RF={rf[shared].mean():.4f} "
              f"| Δ={tfm[shared].mean()-rf[shared].mean():+.4f}")
    print(f"Wall-clock: {(time.time()-t0)/60:.1f} min → {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--max-pipelines", type=int, default=20)
    ap.add_argument("--drift-coeff", type=float, default=0.05)
    a = ap.parse_args()
    main(a.datasets, a.output_dir, tuple(a.seeds), a.max_pipelines, a.drift_coeff)
