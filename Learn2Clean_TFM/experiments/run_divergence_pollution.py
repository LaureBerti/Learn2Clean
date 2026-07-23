"""
experiments/run_divergence_pollution.py

CONTROLLED STRESS-TEST of the reward mechanism (NOT a general-superiority claim). On `adult`
the TFM-reward (R7) beats the RF-reward (R3) because the two rewards pick DIFFERENT imputers:
R3→impute(knn) (trees like KNN's locally-coherent fills), R7→impute(mean/median) (KNN fabricates
off-prior local structure that distorts TabPFN's global normalization). We *engineer* that regime
on other datasets with a synthetic pollution and test the falsifiable prediction:

  IF we inject CLUSTER-LOCAL missingness (missingness concentrated inside local feature-space
  clusters, so KNN imputes plausibly per-cluster but fabricates multi-modal global density),
  THEN R7 should start beating R3, with the same R3→knn / R7→mean·median imputer split —
  whereas plain MCAR (inert per dossier ⑦) should leave R3≈R7.

Pure-accuracy selection (argmax inner-val acc), held-out nested protocol (reuses W.inner_val_acc
for selection and W.panel_on_test for the sacred-test number). Records the SELECTED imputer per
reward so we can see the mechanism fire, not just the accuracy delta.

Usage
-----
  PYTHONPATH=src:experiments python experiments/run_divergence_pollution.py \
      --datasets ionosphere diabetes heart_statlog phoneme bank_marketing hepatitis adult \
      --seeds 42 1 2
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
import run_c2_tfm_reward_nested as G
import run_weight_robustness as W

CONDITIONS = [("mcar", 0.30), ("clm", 0.30), ("clm", 0.50)]  # mcar = inert control; clm = the trigger


def inject_mcar(X, rate, seed):
    rng = np.random.default_rng(seed); Xo = X.copy()
    for c in X.select_dtypes(include="number").columns:
        Xo.loc[rng.random(len(X)) < rate, c] = np.nan
    return Xo


def inject_clm(X, rate, seed, k=4):
    """Cluster-local MAR missingness: cluster rows in standardized numeric space, then drop values
    at high rate ONLY inside the smaller half of clusters. KNN imputation recovers cluster-local
    values (RF-friendly); the per-cluster fills create multi-modal global density (TabPFN-unfriendly)."""
    rng = np.random.default_rng(seed)
    num = X.select_dtypes(include="number")
    if num.shape[1] == 0:
        return X.copy()
    Z = ((num - num.mean()) / (num.std(ddof=0) + 1e-9)).fillna(0.0)
    kk = int(min(k, max(2, len(X) // 50)))
    lab = KMeans(n_clusters=kk, n_init=3, random_state=seed).fit_predict(Z.values)
    sizes = pd.Series(lab).value_counts()
    prone = set(sizes.index[len(sizes) // 2:])                # smaller clusters → missing-prone
    in_prone = np.array([lab[i] in prone for i in range(len(X))])
    Xo = X.copy()
    for c in num.columns:
        m = in_prone & (rng.random(len(X)) < rate)
        Xo.loc[m, c] = np.nan
    return Xo


def select_pure(X_sel, y_sel, pipes, actions, seed, estimator):
    """argmax inner-val accuracy (no composite reward) — isolates the imputer-choice effect."""
    best, bs = (), -np.inf
    for seq in pipes:
        Xc = G.apply_pipeline(X_sel, y_sel, seq, actions)
        if Xc is None or len(Xc) == 0:
            continue
        acc = W.inner_val_acc(Xc, y_sel, seed, estimator)
        if not np.isfinite(acc):
            continue
        if acc > bs:
            bs, best = acc, seq
    return best


def test_tabpfn(X_sel, y_sel, X_test, y_test, seq, actions, seed):
    Xc = G.apply_pipeline(X_sel, y_sel, seq, actions)
    if Xc is None or len(Xc) == 0:
        return np.nan
    Xtp = G.prepare_test_like_train(X_sel, X_test, seq)
    Xtr, ytr, le = G._encode_align(Xc, y_sel)
    shared = [c for c in Xc.select_dtypes(include="number").columns if c in Xtp.columns]
    try:
        yte = le.transform(np.asarray(y_test)); Xte = Xtp[shared].values.astype(float)
        if len(np.unique(ytr)) < 2 or len(yte) == 0:
            return np.nan
        panel = W.panel_on_test(Xtr, ytr, Xte, yte, seed)
        return panel.get("tabpfn", {}).get("acc", np.nan)
    except Exception:
        return np.nan


def imputer_of(seq):
    for i in seq:
        if G.ACTION_GROUPS.get(i) == "impute":
            return G.ACTION_LABELS[i].replace("impute(", "").rstrip(")")
    return "none"


def load_ds(name):
    try:
        from run_saga_comparison import DATASETS as SAGA_DS, load_saga
        if name in SAGA_DS:
            out = load_saga(name)
            if out is not None:
                return out[0].reset_index(drop=True), pd.Series(out[1]).reset_index(drop=True)
    except Exception:
        pass
    from learn2clean_v3.data.openml_loader import load_dataset
    X, y, _ = load_dataset(name, use_cache=True)
    return X.reset_index(drop=True), pd.Series(y).reset_index(drop=True)


def run_one(ds, kind, rate, seed, pipes, actions):
    X, y = load_ds(ds)
    if len(X) > G.SUBSAMPLE_CAP:
        X, _, y, _ = train_test_split(X, y, train_size=G.SUBSAMPLE_CAP, random_state=seed,
                                      stratify=y if y.value_counts().min() >= 2 else None)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    Xd = inject_mcar(X, rate, seed) if kind == "mcar" else inject_clm(X, rate, seed)
    strat = y if y.value_counts().min() >= 2 else None
    try:
        X_sel, X_test, y_sel, y_test = train_test_split(Xd, y, test_size=0.30, random_state=seed, stratify=strat)
    except ValueError:
        X_sel, X_test, y_sel, y_test = train_test_split(Xd, y, test_size=0.30, random_state=seed)
    X_sel, y_sel = X_sel.reset_index(drop=True), y_sel.reset_index(drop=True)
    X_test, y_test = X_test.reset_index(drop=True), y_test.reset_index(drop=True)
    r3 = select_pure(X_sel, y_sel, pipes, actions, seed, "rf")
    r7 = select_pure(X_sel, y_sel, pipes, actions, seed, "tabpfn")
    a3 = test_tabpfn(X_sel, y_sel, X_test, y_test, r3, actions, seed)
    a7 = test_tabpfn(X_sel, y_sel, X_test, y_test, r7, actions, seed)
    return {"dataset": ds, "cond": f"{kind}_{rate}", "seed": seed,
            "r3_acc": a3, "r7_acc": a7, "delta": a7 - a3,
            "r3_imputer": imputer_of(r3), "r7_imputer": imputer_of(r7),
            "imputers_differ": int(imputer_of(r3) != imputer_of(r7))}


def main(datasets, seeds, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    actions = G.build_actions()
    pipes = G.sample_pipelines(G.enumerate_valid_pipelines(max_len=3), 24, 42)
    rows = []
    for ds in datasets:
        for kind, rate in CONDITIONS:
            for seed in seeds:
                try:
                    r = run_one(ds, kind, rate, seed, pipes, actions)
                except Exception as e:
                    r = {"dataset": ds, "cond": f"{kind}_{rate}", "seed": seed, "r3_acc": np.nan,
                         "r7_acc": np.nan, "delta": np.nan, "err": repr(e)[:140]}
                rows.append(r); pd.DataFrame(rows).to_csv(out / "divergence_per_run.csv", index=False)
                print(f"  {ds:14} {r['cond']:9} s{seed}: r3={r.get('r3_acc',np.nan):.4f}({r.get('r3_imputer','?')}) "
                      f"r7={r.get('r7_acc',np.nan):.4f}({r.get('r7_imputer','?')}) Δ={r.get('delta',np.nan):+.4f}", flush=True)
    df = pd.DataFrame(rows)
    agg = df.groupby("cond").agg(r3=("r3_acc", "mean"), r7=("r7_acc", "mean"), delta=("delta", "mean"),
                                 r7_wins=("delta", lambda s: int((s > 0).sum())),
                                 n=("delta", "count"), imp_differ=("imputers_differ", "mean")).reset_index()
    agg.to_csv(out / "divergence_by_cond.csv", index=False)
    # imputer-choice breakdown per condition (does R3→knn / R7→mean·median emerge?)
    mix = df.groupby(["cond", "r3_imputer"]).size().rename("r3_n").reset_index()
    mix.to_csv(out / "divergence_imputer_mix.csv", index=False)
    print("\n=== BY CONDITION (does cluster-local missingness trigger R7>R3?) ===")
    print(agg.to_string(index=False))
    print("\n=== imputer chosen by each reward, per condition ===")
    print(df.groupby(["cond"]).agg(r3_imp=("r3_imputer", lambda s: s.value_counts().to_dict()),
                                   r7_imp=("r7_imputer", lambda s: s.value_counts().to_dict())).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["ionosphere", "diabetes", "heart_statlog", "phoneme", "bank_marketing", "hepatitis", "adult"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/paper_ready/divergence_pollution"))
    a = ap.parse_args()
    main(a.datasets, tuple(a.seeds), a.output_dir)
