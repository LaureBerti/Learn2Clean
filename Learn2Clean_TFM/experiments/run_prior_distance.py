"""

Prototype + sanity-check FOUR ways to measure "distance to the TabPFN prior" (R3-W2). For a set
of leak-free cleaning pipelines on each dataset we compute each prior-distance estimator on the
CLEANED training context, plus TabPFN held-out accuracy/ECE on an inner-val split, then check:

  (a) NON-DEGENERACY — does the estimator actually vary across pipelines? (the dirty-referenced
      drift term of the submission collapsed to no-op because it didn't.)
  (b) CORRELATION — does a smaller prior-distance go with higher TabPFN accuracy / lower ECE?
      (Spearman across pipelines, per dataset + pooled.) A faithful prior term should show
      distance↓ ⇒ acc↑ (negative ρ) and distance↓ ⇒ ECE↓ (positive ρ for C2ST/W1/MMD/NLL).

Estimators
  M1 model_nll  : TabPFN held-out negative log predictive density (model-internal; most faithful).
  M2 mmd        : RBF-MMD between standardized cleaned features and a Gaussian-prior PROXY.
  M3 marg_w1    : mean per-feature Wasserstein-1 to N(0,1) after standardization (marginal-only).
  M4 c2st_auc   : classifier two-sample AUC, cleaned features vs Gaussian-prior proxy (0.5=close).

NOTE on the proxy: TabPFN's true prior is a generator over SCM datasets. Lacking the generator in
the inference package, M2/M3/M4 use a STANDARD-NORMAL proxy of TabPFN's *normalized-feature* prior
(it normalizes + power/quantile-transforms inputs, so its expected marginal is ~Gaussian). This is
a first-order approximation; a fuller version samples TabPFN's actual SCM prior. M1 needs no proxy.

Leak-safe: everything is computed within the training portion; no sacred test is used here.

Usage:  PYTHONPATH=src:experiments python experiments/run_prior_distance.py \
            --datasets hepatitis ionosphere diabetes blood_transfusion --seeds 42 1 2
"""
from __future__ import annotations
import argparse, itertools, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wasserstein_distance, spearmanr
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics.pairwise import rbf_kernel

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
import run_c2_tfm_reward_nested as G
import run_saga_richops as R   # reuse apply_pipeline (all leak-free operators)

REF = np.random.default_rng(0).standard_normal(20000)  # fixed N(0,1) reference for M3


def candidate_pipes():
    """~36 pipelines spanning impute × transform × outlier × scale — chosen so the DISTRIBUTION
    changes a lot (transforms Gaussianize; winsor caps tails) → exercises non-degeneracy (a)."""
    imp = ["mean", "median", "knn"]
    trans = [None, "yeojohnson", "quantile"]
    otl = [None, "winsor"]
    scl = [None, "zscore"]
    return [(i, o, t, s, None, None, None) for i in imp for t in trans for o in otl for s in scl]


def standardize(df):
    Z = StandardScaler().fit_transform(df.values.astype(float))
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


def m_marg_w1(Z):
    return float(np.mean([wasserstein_distance(Z[:, j], REF) for j in range(Z.shape[1])]))


def m_mmd(Z, seed):
    Gz = np.random.default_rng(seed).standard_normal(Z.shape)
    sub = np.vstack([Z, Gz])[:600]
    med = np.median(pdist(sub)) ** 2 + 1e-9
    g = 1.0 / med
    Kxx, Kyy, Kxy = rbf_kernel(Z, Z, g), rbf_kernel(Gz, Gz, g), rbf_kernel(Z, Gz, g)
    n, m = len(Z), len(Gz)
    return float(Kxx.sum() / (n * n) + Kyy.sum() / (m * m) - 2 * Kxy.sum() / (n * m))


def m_c2st(Z, seed):
    Gz = np.random.default_rng(seed + 7).standard_normal(Z.shape)
    X = np.vstack([Z, Gz]); y = np.r_[np.ones(len(Z)), np.zeros(len(Gz))]
    try:
        return float(cross_val_score(LogisticRegression(max_iter=300), X, y, cv=3, scoring="roc_auc").mean())
    except Exception:
        return np.nan


def m_model(tr, ytr2, val, yval, seed):
    """Return (nll, acc, f1, ece) from TabPFN fit on cleaned train, eval cleaned inner-val."""
    from sklearn.metrics import f1_score
    yp, yprob = G._tabpfn_fit_predict(tr.values, ytr2.values, val.values, seed)
    classes = list(np.unique(ytr2.values)); idx = {c: i for i, c in enumerate(classes)}
    yi = np.array([idx.get(v, -1) for v in yval.values]); ok = yi >= 0
    acc = float((yp == yval.values).mean())
    f1 = float(f1_score(yval.values, yp, average="macro"))
    if yprob.ndim == 2 and ok.any():
        p_true = np.clip(yprob[ok, yi[ok]], 1e-9, 1.0)
        nll = float(-np.log(p_true).mean()); ece = float(G.compute_ece(yi[ok], yprob[ok]))
    else:
        nll = ece = np.nan
    return nll, acc, f1, ece


def load_ds(name):
    try:
        from run_saga_comparison import DATASETS as SAGA_DS, load_saga
        if name in SAGA_DS:
            o = load_saga(name)
            if o is not None:
                return o[0].reset_index(drop=True), pd.Series(o[1]).astype(str).reset_index(drop=True)
    except Exception:
        pass
    from learn2clean_v3.data.openml_loader import load_dataset
    X, y, _ = load_dataset(name, use_cache=True)
    return X.reset_index(drop=True), pd.Series(y).astype(str).reset_index(drop=True)


def run_dataset(name, seed, pipes):
    X, y = load_ds(name)
    if len(X) > G.SUBSAMPLE_CAP:
        X, _, y, _ = train_test_split(X, y, train_size=G.SUBSAMPLE_CAP, random_state=0,
                                      stratify=y if y.value_counts().min() >= 2 else None)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    Xd = R.mcar(X, 0.15, seed)
    Xtr, _Xte, ytr, _yte = train_test_split(Xd, y, test_size=0.20, random_state=seed,
                                            stratify=y if y.value_counts().min() >= 2 else None)
    Xi_tr, Xi_val, yi_tr, yi_val = train_test_split(Xtr, ytr, test_size=0.25, random_state=seed,
                                                    stratify=ytr if ytr.value_counts().min() >= 2 else None)
    Xi_tr, yi_tr = Xi_tr.reset_index(drop=True), yi_tr.reset_index(drop=True)
    Xi_val, yi_val = Xi_val.reset_index(drop=True), yi_val.reset_index(drop=True)
    rows = []
    for pipe in pipes:
        out = R.apply_pipeline(Xi_tr, yi_tr, Xi_val, pipe, seed)
        if out is None:
            continue
        tr, ytr2, val = out
        if tr.shape[1] == 0 or len(np.unique(ytr2.values)) < 2:
            continue
        Z = standardize(tr)
        try:
            nll, acc, f1, ece = m_model(tr, ytr2, val, yi_val, seed)
        except Exception:
            continue
        rows.append({"dataset": name, "seed": seed, "pipe": "+".join(str(x) for x in pipe if x) or "noop",
                     "M1_nll": nll, "M2_mmd": m_mmd(Z, seed), "M3_marg_w1": m_marg_w1(Z),
                     "M4_c2st": m_c2st(Z, seed), "acc": acc, "f1": f1, "ece": ece})
    return rows


def main(datasets, seeds, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    pipes = candidate_pipes()
    allrows = []
    for ds in datasets:
        for seed in seeds:
            allrows += run_dataset(ds, seed, pipes)
            pd.DataFrame(allrows).to_csv(out / "prior_distance_per_pipe.csv", index=False)
            print(f"  {ds} s{seed}: {len(allrows)} rows so far", flush=True)
    df = pd.DataFrame(allrows)
    measures = ["M1_nll", "M2_mmd", "M3_marg_w1", "M4_c2st"]
    # (a) non-degeneracy: coefficient of variation across pipelines, per dataset, averaged
    print("\n=== (a) NON-DEGENERACY: spread across pipelines (mean CV over datasets) ===")
    for m in measures:
        cv = df.groupby(["dataset", "seed"])[m].apply(lambda s: s.std() / (abs(s.mean()) + 1e-9)).mean()
        rng = df.groupby(["dataset", "seed"])[m].apply(lambda s: s.max() - s.min()).mean()
        print(f"  {m:12}: mean CV={cv:.3f}  mean range={rng:.4f}  {'NON-degenerate' if cv>0.05 else 'flat?'}")
    # (b) correlation with acc / F1 / ece (Spearman across pipelines), pooled within (dataset,seed).
    # Ensembles use within-group z-scores summed (all measures: higher = farther from prior).
    print("\n=== (b) CORRELATION  ρ(score, acc) / ρ(score, F1) / ρ(score, ece)  — pooled ===")
    def zsum(g, cols):
        return sum((g[c] - g[c].mean()) / (g[c].std(ddof=0) + 1e-9) for c in cols)
    def pooled_rho(scorefn, target):
        rs = []
        for _, g in df.groupby(["dataset", "seed"]):
            if len(g) < 6:
                continue
            s = scorefn(g)
            if np.std(s) > 1e-9 and g[target].nunique() > 2:
                rs.append(spearmanr(s, g[target]).correlation)
        return np.nanmean(rs) if rs else np.nan
    scores = [(m, (lambda g, m=m: g[m])) for m in measures]
    scores += [("M2+M3", lambda g: zsum(g, ["M2_mmd", "M3_marg_w1"])),
               ("M1+M2+M3", lambda g: zsum(g, ["M1_nll", "M2_mmd", "M3_marg_w1"]))]
    for name, fn in scores:
        ra, rf, re = pooled_rho(fn, "acc"), pooled_rho(fn, "f1"), pooled_rho(fn, "ece")
        flag = "right sign" if ra < -0.1 else ("no link" if abs(ra) < 0.1 else "WRONG sign")
        print(f"  {name:12}: ρ(.,acc)={ra:+.3f}  ρ(.,F1)={rf:+.3f}  ρ(.,ece)={re:+.3f}   [{flag}]")
    agg = df.groupby("dataset")[measures + ["acc", "ece"]].mean().reset_index()
    agg.to_csv(out / "prior_distance_summary.csv", index=False)
    print("\nper-pipe → prior_distance_per_pipe.csv ; means → prior_distance_summary.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["hepatitis", "ionosphere", "diabetes", "blood_transfusion"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/paper_ready/prior_distance"))
    a = ap.parse_args()
    main(a.datasets, tuple(a.seeds), a.output_dir)
