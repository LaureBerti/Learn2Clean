"""
experiments/run_prior_distance_faithful.py

A FAITHFUL prior-distance DIAGNOSTIC (route (a)). Instead of comparing a dataset to a
Gaussian proxy (our earlier M1--M4), we compare it to samples drawn from an SCM-based approximation of
TabPFN's OWN prior --- the random structural-causal-model / random-MLP tabular generator TabPFN is
meta-trained on. Distance is a two-sample statistic (energy distance, RBF-MMD, C2ST-AUC) between the
(standardised) cleaned data and a pool of prior samples at matching dimensionality.

This is a DIAGNOSTIC, not a reward (our 8-seed R7-redesign already showed a prior-distance reward is the
worst arm). It answers two questions, held-out protocol, 8 seeds:
  (i)  Does cleaning move data TOWARD the prior?  -> dist(best-acc pipeline) - dist(no-clean) < 0 ?
  (ii) Does prior-proximity TRACK TabPFN accuracy? -> Spearman(dist, TabPFN test acc) across candidate
       pipelines < 0 ? (closer to prior => higher accuracy)

Why "faithful": both the data and the prior pool are per-set z-standardised, so the distance reflects
distributional SHAPE (correlations / nonlinear manifold structure) rather than scale --- i.e. how
"prior-like" the data's structure is, which is what alignment to TabPFN's prior means.

The SCM prior sampler mirrors TabPFN's described prior: a random-depth MLP with random weights /
activations maps latent noise to `n_feat` correlated features (many independent SCMs pooled).

Feature count is held fixed per dataset (candidates use no dimensionality reduction) so distances are
comparable across candidate pipelines.

Usage
-----
  PYTHONPATH=src:experiments python experiments/run_prior_distance_faithful.py \
      --datasets blood_transfusion credit_g hepatitis ionosphere diabetes EEG \
      --seeds 42 1 2 3 4 5 6 7 --n-cand 12
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
# R (operators/load) and G (TabPFN) imported LAZILY in run_one so the SCM+distance core is testable.

OUTER = 0.2
MCAR = 0.15
SUBSAMPLE_CAP = 3000
DIST_CAP = 400          # subsample rows for the O(n^2) two-sample statistics


# --------------------------------------------------------------------------------------------------
# SCM prior sampler --- random-MLP tabular generator approximating TabPFN's prior
# --------------------------------------------------------------------------------------------------
def sample_scm_prior(n_rows, n_feat, seed, n_scm=8):
    """Pool `n_scm` random SCMs (random-depth MLPs, random weights/activations) each emitting `n_feat`
    correlated features from latent Gaussian noise -> a prior-sample matrix (n_rows, n_feat)."""
    rng = np.random.default_rng(seed)
    per = max(8, n_rows // n_scm + 1)
    outs = []
    for _ in range(n_scm):
        latent = int(rng.integers(2, max(3, n_feat + 1)))
        h = rng.standard_normal((per, latent))
        for _ in range(int(rng.integers(1, 4))):                     # random depth 1..3
            out_dim = int(rng.integers(max(2, n_feat // 2), n_feat * 2 + 1))
            # proper 1/sqrt(fan_in) init keeps activations O(1) across layers (no overflow)
            W = rng.standard_normal((h.shape[1], out_dim)) * rng.uniform(0.5, 2.0) / np.sqrt(h.shape[1])
            b = rng.standard_normal(out_dim) * rng.uniform(0.0, 0.5)
            h = h @ W + b
            act = int(rng.integers(0, 3))
            if act == 0:
                h = np.tanh(h)
            elif act == 1:
                h = np.maximum(h, 0.0)                                # ReLU  (act==2: identity)
        Wf = rng.standard_normal((h.shape[1], n_feat)) * rng.uniform(0.5, 2.0) / np.sqrt(h.shape[1])
        X = h @ Wf + rng.standard_normal((per, n_feat)) * rng.uniform(0.0, 0.3)
        outs.append(X)
    P = np.vstack(outs)
    rng.shuffle(P)
    return P[:n_rows]


def _std(A):
    A = np.asarray(A, dtype=float)
    mu, sd = np.nanmean(A, 0), np.nanstd(A, 0) + 1e-8
    return np.nan_to_num((A - mu) / sd, nan=0.0)


def _subsample(A, cap, rng):
    if len(A) <= cap:
        return A
    return A[rng.choice(len(A), cap, replace=False)]


# --------------------------------------------------------------------------------------------------
# Two-sample distances (higher = farther from the prior)
# --------------------------------------------------------------------------------------------------
def energy_distance(A, B):
    dab = cdist(A, B).mean(); daa = cdist(A, A).mean(); dbb = cdist(B, B).mean()
    return float(max(2 * dab - daa - dbb, 0.0))


def rbf_mmd(A, B):
    Z = np.vstack([A, B])
    d = cdist(Z, Z); med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    g = 1.0 / (2.0 * med ** 2 + 1e-12)
    Kaa = rbf_kernel(A, A, g); Kbb = rbf_kernel(B, B, g); Kab = rbf_kernel(A, B, g)
    return float(np.sqrt(max(Kaa.mean() + Kbb.mean() - 2 * Kab.mean(), 0.0)))


def c2st_auc(A, B, seed):
    """Classifier two-sample test: AUC distinguishing data(1) from prior(0). 0.5=indistinguishable."""
    X = np.vstack([A, B]); y = np.r_[np.ones(len(A)), np.zeros(len(B))]
    try:
        proba = cross_val_predict(RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
                                  X, y, cv=3, method="predict_proba")[:, 1]
        return float(roc_auc_score(y, proba))
    except Exception:
        return np.nan


def prior_distances(Z_data, seed):
    """All three faithful prior-distances of standardised data Z (n,d) to an SCM-prior pool."""
    rng = np.random.default_rng(seed)
    d = Z_data.shape[1]
    P = _std(sample_scm_prior(max(len(Z_data), DIST_CAP), d, seed))
    A = _subsample(Z_data, DIST_CAP, rng); B = _subsample(P, DIST_CAP, rng)
    return {"energy": energy_distance(A, B), "mmd": rbf_mmd(A, B), "c2st": c2st_auc(A, B, seed)}


# --------------------------------------------------------------------------------------------------
# Diagnostic run
# --------------------------------------------------------------------------------------------------
def run_one(name, seed, n_cand):
    import run_saga_richops as R
    import run_c2_tfm_reward_nested as G
    X, y = R.load_ds(name)
    cap = getattr(G, "SUBSAMPLE_CAP", SUBSAMPLE_CAP)
    if len(X) > cap:
        X, _, y, _ = train_test_split(X, y, train_size=cap, random_state=0,
                                      stratify=y if y.value_counts().min() >= 2 else None)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    Xd = R.mcar(X, MCAR, seed)
    Xtr, Xte, ytr, yte = train_test_split(Xd, y, test_size=OUTER, random_state=seed,
                                          stratify=y if y.value_counts().min() >= 2 else None)
    Xtr, ytr = Xtr.reset_index(drop=True), ytr.reset_index(drop=True)
    Xte, yte = Xte.reset_index(drop=True), yte.reset_index(drop=True)

    # candidate pipelines: fixed feature count (no dim reduction) so distances are comparable
    pool = [p for p in R.enumerate_pool(True) if p[4] is None]
    rng = np.random.default_rng(seed)
    default = tuple(None if None in o else o[0] for _, o in R.groups_for(True))
    idx = rng.choice(len(pool), min(n_cand, len(pool)), replace=False)
    cands = [default] + [pool[i] for i in idx]

    recs = []
    for pipe in cands:
        out = R.apply_pipeline(Xtr, ytr, Xte, pipe, seed)
        if out is None or len(out[0]) == 0 or out[0].shape[1] == 0:
            continue
        tr, ytr2, te = out
        if len(np.unique(ytr2.values)) < 2:
            continue
        Z = _std(tr.values)
        dist = prior_distances(Z, seed)
        try:
            yp, _ = G._tabpfn_fit_predict(tr.values, ytr2.values, te.values, seed)
            acc = float(accuracy_score(yte.values, yp))
        except Exception:
            acc = np.nan
        recs.append({"pipe": str(pipe), "is_default": int(pipe == default), "acc": acc, **dist})

    df = pd.DataFrame(recs).dropna(subset=["acc"])
    if len(df) < 3:
        return {"dataset": name, "seed": seed, "err": "too few valid candidates"}
    row = {"dataset": name, "seed": seed, "n_cand": len(df)}
    dirty = df[df.is_default == 1]
    best = df.loc[df.acc.idxmax()]
    for m in ("energy", "mmd", "c2st"):
        # (i) cleaning-moves-toward-prior: best-acc pipeline distance minus no-clean distance (<0 = toward)
        row[f"{m}_delta_best_vs_dirty"] = float(best[m] - dirty[m].mean()) if len(dirty) else np.nan
        # (ii) prior-proximity tracks accuracy: Spearman(dist, acc) across candidates (<0 = faithful)
        rho, _ = spearmanr(df[m], df["acc"])
        row[f"{m}_rho_dist_acc"] = float(rho)
        row[f"{m}_dirty"] = float(dirty[m].mean()) if len(dirty) else np.nan
        row[f"{m}_best"] = float(best[m])
    row["best_acc"] = float(best.acc); row["dirty_acc"] = float(dirty.acc.mean()) if len(dirty) else np.nan
    return row


def main(datasets, seeds, n_cand, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in datasets:
        for seed in seeds:
            try:
                r = run_one(ds, seed, n_cand)
            except Exception as e:
                r = {"dataset": ds, "seed": seed, "err": repr(e)[:160]}
            rows.append(r); pd.DataFrame(rows).to_csv(out / "prior_faithful_per_run.csv", index=False)
            print(f"  {ds:14} s{seed}: energy rho={r.get('energy_rho_dist_acc',np.nan):+.3f} "
                  f"Δbest={r.get('energy_delta_best_vs_dirty',np.nan):+.3f} | "
                  f"c2st rho={r.get('c2st_rho_dist_acc',np.nan):+.3f}", flush=True)
    df = pd.DataFrame(rows)
    if "energy_rho_dist_acc" not in df.columns:
        print("\n!!! ALL runs errored:"); [print("  ", r.get("dataset"), r.get("seed"), r.get("err")) for r in rows[:5]]
        return
    num = [c for c in df.columns if c not in ("dataset", "seed")]
    agg = df.groupby("dataset")[num].mean(numeric_only=True).reset_index()
    agg.to_csv(out / "prior_faithful_aggregated.csv", index=False)
    print("\n=== FAITHFUL prior-distance diagnostic (SCM prior; held-out protocol) ===")
    print("(i) cleaning moves toward prior  = mean Δ(best−dirty) < 0 ; "
          "(ii) proximity tracks accuracy = mean Spearman(dist,acc) < 0")
    for m in ("energy", "mmd", "c2st"):
        dcol, rcol = f"{m}_delta_best_vs_dirty", f"{m}_rho_dist_acc"
        print(f"  {m:7}: mean Δ(best−dirty) = {df[dcol].mean():+.4f}  |  mean ρ(dist,acc) = {df[rcol].mean():+.3f}"
              f"  (per-ds ρ: " + ", ".join(f"{r.dataset}={r[rcol]:+.2f}" for _, r in agg.iterrows()) + ")")
    print("Full table → prior_faithful_aggregated.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["blood_transfusion", "credit_g", "hepatitis", "ionosphere", "diabetes", "EEG"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--n-cand", type=int, default=12)
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/paper_ready/prior_faithful"))
    a = ap.parse_args()
    main(a.datasets, tuple(a.seeds), a.n_cand, a.output_dir)
