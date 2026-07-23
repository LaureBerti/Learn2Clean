"""
experiments/run_dq_vs_model.py

Tests the paper's CORE ASSUMPTION directly: does improving DATA quality improve MODEL quality?
i.e. is Delta(data-quality) correlated with Delta(model-quality: accuracy / macro-F1)?

For each dataset/seed we enumerate a spread of candidate cleaning pipelines and, per candidate,
measure BOTH:
  * DATA-quality deltas vs the no-clean baseline (all MODEL-INDEPENDENT, so the correlation is not
    circular):
      d_retention   : fraction of training rows kept
      d_uniqueness  : 1 - duplicate-row ratio
      d_outlierfree : 1 - fraction of |z|>3 cells   (feature cleanliness)
      d_composite   : mean of the three above
      d_observerQ   : the PAPER's observer quality = completeness x (1 - normalized W1 drift)
                      [reported too, but note it embeds the reviewer-flagged drift term]
  * MODEL-quality deltas on the SACRED outer test (TabPFN deployed):
      d_acc, d_f1
Held-out protocol: DQ is computed on the cleaned TRAIN; model metrics come from the untouched outer test.

Output: per-candidate rows (dq_vs_model_per_run.csv) so the Delta(DQ)-vs-Delta(model) relationship can
be correlated and scatter-plotted; plus Spearman rho per DQ metric x model metric, pooled & per dataset.

Usage
-----
  PYTHONPATH=src:experiments python experiments/run_dq_vs_model.py \
      --datasets blood_transfusion credit_g hepatitis diabetes ionosphere EEG \
      --seeds 42 1 2 3 4 5 6 7 --n-cand 16
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wasserstein_distance, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
import run_prior_distance_faithful as PF   # SCM prior sampler + two-sample distances (no TabPFN import)
# R (operators/load) and G (TabPFN) imported lazily in run_one.

OUTER = 0.2
MCAR = 0.15
SUBSAMPLE_CAP = 3000
DQ_KEYS = ["retention", "uniqueness", "outlierfree", "composite", "observerQ"]
PRIOR_KEYS = ["c2st", "energy", "mmd"]       # prior-distance (higher = farther from the SCM prior)


def _numeric(df):
    return df.select_dtypes(include="number")


def data_quality(clean_df, dirty_ref_num):
    """Model-INDEPENDENT data-quality components of a cleaned frame (+ the paper's observer quality
    vs the dirty reference). All in [0,1], higher = cleaner."""
    num = _numeric(clean_df)
    n = max(len(num), 1)
    completeness = 1.0 - float(num.isna().mean().mean()) if num.shape[1] else 1.0
    retention = len(num) / max(len(dirty_ref_num), 1)
    dup_ratio = float(num.round(6).duplicated().mean()) if len(num) else 0.0
    uniqueness = 1.0 - dup_ratio
    vals = num.values.astype(float)
    if vals.size:
        mu, sd = np.nanmean(vals, 0), np.nanstd(vals, 0) + 1e-9
        z = np.abs((vals - mu) / sd)
        outlierfree = 1.0 - float(np.nanmean(z > 3.0))
    else:
        outlierfree = 1.0
    composite = float(np.mean([retention, uniqueness, outlierfree]))
    # paper's observer quality = completeness x (1 - normalized col-wise W1 drift from dirty)
    drift = _drift(num, dirty_ref_num)
    observerQ = completeness * (1.0 - min(drift, 1.0))
    return {"retention": retention, "uniqueness": uniqueness, "outlierfree": outlierfree,
            "composite": composite, "observerQ": observerQ}


def _drift(clean_num, dirty_num):
    shared = [c for c in clean_num.columns if c in dirty_num.columns]
    if not shared:
        return 0.0
    ds = []
    for c in shared:
        a = pd.to_numeric(clean_num[c], errors="coerce").dropna().values
        b = pd.to_numeric(dirty_num[c], errors="coerce").dropna().values
        if len(a) > 1 and len(b) > 1:
            sd = np.std(b) + 1e-9
            ds.append(wasserstein_distance(a, b) / sd)
    return float(np.mean(ds)) if ds else 0.0


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
    dirty_ref = _numeric(Xtr)

    default = tuple(None if None in o else o[0] for _, o in R.groups_for(True))
    pool = R.enumerate_pool(True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), min(n_cand, len(pool)), replace=False)
    cands = [default] + [pool[i] for i in idx]

    def eval_cand(pipe):
        out = R.apply_pipeline(Xtr, ytr, Xte, pipe, seed)
        if out is None or len(out[0]) == 0 or out[0].shape[1] == 0:
            return None
        tr, ytr2, te = out
        if len(np.unique(ytr2.values)) < 2:
            return None
        dq = data_quality(tr, dirty_ref)
        try:
            prior = PF.prior_distances(PF._std(_numeric(tr).values), seed)   # DQ-independent prior-distance
        except Exception:
            prior = {k: np.nan for k in PRIOR_KEYS}
        try:
            yp, _ = G._tabpfn_fit_predict(tr.values, ytr2.values, te.values, seed)
            acc = float(accuracy_score(yte.values, yp)); f1 = float(f1_score(yte.values, yp, average="macro"))
        except Exception:
            return None
        return dq, prior, acc, f1

    base = eval_cand(default)
    if base is None:
        return [{"dataset": name, "seed": seed, "err": "baseline failed"}]
    dq0, pr0, acc0, f10 = base
    rows = []
    for pipe in cands:
        r = eval_cand(pipe)
        if r is None:
            continue
        dq, prior, acc, f1 = r
        row = {"dataset": name, "seed": seed, "pipe": str(pipe),
               "acc": acc, "f1": f1, "d_acc": acc - acc0, "d_f1": f1 - f10}
        for k in DQ_KEYS:
            row[f"d_{k}"] = dq[k] - dq0[k]
        for k in PRIOR_KEYS:                        # absolute prior-distance + delta vs no-clean
            row[k] = prior[k]; row[f"d_{k}"] = prior[k] - pr0[k]
        rows.append(row)
    return rows


def _rho(df, xa, xb):
    m = df[xa].notna() & df[xb].notna()
    return spearmanr(df[xa][m], df[xb][m]) if m.sum() > 5 else (np.nan, np.nan)


def _report(df):
    star = lambda p: "*" if (p is not None and not np.isnan(p) and p < 0.05) else " "
    out = {}
    print(f"\n=== Spearman rho over all candidates (n={len(df)}); Delta = vs no-clean ===")
    print("  (paper's assumption: Δdata-quality should track Δmodel-quality, rho>0)")
    print(f"  {'metric':13} {'rho(.,Δacc)':>13} {'rho(.,Δf1)':>13}")
    for dq in DQ_KEYS:
        ra, pa = _rho(df, f"d_{dq}", "d_acc"); rf, pf = _rho(df, f"d_{dq}", "d_f1")
        out[(dq, "acc")] = (ra, pa); out[(dq, "f1")] = (rf, pf)
        print(f"  DQ:{dq:10} {ra:+.3f}{star(pa)}       {rf:+.3f}{star(pf)}")
    print("  " + "-" * 40 + "  (prior-distance: rho<0 would mean closer-to-prior -> better model)")
    for pr in PRIOR_KEYS:
        ra, pa = _rho(df, f"d_{pr}", "d_acc"); rf, pf = _rho(df, f"d_{pr}", "d_f1")
        out[(pr, "acc")] = (ra, pa); out[(pr, "f1")] = (rf, pf)
        print(f"  prior:{pr:7} {ra:+.3f}{star(pa)}       {rf:+.3f}{star(pf)}")
    # THE TRIANGLE: connect data-quality <-> model <-> prior on the same pipelines
    tri = {
        "Δcomposite-DQ  ~ Δf1     (cleaner data -> better F1?)": _rho(df, "d_composite", "d_f1"),
        "Δcomposite-DQ  ~ Δc2st   (cleaner data -> closer prior?)": _rho(df, "d_composite", "d_c2st"),
        "Δc2st(prior)   ~ Δf1     (closer prior -> better F1?)": _rho(df, "d_c2st", "d_f1"),
    }
    print("\n  --- THE TRIANGLE (does cleaner data mean better F1 AND closer prior, and are they linked?) ---")
    for label, (r, p) in tri.items():
        print(f"    {label:56} rho={r:+.3f}{star(p)}")
    out["triangle"] = {k: v for k, v in tri.items()}
    return out


def main(datasets, seeds, n_cand, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in datasets:
        for seed in seeds:
            try:
                rs = run_one(ds, seed, n_cand)
            except Exception as e:
                rs = [{"dataset": ds, "seed": seed, "err": repr(e)[:160]}]
            rows += rs; pd.DataFrame(rows).to_csv(out / "dq_vs_model_per_run.csv", index=False)
            ok = [r for r in rs if "d_composite" in r]
            print(f"  {ds:14} s{seed}: {len(ok)} candidates", flush=True)
    df = pd.DataFrame(rows)
    if "d_composite" not in df.columns:
        print("\n!!! all errored"); [print("  ", r) for r in rows[:5]]; return
    df = df.dropna(subset=["d_composite", "d_acc", "d_f1"])
    out_stats = _report(df)
    # per-dataset composite correlation
    print("\n  per-dataset rho(Δcomposite, Δf1):")
    for ds, g in df.groupby("dataset"):
        if len(g) > 5:
            rho, _ = spearmanr(g.d_composite, g.d_f1)
            print(f"    {ds:14} {rho:+.3f}  (n={len(g)})")
    pd.DataFrame([{"x": k[0], "model": k[1], "rho": v[0], "p": v[1]}
                  for k, v in out_stats.items() if isinstance(k, tuple)]).to_csv(
                      out / "dq_vs_model_correlations.csv", index=False)
    print("\nPer-candidate data → dq_vs_model_per_run.csv ; correlations → dq_vs_model_correlations.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["blood_transfusion", "credit_g", "hepatitis", "diabetes", "ionosphere", "EEG"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--n-cand", type=int, default=16)
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/paper_ready/dq_vs_model"))
    a = ap.parse_args()
    main(a.datasets, tuple(a.seeds), a.n_cand, a.output_dir)
