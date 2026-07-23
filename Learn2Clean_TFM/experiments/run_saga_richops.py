"""
experiments/run_saga_richops.py

Tests the operator-richness hypothesis (dossier ⑥): does extending our pool with SAGA-style
RICH operators (MICE/IterativeImputer, PCA, SMOTE) close the gap to SAGA? Runs held-out protocol nested
selection (R7 = TabPFN inner-val acc) over BASE vs RICH pools on the SAGA + a few benchmark
datasets, reports test TabPFN accuracy AND wall-clock for each pool.

Held-out protocol contract (identical to run_c2_tfm_reward_nested):
  outer 80/20 (sacred test) → inner 75/25 within train (selection only).
  Train-context transforms are FIT on the inner-train and APPLIED to inner-val (for selection)
  and to the sacred test (for the final number). Row-removing ops (outlier, SMOTE) shape the
  training context only — never the test.

Operators
  impute   : mean | median | knn | mice(IterativeImputer)
  outlier  : none | iqr | zscore              (train-rows only)
  scale    : none | minmax | zscore
  dimreduce: none | pca(k)                     (fit train, transform test)
  balance  : none | smote                      (train context only; needs imblearn)

Usage
-----
  PYTHONPATH=src:experiments python experiments/run_saga_richops.py \
      --datasets EEG Titanic AnimalShelter hepatitis ionosphere diabetes --seeds 42 1 2
"""
from __future__ import annotations
import argparse, sys, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
import run_c2_tfm_reward_nested as G  # reuse TabPFN helpers + split constants

OUTER, INNER = G.OUTER_TEST_SIZE, G.INNER_VAL_SIZE
MCAR = 0.15

# pool definition: one op per group max, applied in fixed order.
# outlier ops come in two flavours: *_rm REMOVE rows (train-context only); winsor/clipz REPLACE
# values (cap to train-fit bounds, applied to train AND test) — SAGA-style outlier replacement.
# impute: mean/median; knn k=5/3/10; mice=IterativeImputer(BayesianRidge); iterrf=IterativeImputer(RandomForest, missForest-style)
IMPUTE = ["mean", "median", "knn", "knn3", "knn10", "mice", "iterrf"]
OUTLIER = [None, "iqr_rm", "z_rm", "winsor", "clipz"]   # rm = remove rows; winsor/clipz = replace
TRANSFORM = [None, "boxcox", "yeojohnson", "quantile"]  # power transforms (Box-Cox/Yeo-Johnson) | Quantile→normal
SCALE = [None, "minmax", "zscore", "log"]              # +log = log1p on shift-to-positive
DIM = [None, "pca", "rndproj", "sparseproj"]           # PCA | Gaussian | Sparse random projection
BAL = [None, "smote"]
DEDUP = [None, "dedup", "dedup_merge"]                  # drop, or merge dups (majority-vote label)
# our original "7-op world": mean/median/knn impute, iqr/zscore REMOVAL, minmax/zscore scale; none
# of the rich ops (mice, knn3/10, outlier-replacement, boxcox, log, pca, smote, dedup).
BASE_IMPUTE, BASE_OUTLIER, BASE_TRANSFORM, BASE_SCALE, BASE_DIM, BASE_BAL, BASE_DEDUP = \
    ["mean", "median", "knn"], [None, "iqr_rm", "z_rm"], [None], [None, "minmax", "zscore"], [None], [None], [None]


def load_ds(name):
    try:
        from run_saga_comparison import DATASETS as SAGA_DS, load_saga
        if name in SAGA_DS:
            out = load_saga(name)
            if out is not None:
                X, y = out[0], out[1]
                return X.reset_index(drop=True), pd.Series(y).astype(str).reset_index(drop=True)
    except Exception:
        pass
    from learn2clean_v3.data.openml_loader import load_dataset
    X, y, _ = load_dataset(name, use_cache=True)
    return X.reset_index(drop=True), pd.Series(y).astype(str).reset_index(drop=True)


def mcar(X, rate, seed):
    rng = np.random.default_rng(seed); X = X.copy()
    for c in X.select_dtypes(include="number").columns:
        X.loc[rng.random(len(X)) < rate, c] = np.nan
    return X


def _imputer(name):
    from sklearn.ensemble import RandomForestRegressor
    return {"mean": SimpleImputer(strategy="mean"), "median": SimpleImputer(strategy="median"),
            "knn": KNNImputer(n_neighbors=5), "knn3": KNNImputer(n_neighbors=3),
            "knn10": KNNImputer(n_neighbors=10),
            "mice": IterativeImputer(max_iter=10, random_state=0, sample_posterior=False),
            "iterrf": IterativeImputer(estimator=RandomForestRegressor(
                n_estimators=10, max_depth=6, random_state=0, n_jobs=-1),
                max_iter=5, random_state=0)}[name]


def apply_pipeline(Xtr, ytr, Xte, pipe, seed):
    """Held-out protocol: fit column transforms on Xtr, apply to Xte. Row ops (outlier removal/dedup/smote)
    touch train only; value ops (winsor/clipz) cap to train-fit bounds and apply to both.
    Returns (Xtr_clean, ytr_clean, Xte_clean) as numeric DataFrames, or None on failure."""
    imp, otl, trans, scl, dim, bal, ddp = pipe
    tr = Xtr.select_dtypes(include="number").copy()
    te = Xte.select_dtypes(include="number").copy()
    shared = [c for c in tr.columns if c in te.columns]
    tr, te = tr[shared], te[shared]
    if tr.shape[1] == 0:
        return None
    ytr2 = ytr.copy()
    try:
        # impute (fit train, apply both)
        im = _imputer(imp); im.fit(tr.values)
        tr = pd.DataFrame(im.transform(tr.values), columns=shared, index=tr.index)
        te = pd.DataFrame(im.transform(te.values), columns=shared, index=te.index)
        # outlier handling — bounds always fit on TRAIN
        if otl in ("iqr_rm", "z_rm"):                       # REMOVE rows (train only)
            if otl == "z_rm":
                z = (tr - tr.mean()) / (tr.std(ddof=0) + 1e-9)
                keep = (z.abs() <= 3.0).all(axis=1)
            else:
                q1, q3 = tr.quantile(.25), tr.quantile(.75); iqr = (q3 - q1) + 1e-9
                keep = ((tr >= q1 - 1.5 * iqr) & (tr <= q3 + 1.5 * iqr)).all(axis=1)
            if keep.sum() >= max(20, int(0.5 * len(tr))):
                tr, ytr2 = tr[keep], ytr2[keep]
        elif otl in ("winsor", "clipz"):                    # REPLACE values (cap; apply to both)
            if otl == "winsor":
                q1, q3 = tr.quantile(.25), tr.quantile(.75); iqr = (q3 - q1) + 1e-9
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            else:
                m, s = tr.mean(), tr.std(ddof=0) + 1e-9
                lo, hi = m - 3.0 * s, m + 3.0 * s
            tr = tr.clip(lower=lo, upper=hi, axis=1)
            te = te.clip(lower=lo, upper=hi, axis=1)
        # transform (fit train, apply both): Box-Cox needs >0 (shift by train-min); Quantile→normal
        if trans == "boxcox":
            from sklearn.preprocessing import PowerTransformer
            shift = (tr.min(axis=0)).clip(upper=0.0) * -1.0 + 1e-6   # make train cols strictly positive
            trp, tep = tr + shift, (te + shift).clip(lower=1e-9)
            pt = PowerTransformer(method="box-cox", standardize=False); pt.fit(trp.values)
            tr = pd.DataFrame(pt.transform(trp.values), columns=shared, index=tr.index)
            te = pd.DataFrame(pt.transform(tep.values), columns=shared, index=te.index)
        elif trans == "yeojohnson":
            from sklearn.preprocessing import PowerTransformer
            pt = PowerTransformer(method="yeo-johnson", standardize=False); pt.fit(tr.values)
            tr = pd.DataFrame(pt.transform(tr.values), columns=shared, index=tr.index)
            te = pd.DataFrame(pt.transform(te.values), columns=shared, index=te.index)
        elif trans == "quantile":
            from sklearn.preprocessing import QuantileTransformer
            qt = QuantileTransformer(output_distribution="normal",
                                     n_quantiles=int(min(len(tr), 1000)), random_state=0)
            qt.fit(tr.values)
            tr = pd.DataFrame(qt.transform(tr.values), columns=shared, index=tr.index)
            te = pd.DataFrame(qt.transform(te.values), columns=shared, index=te.index)
        # dedup — drop duplicate train rows; "dedup_merge" resolves label conflicts by majority vote
        if ddp in ("dedup", "dedup_merge"):
            key = tr.round(6)
            if ddp == "dedup_merge":
                gk = key.apply(lambda r: hash(tuple(r)), axis=1)
                ytr2 = ytr2.groupby(gk.values).transform(lambda s: s.mode().iloc[0])
            keep = ~key.duplicated()
            if keep.sum() >= max(20, int(0.5 * len(tr))):
                tr, ytr2 = tr[keep], ytr2[keep]
        # scale (fit train, apply both); "log" = log1p on shift-to-positive
        if scl == "log":
            shift = (tr.min(axis=0)).clip(upper=0.0) * -1.0
            tr = np.log1p((tr + shift).clip(lower=0.0))
            te = np.log1p((te + shift).clip(lower=0.0))
        elif scl is not None:
            sc = MinMaxScaler() if scl == "minmax" else StandardScaler(); sc.fit(tr.values)
            tr = pd.DataFrame(sc.transform(tr.values), columns=shared, index=tr.index)
            te = pd.DataFrame(sc.transform(te.values), columns=shared, index=te.index)
        # dim reduction (fit train, transform both): PCA | Gaussian | Sparse random projection
        if dim in ("pca", "rndproj", "sparseproj") and tr.shape[1] > 2:
            k = max(2, min(tr.shape[1] - 1, 10))
            if dim == "pca":
                red = PCA(n_components=k, random_state=0)
            elif dim == "rndproj":
                from sklearn.random_projection import GaussianRandomProjection
                red = GaussianRandomProjection(n_components=k, random_state=0)
            else:
                from sklearn.random_projection import SparseRandomProjection
                red = SparseRandomProjection(n_components=k, random_state=0, dense_output=True)
            red.fit(tr.values)
            cols = [f"d{i}" for i in range(k)]
            tr = pd.DataFrame(red.transform(tr.values), columns=cols, index=tr.index)
            te = pd.DataFrame(red.transform(te.values), columns=cols, index=te.index)
        # smote (train context only)
        if bal == "smote":
            from imblearn.over_sampling import SMOTE
            vc = ytr2.value_counts()
            if len(vc) >= 2 and vc.min() >= 6:
                k = min(5, vc.min() - 1)
                sm = SMOTE(random_state=seed, k_neighbors=k)
                Xr, yr = sm.fit_resample(tr.values, ytr2.values)
                tr = pd.DataFrame(Xr, columns=tr.columns); ytr2 = pd.Series(yr)
        return tr, ytr2.reset_index(drop=True), te
    except Exception:
        return None


def enumerate_pool(rich: bool):
    imps = IMPUTE if rich else BASE_IMPUTE
    otls = OUTLIER if rich else BASE_OUTLIER
    trans = TRANSFORM if rich else BASE_TRANSFORM
    scls = SCALE if rich else BASE_SCALE
    dims = DIM if rich else BASE_DIM
    bals = BAL if rich else BASE_BAL
    ddps = DEDUP if rich else BASE_DEDUP
    return list(itertools.product(imps, otls, trans, scls, dims, bals, ddps))


def groups_for(rich: bool):
    """Ordered (name, options) per operator group. Greedy search cost scales with the SUM of
    these sizes (~29 rich), not their PRODUCT (10k+), so adding operators stays cheap + fair."""
    if rich:
        return [("impute", IMPUTE), ("outlier", OUTLIER), ("transform", TRANSFORM),
                ("scale", SCALE), ("dim", DIM), ("balance", BAL), ("dedup", DEDUP)]
    return [("impute", BASE_IMPUTE), ("outlier", BASE_OUTLIER), ("transform", BASE_TRANSFORM),
            ("scale", BASE_SCALE), ("dim", BASE_DIM), ("balance", BASE_BAL), ("dedup", BASE_DEDUP)]


def _inner_score(pipe, Xi_tr, yi_tr, Xi_val, yi_val, seed, estimator, metric):
    """Inner-val SELECTION score under estimator ('rf'=R3 / 'tabpfn'=R7) and metric
    ('acc' / 'f1' / 'ece'). All returned so HIGHER is better → for 'ece' we return -ECE (argmin)."""
    from sklearn.metrics import accuracy_score, f1_score
    out = apply_pipeline(Xi_tr, yi_tr, Xi_val, pipe, seed)
    if out is None:
        return -1.0
    tr, ytr2, val = out
    try:
        if estimator == "rf":
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
            clf.fit(tr.values, ytr2.values)
            yp = clf.predict(val.values)
            prob = clf.predict_proba(val.values) if metric == "ece" else None
        else:
            yp, prob = G._tabpfn_fit_predict(tr.values, ytr2.values, val.values, seed)
        if metric == "f1":
            return float(f1_score(yi_val.values, yp, average="macro"))
        if metric == "ece":
            classes = list(np.unique(ytr2.values)); idx = {c: i for i, c in enumerate(classes)}
            yi = np.array([idx.get(v, -1) for v in yi_val.values]); ok = yi >= 0
            if prob is None or prob.ndim != 2 or not ok.any():
                return -1.0
            return -float(G.compute_ece(yi[ok], prob[ok]))   # maximize -ECE == minimize ECE
        return float(accuracy_score(yi_val.values, yp))
    except Exception:
        return -1.0


def select_pipeline(Xtr_out, ytr_out, rich, seed, estimator, metric, passes=2):
    """GREEDY COORDINATE SEARCH selecting by (estimator, metric). Returns the chosen pipeline."""
    Xi_tr, Xi_val, yi_tr, yi_val = train_test_split(
        Xtr_out, ytr_out, test_size=INNER, random_state=seed,
        stratify=ytr_out if ytr_out.value_counts().min() >= 2 else None)
    groups = groups_for(rich)
    cur = tuple(None if None in opts else opts[0] for _, opts in groups)   # impute has no None → 'mean'
    best_s = _inner_score(cur, Xi_tr, yi_tr, Xi_val, yi_val, seed, estimator, metric); n_eval = 1
    for _ in range(passes):
        improved = False
        for gi, (_, opts) in enumerate(groups):
            for opt in opts:
                if opt == cur[gi]:
                    continue
                trial = cur[:gi] + (opt,) + cur[gi + 1:]
                s = _inner_score(trial, Xi_tr, yi_tr, Xi_val, yi_val, seed, estimator, metric); n_eval += 1
                if s > best_s:
                    best_s, cur, improved = s, trial, True
        if not improved:
            break
    return cur, n_eval


def test_metrics(Xtr_out, ytr_out, Xte, yte, pipe, seed):
    """Deploy TabPFN on the sacred test; return acc, macro-F1, macro-precision, macro-recall, ECE."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    nan = {"acc": np.nan, "f1": np.nan, "prec": np.nan, "rec": np.nan, "ece": np.nan}
    out = apply_pipeline(Xtr_out, ytr_out, Xte, pipe, seed)
    if out is None:
        return nan
    tr, ytr2, te = out
    try:
        yp, yprob = G._tabpfn_fit_predict(tr.values, ytr2.values, te.values, seed)
        yt = yte.values
        m = {"acc": float(accuracy_score(yt, yp)),
             "f1": float(f1_score(yt, yp, average="macro")),
             "prec": float(precision_score(yt, yp, average="macro", zero_division=0)),
             "rec": float(recall_score(yt, yp, average="macro", zero_division=0))}
        # ECE: map true labels to predict_proba columns (sklearn/TabPFN sort classes = np.unique(ytr2))
        classes = list(np.unique(ytr2.values))
        idx = {c: i for i, c in enumerate(classes)}
        yt_idx = np.array([idx.get(v, -1) for v in yt]); ok = yt_idx >= 0
        m["ece"] = float(G.compute_ece(yt_idx[ok], yprob[ok])) if ok.any() and yprob.ndim == 2 else np.nan
        return m
    except Exception:
        return nan


def run_one(name, seed, cap, select_metrics=("acc", "f1"), rich_only=False,
            pool_baselines=False, skip_greedy=False):
    X, y = load_ds(name)
    if len(X) > G.SUBSAMPLE_CAP:
        X, _, y, _ = train_test_split(X, y, train_size=G.SUBSAMPLE_CAP, random_state=0,
                                      stratify=y if y.value_counts().min() >= 2 else None)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    Xd = mcar(X, MCAR, seed)
    Xtr, Xte, ytr, yte = train_test_split(Xd, y, test_size=OUTER, random_state=seed,
                                          stratify=y if y.value_counts().min() >= 2 else None)
    Xtr, ytr = Xtr.reset_index(drop=True), ytr.reset_index(drop=True)
    Xte, yte = Xte.reset_index(drop=True), yte.reset_index(drop=True)
    row = {"dataset": name, "seed": seed}

    # POOL-DEPENDENT baselines on the rich pool (B-RAND = random pipeline; B-SC = full pipeline).
    # These MUST reflect the operative pool: B-RAND draws one random operator per group, B-SC turns
    # every group on (first non-None option). Held-out protocol: fit on train context, deploy TabPFN on test.
    if pool_baselines:
        groups = groups_for(True)
        rng = np.random.default_rng(seed)
        rand_pipe = tuple(opts[int(rng.integers(len(opts)))] for _, opts in groups)
        full_pipe = tuple(next((o for o in opts if o is not None), opts[0]) for _, opts in groups)
        for bname, pipe in [("richB_rand", rand_pipe), ("richB_full", full_pipe)]:
            m = test_metrics(Xtr, ytr, Xte, yte, pipe, seed)
            for mk, mv in m.items():
                row[f"{bname}_{mk}"] = mv
            row[f"{bname}_pipe"] = str(pipe)

    # arms = pool {base,rich} × reward {r3=rf, r7=tabpfn} × selection-metric (select_metrics).
    # Each arm's winner is DEPLOYED on TabPFN and scored on ALL test metrics (acc/f1/prec/rec/ece).
    pools = [] if skip_greedy else ([("rich", True)] if rich_only else [("base", False), ("rich", True)])
    for tag, rich in pools:
        for rw, est in [("r3", "rf"), ("r7", "tabpfn")]:
            for sm in select_metrics:
                t0 = time.time()
                pipe, n_eval = select_pipeline(Xtr, ytr, rich, seed, est, sm)
                m = test_metrics(Xtr, ytr, Xte, yte, pipe, seed)
                key = f"{tag}_{rw}{sm}"                      # e.g. rich_r7ece
                for mk, mv in m.items():
                    row[f"{key}_{mk}"] = mv
                row[f"{key}_pipe"] = str(pipe); row[f"{key}_sec"] = time.time() - t0
    # headline contrasts: R7−R3 under MATCHED selection metric, on the metric it selected for.
    # Guard on key existence: base arms absent under --rich-only; all greedy arms absent under --skip-greedy.
    for sm, tgt in [("acc", "acc"), ("f1", "f1"), ("ece", "ece")]:
        if sm in select_metrics:
            for tag in ("base", "rich"):
                k7, k3 = f"{tag}_r7{sm}_{tgt}", f"{tag}_r3{sm}_{tgt}"
                if k7 in row and k3 in row:
                    row[f"{tag}_{sm}_delta"] = row[k7] - row[k3]
    return row


def main(datasets, seeds, cap, output_dir, select_metrics=("acc", "f1"),
         rich_only=False, pool_baselines=False, skip_greedy=False):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in datasets:
        for seed in seeds:
            try:
                r = run_one(ds, seed, cap, select_metrics, rich_only=rich_only,
                            pool_baselines=pool_baselines, skip_greedy=skip_greedy)
            except Exception as e:
                r = {"dataset": ds, "seed": seed, "err": repr(e)[:160]}
            rows.append(r); pd.DataFrame(rows).to_csv(out / "saga_richops_per_run.csv", index=False)
            if skip_greedy:
                msg = f"B-RAND f1={r.get('richB_rand_f1', np.nan):.3f} B-SC f1={r.get('richB_full_f1', np.nan):.3f}"
            else:
                msg = " | ".join(f"{sm}-sel Δ(R7-R3)={r.get(f'rich_{sm}_delta', np.nan):+.4f}" for sm in select_metrics)
            print(f"  {ds:14} s{seed}: {msg}", flush=True)
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if any(c.endswith(f"_{m}") for m in ("acc", "f1", "prec", "rec", "ece"))
                   or c.endswith("_delta")]
    agg = df.groupby("dataset")[metric_cols].mean().reset_index()
    agg.to_csv(out / "saga_richops_aggregated.csv", index=False)
    print("\n=== RICH pool — R7 vs R3, per selection objective (deployed=TabPFN) ===")
    # ECE: lower is better, so an R7 advantage is a NEGATIVE delta; flag degenerate (acc collapse).
    for sm in select_metrics:
        dcol = f"rich_{sm}_delta"
        if dcol not in agg:
            continue
        d = agg[dcol]
        print(f"  {sm}-selection: mean Δ(R7−R3) on {sm} = {d.mean():+.4f}  (per ds: "
              + ", ".join(f"{r.dataset}={r[dcol]:+.4f}" for _, r in agg.iterrows()) + ")")
        if sm == "ece":
            print("    accuracy of the ECE-selected pipelines (collapse check):")
            for _, r in agg.iterrows():
                print(f"      {r.dataset:16} r3ece_acc={r.get('rich_r3ece_acc',np.nan):.3f} "
                      f"r7ece_acc={r.get('rich_r7ece_acc',np.nan):.3f}  r7ece_ECE={r.get('rich_r7ece_ece',np.nan):.4f}")
    print("\nFull per-dataset table → saga_richops_aggregated.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["EEG", "Titanic", "AnimalShelter", "hepatitis", "ionosphere", "diabetes"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--max-pipelines", type=int, default=40)
    ap.add_argument("--select-metrics", nargs="+", default=["acc", "f1"], choices=["acc", "f1", "ece"])
    ap.add_argument("--rich-only", action="store_true", help="skip the base-pool arms (~half the cost)")
    ap.add_argument("--pool-baselines", action="store_true",
                    help="also compute rich-pool B-RAND (random pipeline) + B-SC (full pipeline)")
    ap.add_argument("--skip-greedy", action="store_true",
                    help="skip greedy arms entirely (use with --pool-baselines for a cheap B-RAND/B-SC-only run)")
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/paper_ready/saga_richops"))
    a = ap.parse_args()
    main(a.datasets, tuple(a.seeds), a.max_pipelines, a.output_dir, tuple(a.select_metrics),
         rich_only=a.rich_only, pool_baselines=a.pool_baselines, skip_greedy=a.skip_greedy)
