"""
experiments/run_saga_pyimpl.py

APPROXIMATE Python reimplementation of SAGA-style cleaning-pipeline search (cf. Siddiqi et al.), so we
can report a SAGA-style accuracy AND wall-clock time on OUR datasets / OUR held-out protocol split --- a proxy
for the real-engine SystemDS run (which failed on a cross-version signature mismatch). NOTE: this is
an approximation, not a faithful port --- real SAGA optimises PHYSICAL pipelines with a Multi-Armed
Bandit on top of evolutionary logical enumeration; here we use a single-stage genetic search and omit
the bandit stage.

What SAGA does, and what we mirror:
  * A primitive library organised by functional category (MVI, OTLR, SCALE, CI, DIM), each with a
    handful of operators/hyperparameters.  (We omit categories not in SAGA's library: power
    transforms, dedup, random projection.)
  * An EVOLUTIONARY top-k search: a population of pipelines is evaluated by a downstream model with
    cross-validation; the top-k elites are kept and the next generation is bred by crossover +
    mutation; repeat for max_iter generations.  Returns the best pipeline.
  * The evaluator is SAGA's own: multinomial logistic regression (`multiLogReg`) accuracy under
    k-fold CV.  We use sklearn LogisticRegression(multinomial) --- the faithful analogue.

Budget used in THIS experiment: topK=3, population=20, generations=5, cvk=3. (These are the values we
ran; they do NOT match the SystemDS `topk_cleaning` builtin defaults, which are topK=5, resource=20,
max_iter=10, cvk=2.)

HELD-OUT PROTOCOL: an outer 80/20 split holds out a sacred test that the SAGA search NEVER sees (the search
does CV strictly inside the outer-train).  The best pipeline is then deployed once on the sacred test.
Every per-fold / final cleaning transform is fit on its train rows only and applied to the held-out
rows --- identical contract to run_saga_richops / run_c2_tfm_reward_nested.

Reported per (dataset, seed):
  saga_clean_acc / _f1   : best SAGA pipeline, multiLogReg, on the sacred test
  saga_dirty_acc / _f1   : no-cleaning baseline (mean-impute only, to be runnable), same model/test
  saga_clean_acc_tabpfn  : the SAME SAGA-selected pipeline deployed on TabPFN (for our head-to-head)
  saga_sec               : wall-clock of the SAGA search (the time number)
  n_eval                 : distinct pipeline evaluations performed

Usage
-----
  PYTHONPATH=src:experiments python experiments/run_saga_pyimpl.py \
      --datasets EEG Titanic AnimalShelter hepatitis ionosphere diabetes \
      --seeds 42 1 2 3 4 5 6 7 [--with-tabpfn]
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
import run_saga_richops as R              # operator library + held-out protocol apply_pipeline
import run_c2_tfm_reward_nested as G      # split constants + (optional) TabPFN deploy

OUTER = G.OUTER_TEST_SIZE                 # sacred outer test fraction (0.2)
MCAR = R.MCAR                             # 0.15, same corruption as the rich-pool experiment

# --- SAGA-faithful primitive library (one operator per category; subset matching SAGA's) -----
SAGA_MVI   = ["mean", "median", "knn", "knn3", "knn10", "mice"]   # imputeByMean/Median/kNN/MICE
SAGA_OTLR  = [None, "iqr_rm", "z_rm", "winsor", "clipz"]          # outlierByIQR/BySd, winsorize
SAGA_SCALE = [None, "minmax", "zscore"]                           # scale
SAGA_CI    = [None, "smote"]                                      # class-imbalance (over-sampling)
SAGA_DIM   = [None, "pca"]                                        # pca
CATS = [SAGA_MVI, SAGA_OTLR, SAGA_SCALE, SAGA_CI, SAGA_DIM]

# SAGA `topk_cleaning` driver defaults --- kept identical for a faithful budget
POP, GENS, TOPK, CVK = 20, 5, 3, 3


def genome_to_pipe(g):
    """(mvi, otlr, scale, ci, dim) -> apply_pipeline's 7-tuple (imp,otl,trans,scl,dim,bal,ddp)."""
    mvi, otl, scl, ci, dim = g
    return (mvi, otl, None, scl, dim, ci, None)


DIRTY = ("mean", None, None, None, None)   # no-cleaning baseline (minimal impute to run the model)


def _logreg():
    return LogisticRegression(max_iter=1000, solver="lbfgs")


def cv_fitness(g, Xtr, ytr, seed):
    """SAGA's evaluator: multiLogReg accuracy under stratified k-fold CV, transforms fit per-fold
    (held-out protocol within the search). Returns mean fold accuracy, or -1 on failure."""
    pipe = genome_to_pipe(g)
    yv = ytr.values
    if len(np.unique(yv)) < 2:
        return -1.0
    folds = StratifiedKFold(n_splits=min(CVK, int(pd.Series(yv).value_counts().min())),
                            shuffle=True, random_state=seed) if pd.Series(yv).value_counts().min() >= 2 else None
    if folds is None:
        return -1.0
    accs = []
    for tr_i, va_i in folds.split(Xtr, yv):
        out = R.apply_pipeline(Xtr.iloc[tr_i].reset_index(drop=True), ytr.iloc[tr_i].reset_index(drop=True),
                               Xtr.iloc[va_i].reset_index(drop=True), pipe, seed)
        if out is None or len(out[0]) == 0:
            continue
        tr, ytr2, va = out
        if len(np.unique(ytr2.values)) < 2 or tr.shape[1] == 0:
            continue
        try:
            clf = _logreg().fit(tr.values, ytr2.values)
            accs.append(accuracy_score(ytr.iloc[va_i].values, clf.predict(va.values)))
        except Exception:
            continue
    return float(np.mean(accs)) if accs else -1.0


def rand_genome(rng):
    return tuple(cat[rng.integers(len(cat))] for cat in CATS)


def crossover(a, b, rng):
    return tuple(a[i] if rng.random() < 0.5 else b[i] for i in range(len(CATS)))


def mutate(g, rng):
    i = rng.integers(len(CATS)); g = list(g); g[i] = CATS[i][rng.integers(len(CATS[i]))]
    return tuple(g)


def saga_search(Xtr, ytr, seed):
    """Evolutionary top-k pipeline search (SAGA-faithful). Returns (best_genome, n_eval)."""
    rng = np.random.default_rng(seed)
    cache = {}

    def fit(g):
        if g not in cache:
            cache[g] = cv_fitness(g, Xtr, ytr, seed)
        return cache[g]

    pop = list({rand_genome(rng) for _ in range(POP)})
    while len(pop) < POP:
        pop.append(rand_genome(rng))
    for _ in range(GENS):
        pop = sorted(pop, key=fit, reverse=True)
        elites = pop[:TOPK]
        children = []
        while len(children) < POP - TOPK:
            a, b = elites[rng.integers(len(elites))], elites[rng.integers(len(elites))]
            c = mutate(crossover(a, b, rng), rng)
            children.append(c)
        pop = elites + children
    best = max(cache, key=cache.get)
    return best, len(cache)


def deploy_logreg(pipe, Xtr, ytr, Xte, yte, seed):
    out = R.apply_pipeline(Xtr, ytr, Xte, pipe, seed)
    if out is None or len(out[0]) == 0:
        return np.nan, np.nan
    tr, ytr2, te = out
    if len(np.unique(ytr2.values)) < 2 or tr.shape[1] == 0:
        return np.nan, np.nan
    try:
        clf = _logreg().fit(tr.values, ytr2.values)
        yp = clf.predict(te.values)
        return float(accuracy_score(yte.values, yp)), float(f1_score(yte.values, yp, average="macro"))
    except Exception:
        return np.nan, np.nan


def deploy_tabpfn(pipe, Xtr, ytr, Xte, yte, seed):
    out = R.apply_pipeline(Xtr, ytr, Xte, pipe, seed)
    if out is None or len(out[0]) == 0:
        return np.nan, np.nan
    tr, ytr2, te = out
    if len(np.unique(ytr2.values)) < 2 or tr.shape[1] == 0:
        return np.nan, np.nan
    try:
        yp, _ = G._tabpfn_fit_predict(tr.values, ytr2.values, te.values, seed)
        return float(accuracy_score(yte.values, yp)), float(f1_score(yte.values, yp, average="macro"))
    except Exception:
        return np.nan, np.nan


def run_one(name, seed, with_tabpfn):
    X, y = R.load_ds(name)
    if len(X) > G.SUBSAMPLE_CAP:
        X, _, y, _ = train_test_split(X, y, train_size=G.SUBSAMPLE_CAP, random_state=0,
                                      stratify=y if y.value_counts().min() >= 2 else None)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    Xd = R.mcar(X, MCAR, seed)
    Xtr, Xte, ytr, yte = train_test_split(Xd, y, test_size=OUTER, random_state=seed,
                                          stratify=y if y.value_counts().min() >= 2 else None)
    Xtr, ytr = Xtr.reset_index(drop=True), ytr.reset_index(drop=True)
    Xte, yte = Xte.reset_index(drop=True), yte.reset_index(drop=True)

    t0 = time.time()
    best, n_eval = saga_search(Xtr, ytr, seed)
    sec = time.time() - t0

    clean_acc, clean_f1 = deploy_logreg(genome_to_pipe(best), Xtr, ytr, Xte, yte, seed)
    dirty_acc, dirty_f1 = deploy_logreg(genome_to_pipe(DIRTY), Xtr, ytr, Xte, yte, seed)
    row = {"dataset": name, "seed": seed, "saga_clean_acc": clean_acc, "saga_clean_f1": clean_f1,
           "saga_dirty_acc": dirty_acc, "saga_dirty_f1": dirty_f1, "saga_sec": sec,
           "n_eval": n_eval, "pipe": str(best)}
    if with_tabpfn:
        row["saga_clean_acc_tabpfn"], row["saga_clean_f1_tabpfn"] = \
            deploy_tabpfn(genome_to_pipe(best), Xtr, ytr, Xte, yte, seed)
    return row


def main(datasets, seeds, with_tabpfn, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in datasets:
        for seed in seeds:
            try:
                r = run_one(ds, seed, with_tabpfn)
            except Exception as e:
                r = {"dataset": ds, "seed": seed, "err": repr(e)[:160]}
            rows.append(r); pd.DataFrame(rows).to_csv(out / "saga_pyimpl_per_run.csv", index=False)
            print(f"  {ds:14} s{seed}: clean={r.get('saga_clean_acc',np.nan):.3f} "
                  f"dirty={r.get('saga_dirty_acc',np.nan):.3f} "
                  f"F1={r.get('saga_clean_f1',np.nan):.3f} ({r.get('saga_sec',0):.1f}s, "
                  f"{r.get('n_eval',0)} evals)", flush=True)
    df = pd.DataFrame(rows)
    if "saga_clean_acc" not in df.columns:
        print("\n!!! ALL runs errored — no metrics produced. First errors:")
        for r in rows[:5]:
            print("   ", r.get("dataset"), r.get("seed"), "->", r.get("err"))
        return
    num = [c for c in df.columns if c not in ("dataset", "seed", "pipe", "err")]
    agg = df.groupby("dataset")[num].mean(numeric_only=True).reset_index()
    agg.to_csv(out / "saga_pyimpl_aggregated.csv", index=False)
    print("\n=== SAGA (Python reimplementation) --- multiLogReg, held-out protocol, by dataset ===")
    for _, r in agg.iterrows():
        line = (f"  {r.dataset:14} clean_acc={r.saga_clean_acc:.3f} dirty_acc={r.saga_dirty_acc:.3f} "
                f"(Δ={r.saga_clean_acc-r.saga_dirty_acc:+.3f})  F1={r.saga_clean_f1:.3f}  {r.saga_sec:.1f}s")
        if with_tabpfn and "saga_clean_acc_tabpfn" in agg.columns:
            line += f"  [TabPFN-deploy acc={r.saga_clean_acc_tabpfn:.3f}]"
        print(line)
    print(f"\n  mean clean−dirty (acc) = {(agg.saga_clean_acc-agg.saga_dirty_acc).mean():+.4f} | "
          f"mean search time = {agg.saga_sec.mean():.1f}s")
    print("Full table → saga_pyimpl_aggregated.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["EEG", "Titanic", "AnimalShelter", "hepatitis", "ionosphere", "diabetes"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--with-tabpfn", action="store_true", help="also deploy the SAGA-selected pipeline on TabPFN")
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/paper_ready/saga_pyimpl"))
    a = ap.parse_args()
    main(a.datasets, tuple(a.seeds), a.with_tabpfn, a.output_dir)
