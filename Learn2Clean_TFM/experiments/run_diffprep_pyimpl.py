"""
experiments/run_diffprep_pyimpl.py

FAITHFUL Python reimplementation of DiffPrep (Li, Chen, Chu, Rong; PACMMOD 2023, "DiffPrep:
Differentiable Data Preprocessing Pipeline Search for Learning over Tabular Data") --- the
DiffPrep-Fix variant (fixed transformation-step order). Lets us report a DiffPrep accuracy AND
wall-clock on OUR datasets / OUR held-out protocol split, complementing its published numbers.

Method (what we mirror):
  * A fixed-order pipeline of preprocessing STEPS (imputation -> outlier -> normalization). Each step
    has a few candidate operators.
  * CONTINUOUS RELAXATION: each step's output is a softmax-weighted mixture of its candidates'
    outputs, x_out = sum_c softmax(alpha_step)_c * op_c(x_in). All operators are differentiable in x
    (constant-fill imputation, clamp-based outlier capping, affine normalization with train-fit
    stats), so gradients flow to the architecture parameters alpha.
  * BI-LEVEL optimisation (first-order DARTS): the downstream model weights w are updated on the
    TRAIN split; the architecture params alpha are updated on the VALIDATION split; alternated.
  * Downstream model: logistic regression (a single linear layer + softmax CE) --- DiffPrep's
    primary learner.
  * After search, alpha is discretised (argmax per step); we retrain a fresh LR on the discovered
    discrete pipeline and report test accuracy (discover-then-retrain, as in the paper).

HELD-OUT PROTOCOL: outer 80/20 sacred test never seen by the search; transform stats are fit on train rows
only; alpha is tuned on an inner validation split; the discrete pipeline is finally fit on the full
outer-train and evaluated once on the sacred test.

Reported per (dataset, seed):
  diffprep_clean_acc/_f1     : discovered discrete pipeline, LR, on the sacred test
  diffprep_dirty_acc/_f1     : no-preprocessing baseline (mean-impute only), LR, same test
  diffprep_clean_acc_tabpfn  : the SAME discovered pipeline deployed on TabPFN (head-to-head)
  diffprep_sec               : wall-clock of the differentiable search
  arch                       : the discovered (impute, outlier, normalize) operators

Usage
-----
  PYTHONPATH=src:experiments python experiments/run_diffprep_pyimpl.py \
      --datasets EEG Titanic AnimalShelter hepatitis ionosphere diabetes blood_transfusion credit_g \
      --seeds 42 1 2 3 4 5 6 7 [--with-tabpfn]
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
# R (load_ds/mcar) and G (TabPFN deploy + caps) are imported LAZILY inside run_one(): both pull in
# TabPFN at import time, which we don't want to require to import/test the differentiable core.

OUTER = 0.2                           # sacred outer-test fraction (matches G.OUTER_TEST_SIZE)
MCAR = 0.15                           # corruption rate (matches R.MCAR)
SUBSAMPLE_CAP = 3000                  # fallback; overridden by G.SUBSAMPLE_CAP at runtime
INNER = 0.25                          # inner train/val split for the bi-level search
EPOCHS, LR_W, LR_A = 150, 5e-2, 5e-2

IMPUTE_CANDS   = ["mean", "median", "zero"]               # missing-value imputation
OUTLIER_CANDS  = ["none", "sd3", "iqr"]                   # identity / clip 3-sigma / clip 1.5-IQR
NORMAL_CANDS   = ["none", "standard", "minmax", "robust"] # identity / z-score / min-max / robust
STEPS = [("impute", IMPUTE_CANDS), ("outlier", OUTLIER_CANDS), ("normalize", NORMAL_CANDS)]


# ---------------------------------------------------------------------------------------------
# Train-fit statistics (computed on train rows only; reused for val/test => held-out protocol)
# ---------------------------------------------------------------------------------------------
def fit_stats(Xtr_np):
    col_mean = np.nanmean(Xtr_np, axis=0); col_mean = np.nan_to_num(col_mean, nan=0.0)
    col_med  = np.nanmedian(Xtr_np, axis=0); col_med = np.nan_to_num(col_med, nan=0.0)
    # outlier/normalize stats computed on mean-imputed train (a stable reference grid)
    Xi = np.where(np.isnan(Xtr_np), col_mean, Xtr_np)
    mu, sd = Xi.mean(0), Xi.std(0) + 1e-8
    mn, mx = Xi.min(0), Xi.max(0)
    q1, q3 = np.percentile(Xi, 25, axis=0), np.percentile(Xi, 75, axis=0); iqr = (q3 - q1) + 1e-8
    return dict(mean=col_mean, median=col_med, mu=mu, sd=sd, mn=mn, mx=mx, q1=q1, q3=q3, iqr=iqr)


def T(s, dev):  # numpy stat -> tensor
    return torch.tensor(s, dtype=torch.float32, device=dev)


# ---------------------------------------------------------------------------------------------
# Differentiable soft pipeline (gradients flow to alpha)
# ---------------------------------------------------------------------------------------------
def step_candidates(name, x, st, dev):
    """Return list of candidate-operator outputs for a step, given current tensor x.
    `impute` additionally consumes the missingness mask carried on x (handled by caller)."""
    if name == "outlier":
        mu, sd = T(st["mu"], dev), T(st["sd"], dev)
        q1, q3, iqr = T(st["q1"], dev), T(st["q3"], dev), T(st["iqr"], dev)
        return [x,
                torch.clamp(x, mu - 3 * sd, mu + 3 * sd),                 # 3-sigma cap
                torch.clamp(x, q1 - 1.5 * iqr, q3 + 1.5 * iqr)]           # 1.5-IQR cap
    if name == "normalize":
        return [x,
                (x - T(st["mu"], dev)) / T(st["sd"], dev),
                (x - T(st["mn"], dev)) / (T(st["mx"], dev) - T(st["mn"], dev) + 1e-8),
                (x - T(st["median"], dev)) / T(st["iqr"], dev)]
    raise ValueError(name)


class SoftPipeline(nn.Module):
    def __init__(self, d, n_classes, st, dev):
        super().__init__()
        self.st, self.dev = st, dev
        self.alpha = nn.ParameterList([nn.Parameter(torch.zeros(len(c))) for _, c in STEPS])
        self.clf = nn.Linear(d, n_classes)

    def forward(self, x0, mask):
        # step 0: imputation (mixture over fill vectors at the missing entries)
        fills = [T(self.st["mean"], self.dev), T(self.st["median"], self.dev),
                 torch.zeros_like(T(self.st["mean"], self.dev))]
        w_imp = F.softmax(self.alpha[0], dim=0)
        fill = sum(w_imp[c] * fills[c] for c in range(len(fills)))      # (d,)
        x = x0 * (1 - mask) + fill * mask                              # observed kept, missing mixed
        # steps 1..: outlier, normalize
        for si in range(1, len(STEPS)):
            name = STEPS[si][0]
            cands = step_candidates(name, x, self.st, self.dev)
            w = F.softmax(self.alpha[si], dim=0)
            x = sum(w[c] * cands[c] for c in range(len(cands)))
        return self.clf(x)

    def arch(self):
        return tuple(STEPS[i][1][int(torch.argmax(a).item())] for i, a in enumerate(self.alpha))


def to_tensors(X_np, dev):
    mask = np.isnan(X_np).astype(np.float32)
    x0 = np.nan_to_num(X_np, nan=0.0).astype(np.float32)
    return torch.tensor(x0, device=dev), torch.tensor(mask, device=dev)


def diffprep_search(Xtr_np, ytr_idx, Xva_np, yva_idx, st, n_classes, seed):
    dev = "cpu"
    torch.manual_seed(seed)
    model = SoftPipeline(Xtr_np.shape[1], n_classes, st, dev).to(dev)
    x0_tr, m_tr = to_tensors(Xtr_np, dev); x0_va, m_va = to_tensors(Xva_np, dev)
    yt = torch.tensor(ytr_idx, device=dev); yv = torch.tensor(yva_idx, device=dev)
    w_opt = torch.optim.Adam(model.clf.parameters(), lr=LR_W, weight_decay=1e-4)
    a_opt = torch.optim.Adam(model.alpha.parameters(), lr=LR_A)
    for _ in range(EPOCHS):
        # (1) update model weights w on TRAIN
        model.train(); w_opt.zero_grad()
        loss_tr = F.cross_entropy(model(x0_tr, m_tr), yt)
        loss_tr.backward(); w_opt.step()
        # (2) update architecture alpha on VALIDATION (first-order DARTS)
        a_opt.zero_grad()
        loss_va = F.cross_entropy(model(x0_va, m_va), yv)
        loss_va.backward(); a_opt.step()
    return model.arch()


# ---------------------------------------------------------------------------------------------
# Discrete pipeline (numpy) for the final discover-then-retrain deployment
# ---------------------------------------------------------------------------------------------
def apply_discrete(arch, Xtr_np, Xte_np):
    st = fit_stats(Xtr_np)
    imp, otl, nrm = arch

    def impute(Z):
        fill = {"mean": st["mean"], "median": st["median"], "zero": np.zeros_like(st["mean"])}[imp]
        return np.where(np.isnan(Z), fill, Z)

    def outlier(Z):
        if otl == "sd3":
            return np.clip(Z, st["mu"] - 3 * st["sd"], st["mu"] + 3 * st["sd"])
        if otl == "iqr":
            return np.clip(Z, st["q1"] - 1.5 * st["iqr"], st["q3"] + 1.5 * st["iqr"])
        return Z

    def normalize(Z):
        if nrm == "standard":
            return (Z - st["mu"]) / st["sd"]
        if nrm == "minmax":
            return (Z - st["mn"]) / (st["mx"] - st["mn"] + 1e-8)
        if nrm == "robust":
            return (Z - st["median"]) / st["iqr"]
        return Z

    f = lambda Z: normalize(outlier(impute(Z)))
    return f(Xtr_np), f(Xte_np)


def deploy_lr(arch, Xtr_np, ytr, Xte_np, yte):
    tr, te = apply_discrete(arch, Xtr_np, Xte_np)
    try:
        clf = LogisticRegression(max_iter=1000).fit(tr, ytr)
        yp = clf.predict(te)
        return float(accuracy_score(yte, yp)), float(f1_score(yte, yp, average="macro"))
    except Exception:
        return np.nan, np.nan


def deploy_tabpfn(arch, Xtr_np, ytr, Xte_np, yte, seed):
    import run_c2_tfm_reward_nested as G   # lazy: pulls TabPFN at import
    tr, te = apply_discrete(arch, Xtr_np, Xte_np)
    try:
        yp, _ = G._tabpfn_fit_predict(tr, ytr, te, seed)
        return float(accuracy_score(yte, yp)), float(f1_score(yte, yp, average="macro"))
    except Exception:
        return np.nan, np.nan


def run_one(name, seed, with_tabpfn):
    import run_saga_richops as R          # lazy: pulls TabPFN at import
    cap = getattr(__import__("run_c2_tfm_reward_nested"), "SUBSAMPLE_CAP", SUBSAMPLE_CAP)
    X, y = R.load_ds(name)
    if len(X) > cap:
        X, _, y, _ = train_test_split(X, y, train_size=cap, random_state=0,
                                      stratify=y if y.value_counts().min() >= 2 else None)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    Xnum = X.select_dtypes(include="number")
    if Xnum.shape[1] == 0:
        return {"dataset": name, "seed": seed, "err": "no numeric columns"}
    Xd = R.mcar(Xnum, MCAR, seed).values.astype(np.float32)
    classes = sorted(pd.unique(y)); cidx = {c: i for i, c in enumerate(classes)}
    yi = np.array([cidx[v] for v in y]); n_classes = len(classes)

    Xtr, Xte, ytr, yte = train_test_split(Xd, yi, test_size=OUTER, random_state=seed,
                                          stratify=yi if pd.Series(yi).value_counts().min() >= 2 else None)
    strat = ytr if pd.Series(ytr).value_counts().min() >= 2 else None
    Xt, Xv, yt, yv = train_test_split(Xtr, ytr, test_size=INNER, random_state=seed, stratify=strat)

    st = fit_stats(Xt)
    t0 = time.time()
    arch = diffprep_search(Xt, yt, Xv, yv, st, n_classes, seed)
    sec = time.time() - t0

    clean_acc, clean_f1 = deploy_lr(arch, Xtr, ytr, Xte, yte)
    dirty_acc, dirty_f1 = deploy_lr(("mean", "none", "none"), Xtr, ytr, Xte, yte)
    row = {"dataset": name, "seed": seed, "diffprep_clean_acc": clean_acc, "diffprep_clean_f1": clean_f1,
           "diffprep_dirty_acc": dirty_acc, "diffprep_dirty_f1": dirty_f1, "diffprep_sec": sec,
           "arch": str(arch)}
    if with_tabpfn:
        row["diffprep_clean_acc_tabpfn"], row["diffprep_clean_f1_tabpfn"] = \
            deploy_tabpfn(arch, Xtr, ytr, Xte, yte, seed)
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
            rows.append(r); pd.DataFrame(rows).to_csv(out / "diffprep_pyimpl_per_run.csv", index=False)
            print(f"  {ds:14} s{seed}: clean={r.get('diffprep_clean_acc',np.nan):.3f} "
                  f"dirty={r.get('diffprep_dirty_acc',np.nan):.3f} F1={r.get('diffprep_clean_f1',np.nan):.3f} "
                  f"arch={r.get('arch','')} ({r.get('diffprep_sec',0):.1f}s)", flush=True)
    df = pd.DataFrame(rows)
    if "diffprep_clean_acc" not in df.columns:
        print("\n!!! ALL runs errored — no metrics produced. First errors:")
        for r in rows[:5]:
            print("   ", r.get("dataset"), r.get("seed"), "->", r.get("err"))
        return
    num = [c for c in df.columns if c not in ("dataset", "seed", "arch", "err")]
    agg = df.groupby("dataset")[num].mean(numeric_only=True).reset_index()
    agg.to_csv(out / "diffprep_pyimpl_aggregated.csv", index=False)
    print("\n=== DiffPrep (Python reimplementation, DiffPrep-Fix) --- LR, held-out protocol, by dataset ===")
    for _, r in agg.iterrows():
        line = (f"  {r.dataset:14} clean_acc={r.diffprep_clean_acc:.3f} dirty_acc={r.diffprep_dirty_acc:.3f} "
                f"(Δ={r.diffprep_clean_acc-r.diffprep_dirty_acc:+.3f})  F1={r.diffprep_clean_f1:.3f}  {r.diffprep_sec:.1f}s")
        if with_tabpfn and "diffprep_clean_acc_tabpfn" in agg.columns:
            line += f"  [TabPFN-deploy acc={r.diffprep_clean_acc_tabpfn:.3f}]"
        print(line)
    print(f"\n  mean clean−dirty (acc) = {(agg.diffprep_clean_acc-agg.diffprep_dirty_acc).mean():+.4f} | "
          f"mean search time = {agg.diffprep_sec.mean():.1f}s")
    print("Full table → diffprep_pyimpl_aggregated.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["EEG", "Titanic", "AnimalShelter", "hepatitis", "ionosphere", "diabetes"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--with-tabpfn", action="store_true")
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/paper_ready/diffprep_pyimpl"))
    a = ap.parse_args()
    main(a.datasets, tuple(a.seeds), a.with_tabpfn, a.output_dir)
