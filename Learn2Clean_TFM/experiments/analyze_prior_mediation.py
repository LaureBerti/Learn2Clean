"""
experiments/analyze_prior_mediation.py

Prior-alignment mediation diagnostic (paper Section 5.4, Table 4).

Question: is distance *to the TabPFN prior* — not distance *from the dirty data* — the
signal that co-varies with downstream macro-F1?

Given per-pipeline distances to an SCM approximation of the prior (M2_mmd, an RBF-MMD),
the implemented drift-from-dirty Wasserstein term (M3_marg_w1), and the downstream
macro-F1 of each cleaned pipeline, we compute WITHIN each (dataset, seed) group and
average the per-group coefficients (104 groups over the 13 held-out datasets x 8 seeds):

  * raw Spearman rho(distance-to-prior, F1)
  * partial rho(distance-to-prior, F1 | drift)      -> survives  => prior carries signal
  * partial rho(drift, F1 | distance-to-prior)      -> vanishes  => drift acts only via prior
  * rho(drift, distance-to-prior)                   -> +0.50     => cleaning overshoots the prior

Reproduces the Table 4 numbers:
  distance-to-prior (raw)        rho = -0.177   p = 2e-5
  distance-to-prior | drift      rho = -0.116   p = 0.019
  drift | distance-to-prior      rho = -0.063   p = 0.14 (n.s.)

Usage:
  PYTHONPATH=src python experiments/analyze_prior_mediation.py \
      --input outputs/paper_ready/prior_distance_MERGED_per_pipe.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, ttest_1samp

PRIOR = "M2_mmd"        # RBF-MMD distance to the SCM prior
DRIFT = "M3_marg_w1"    # implemented drift-from-dirty (marginal Wasserstein-1)
TARGET = "f1"           # downstream TabPFN macro-F1


def _partial_spearman(g: pd.DataFrame, x: str, y: str, z: str) -> float:
    """Partial Spearman rho(x, y | z) = partial Pearson on the rank-transformed columns."""
    R = {c: rankdata(g[c].values) for c in (x, y, z)}
    def resid(a, b):
        B = np.vstack([np.ones_like(b), b]).T
        beta, *_ = np.linalg.lstsq(B, a, rcond=None)
        return a - B @ beta
    rx, ry = resid(R[x], R[z]), resid(R[y], R[z])
    if rx.std() < 1e-9 or ry.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _within_group_mean(df: pd.DataFrame, fn) -> tuple[float, float, int]:
    """Average a per-(dataset,seed) coefficient; report a one-sample t-test vs 0."""
    vals = []
    for _, g in df.groupby(["dataset", "seed"]):
        if len(g) < 8:
            continue
        v = fn(g)
        if np.isfinite(v):
            vals.append(v)
    arr = np.asarray(vals)
    _, p = ttest_1samp(arr, 0.0)
    return float(arr.mean()), float(p), len(arr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/paper_ready/prior_distance_MERGED_per_pipe.csv")
    a = ap.parse_args()
    df = pd.read_csv(a.input)
    print(f"loaded {len(df)} pipelines over {df['dataset'].nunique()} datasets, "
          f"{df.groupby(['dataset','seed']).ngroups} (dataset,seed) groups\n")

    rows = [
        ("distance-to-prior (raw)",     lambda g: spearmanr(g[PRIOR], g[TARGET]).correlation),
        ("distance-to-prior | drift",   lambda g: _partial_spearman(g, PRIOR, TARGET, DRIFT)),
        ("drift | distance-to-prior",   lambda g: _partial_spearman(g, DRIFT, TARGET, PRIOR)),
        ("drift vs distance-to-prior",  lambda g: spearmanr(g[DRIFT], g[PRIOR]).correlation),
    ]
    print(f"{'correlation with macro-F1 (within-group mean)':<34} {'rho':>8} {'p':>10}  n")
    print("-" * 60)
    for name, fn in rows:
        rho, p, n = _within_group_mean(df, fn)
        note = " (n.s.)" if p >= 0.05 else ""
        print(f"{name:<34} {rho:>+8.3f} {p:>10.1e}  {n}{note}")
    print("\nReading: prior-distance predicts F1 even controlling for drift, whereas drift's "
          "association\nwith F1 vanishes once prior-distance is controlled -> distance-to-prior "
          "subsumes drift.")


if __name__ == "__main__":
    main()
