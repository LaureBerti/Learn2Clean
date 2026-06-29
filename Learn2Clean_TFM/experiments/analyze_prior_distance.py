"""Merge the prior-distance shards and report (a) non-degeneracy + (b) Spearman correlation of
each estimator (M1-M4) and the ensembles (M2+M3, M1+M2+M3) with acc / macro-F1 / ECE, pooled
within (dataset,seed) over all 13 datasets x 8 seeds."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

PR = Path(__file__).parents[1] / "outputs/paper_ready"
MEAS = ["M1_nll", "M2_mmd", "M3_marg_w1", "M4_c2st"]


def zsum(g, cols):
    return sum((g[c] - g[c].mean()) / (g[c].std(ddof=0) + 1e-9) for c in cols)


def main():
    parts = [pd.read_csv(p) for p in PR.glob("prior_distance_*/prior_distance_per_pipe.csv")]
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(PR / "prior_distance_MERGED_per_pipe.csv", index=False)
    print(f"merged rows={len(df)}  datasets={df.dataset.nunique()}  groups={df.groupby(['dataset','seed']).ngroups}")

    print("\n=== (a) NON-DEGENERACY (mean CV across pipelines, over groups) ===")
    for m in MEAS:
        cv = df.groupby(["dataset", "seed"])[m].apply(lambda s: s.std() / (abs(s.mean()) + 1e-9)).mean()
        print(f"  {m:12}: CV={cv:.3f}  {'NON-degenerate' if cv > 0.05 else 'flat'}")

    def prho(fn, tgt):
        rs = []
        for _, g in df.groupby(["dataset", "seed"]):
            if len(g) < 6:
                continue
            s = fn(g)
            if np.std(s) > 1e-9 and g[tgt].nunique() > 2:
                rs.append(spearmanr(s, g[tgt]).correlation)
        return np.nanmean(rs) if rs else np.nan

    scores = [(m, (lambda g, m=m: g[m])) for m in MEAS]
    scores += [("M2+M3", lambda g: zsum(g, ["M2_mmd", "M3_marg_w1"])),
               ("M1+M2+M3", lambda g: zsum(g, ["M1_nll", "M2_mmd", "M3_marg_w1"]))]
    print("\n=== (b) CORRELATION  ρ(score, acc) / ρ(score, F1) / ρ(score, ECE) — pooled ===")
    out = []
    for name, fn in scores:
        ra, rf, re = prho(fn, "acc"), prho(fn, "f1"), prho(fn, "ece")
        print(f"  {name:12}: acc={ra:+.3f}  F1={rf:+.3f}  ECE={re:+.3f}")
        out.append({"score": name, "rho_acc": ra, "rho_f1": rf, "rho_ece": re})
    pd.DataFrame(out).to_csv(PR / "prior_distance_correlations.csv", index=False)
    print(f"\nsaved → prior_distance_correlations.csv")


if __name__ == "__main__":
    main()
