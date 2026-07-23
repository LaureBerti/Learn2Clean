"""
experiments/merge_d1_seeds.py

Merge the original 5-seed D1 results with the resumed 3-seed results into the full
8-seed held-out protocol D1, re-aggregate (mean ± 95% CI per dataset), and re-run the paired
Wilcoxon (TFM>RF) over the per-dataset means. Resume logic — no recomputation of the
first 5 seeds.

Usage
-----
  PYTHONPATH=src:experiments python experiments/merge_d1_seeds.py \
      --orig  outputs/paper_ready/d1_vm_results/c2_tfm_reward_nested/results_per_seed.csv \
      --resume outputs/paper_ready/c2_nested_seeds567/results_per_seed.csv \
      --out   outputs/paper_ready/d1_8seed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import run_c2_tfm_reward_nested as G


def main(orig: str, resume: str, out: str) -> None:
    out_dir = Path(out); out_dir.mkdir(parents=True, exist_ok=True)
    df_o = pd.read_csv(orig)
    df_r = pd.read_csv(resume)
    df = pd.concat([df_o, df_r], ignore_index=True)
    # Resume logic: drop any accidental duplicate (dataset, seed) keeping the first.
    df = df.drop_duplicates(subset=["dataset", "seed"], keep="first").reset_index(drop=True)
    df.to_csv(out_dir / "results_per_seed.csv", index=False)

    n_seeds = df.groupby("dataset")["seed"].nunique()
    print(f"Merged: {len(df)} rows | seeds/dataset: {sorted(df['seed'].unique())} "
          f"({int(n_seeds.min())}–{int(n_seeds.max())} per dataset)")

    agg = G.aggregate(df)
    agg.to_csv(out_dir / "results_aggregated.csv", index=False)

    pivot = agg.set_index("dataset")
    rf, tfm = pivot["rf_acc_mean"].dropna(), pivot["tfm_acc_mean"].dropna()
    shared = rf.index.intersection(tfm.index)
    print(f"\n8-seed held-out protocol D1 — {len(shared)} datasets")
    if len(shared) >= 2:
        stat, p = wilcoxon(tfm[shared].values, rf[shared].values, alternative="greater")
        wins = int((tfm[shared].values > rf[shared].values).sum())
        print(f"Wilcoxon TFM>RF: stat={stat:.3f} p={p:.4f}")
        print(f"TFM wins {wins}/{len(shared)} | TFM mean={tfm[shared].mean():.4f} "
              f"RF={rf[shared].mean():.4f} Δ={tfm[shared].mean()-rf[shared].mean():+.4f}")
    print(f"Saved → {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--resume", required=True)
    ap.add_argument("--out", default="outputs/paper_ready/d1_8seed")
    a = ap.parse_args()
    main(a.orig, a.resume, a.out)
