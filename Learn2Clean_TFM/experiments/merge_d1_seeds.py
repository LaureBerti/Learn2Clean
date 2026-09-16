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
