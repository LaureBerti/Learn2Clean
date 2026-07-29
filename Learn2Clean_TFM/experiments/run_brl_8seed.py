"""
experiments/run_brl_8seed.py

8-seed driver for the B-RL baselines (B-RL-RF, B-RL-TFM) — variance for Table 2
Thin wrapper around run_brl_baselines.main: reassigns the
module OUT_DIR per seed, runs all datasets, then merges into a single per-seed CSV
plus a mean ± 95% CI aggregate per (dataset, mode).

This is the EXPENSIVE table row: PPO training (RF reward) + TFM fine-tune per
dataset per seed. Designed to run unattended.

Usage
-----
  PYTHONPATH=src:experiments python experiments/run_brl_8seed.py \
      --seeds 42 1 2 3 4 5 6 7 --output-dir outputs/paper_ready/brl_8seed
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_brl_baselines as BRL


def main(seeds, datasets, n_steps, n_tfm_steps, output_dir) -> None:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    all_frames = []
    t0 = time.time()
    for seed in seeds:
        seed_dir = base / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        BRL.OUT_DIR = seed_dir                      # redirect the harness output
        print(f"\n########## SEED {seed} ##########", flush=True)
        BRL.main(dataset_names=datasets, n_steps=n_steps,
                 n_tfm_steps=n_tfm_steps, seed=seed)
        csv = seed_dir / "results.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df["seed"] = seed
            all_frames.append(df)
            # persist the running merge after every seed (crash-safe)
            pd.concat(all_frames, ignore_index=True).to_csv(
                base / "results_per_seed.csv", index=False)
        print(f"  [seed {seed} done] cumulative {(time.time()-t0)/3600:.2f}h", flush=True)

    if not all_frames:
        print("No results."); return
    per_seed = pd.concat(all_frames, ignore_index=True)
    per_seed.to_csv(base / "results_per_seed.csv", index=False)

    def ci95(x):
        x = x.dropna().values
        if len(x) < 2:
            return 0.0
        return 1.96 * x.std(ddof=1) / np.sqrt(len(x))

    def _std(x):
        x = x.dropna().values
        return float(x.std(ddof=1)) if len(x) > 1 else 0.0

    aggspec = dict(
        acc_mean=("tabpfn_acc", "mean"), acc_std=("tabpfn_acc", _std), acc_ci95=("tabpfn_acc", ci95),
        ece_mean=("ece", "mean"), ece_std=("ece", _std), ece_ci95=("ece", ci95),
        n_seeds=("seed", "nunique"),
    )
    # F1/prec/rec were added to the harness; guard in case an old CSV lacks them.
    for m in ("f1", "prec", "rec"):
        if m in per_seed.columns:
            aggspec[f"{m}_mean"] = (m, "mean")
            aggspec[f"{m}_std"] = (m, _std)
            aggspec[f"{m}_ci95"] = (m, ci95)
    agg = per_seed.groupby(["dataset", "mode"]).agg(**aggspec).reset_index()
    agg.to_csv(base / "results_aggregated.csv", index=False)
    print(f"\nDONE {(time.time()-t0)/3600:.2f}h → {base}")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--n-steps", type=int, default=5_000)
    ap.add_argument("--n-tfm-steps", type=int, default=1_000)
    ap.add_argument("--output-dir",
                    default="outputs/paper_ready/brl_8seed")
    a = ap.parse_args()
    main(a.seeds, a.datasets, a.n_steps, a.n_tfm_steps, a.output_dir)
