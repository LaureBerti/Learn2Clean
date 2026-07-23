"""
experiments/wr_merge_analyze.py

Merge the per-shard pool_metrics.csv files (from the 5-VM weight-robustness fan-out)
and run the full sweep + Pareto + summary once over the combined 10-dataset pool.
Pure analysis — no model calls.

Usage
-----
  PYTHONPATH=src:experiments python experiments/wr_merge_analyze.py \
      --shards outputs/paper_ready/wr_vm_results/wr_shard_0 ... \
      --out outputs/paper_ready/weight_robustness_merged
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

import run_weight_robustness as W


def main(shard_glob: str, out: str, step: float) -> None:
    out_dir = Path(out); out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(shard_glob))
    if not files:
        print(f"No pool_metrics.csv matched {shard_glob}"); return
    pool = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    pool = pool.drop_duplicates(subset=["dataset", "seed", "pipe"], keep="first")
    pool.to_csv(out_dir / "pool_metrics.csv", index=False)
    print(f"Merged {len(files)} shards → {len(pool)} pool rows, "
          f"{pool['dataset'].nunique()} datasets, seeds {sorted(pool['seed'].unique())}")

    weights = W.simplex_grid(step)
    cfg = W.sweep(pool, weights); cfg.to_csv(out_dir / "wr_per_config.csv", index=False)
    par = W.pareto(pool, weights); par.to_csv(out_dir / "wr_pareto.csv", index=False)
    print("\n" + W.summarize(cfg, par, out_dir))
    print(f"\nSaved → {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-glob", default=None,
                    help="glob for shard pool_metrics.csv, e.g. '.../wr_shard_*/weight_robustness/pool_metrics.csv'")
    ap.add_argument("--out", default="outputs/paper_ready/weight_robustness_merged")
    ap.add_argument("--step", type=float, default=0.1)
    a = ap.parse_args()
    main(a.shard_glob, a.out, a.step)
