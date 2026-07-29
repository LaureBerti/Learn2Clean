"""
experiments/investigate_tabpfn_tuning.py

Robustness check: is the held-out protocol negative result
(TFM-reward ≈ RF-reward) an artifact of an under-powered TabPFN configuration?

D1 used TabPFN defaults: n_estimators=8, softmax_temperature=0.9,
balance_probabilities=False, context cap=1024. This driver re-runs the EXACT
held-out protocol nested comparison under several TabPFN configurations and reports, per
config: TFM mean acc, RF mean acc, their gap, and how many datasets TFM wins.

If a stronger TabPFN (larger ensemble, larger ICL context, probability balancing)
opens a TFM>RF gap, then tuning matters and the null is premature. If the gap stays
≈0 across configs, the negative result is robust to TabPFN tuning.

Usage
-----
  PYTHONPATH=src python experiments/investigate_tabpfn_tuning.py \
      --datasets hepatitis ionosphere diabetes blood_transfusion --seeds 42 1 2
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import run_c2_tfm_reward_nested as G

# Each config is a set of env overrides read by G._tabpfn_cfg() at call time.
CONFIGS: Dict[str, Dict[str, str]] = {
    "default_e8_c1024":      {"TABPFN_N_ESTIMATORS": "8",  "TABPFN_CTX_CAP": "1024", "TABPFN_BALANCE": "0"},
    "ens32_c1024":           {"TABPFN_N_ESTIMATORS": "32", "TABPFN_CTX_CAP": "1024", "TABPFN_BALANCE": "0"},
    "ens8_ctx4096":          {"TABPFN_N_ESTIMATORS": "8",  "TABPFN_CTX_CAP": "4096", "TABPFN_BALANCE": "0"},
    "ens32_ctx4096_balanced":{"TABPFN_N_ESTIMATORS": "32", "TABPFN_CTX_CAP": "4096", "TABPFN_BALANCE": "1"},
}


def _set_env(cfg: Dict[str, str]) -> None:
    for k, v in cfg.items():
        os.environ[k] = v


def main(dataset_names: List[str], seeds, max_pipelines: int, output_dir=None) -> None:
    out_dir = Path(output_dir) if output_dir else (
        Path(__file__).parents[1] / "outputs" / "paper_ready" / "tabpfn_tuning_investigation")
    out_dir.mkdir(parents=True, exist_ok=True)

    actions = G.build_actions()
    all_pipelines = G.enumerate_valid_pipelines(max_len=3)
    rows: List[Dict] = []
    summary: List[Dict] = []
    t0 = time.time()

    for cfg_name, cfg in CONFIGS.items():
        _set_env(cfg)
        print(f"\n{'='*64}\nCONFIG: {cfg_name}  {cfg}", flush=True)
        cfg_rows: List[Dict] = []
        for ds in dataset_names:
            for seed in seeds:
                pipelines = G.sample_pipelines(all_pipelines, max_pipelines, seed)
                ts = time.time()
                r = G.run_one(ds, seed, pipelines, actions)
                if r is None:
                    continue
                r["config"] = cfg_name
                rows.append(r); cfg_rows.append(r)
                print(f"  {ds:16} s{seed}: rf={r['rf_acc']:.4f} tfm={r['tfm_acc']:.4f} "
                      f"Δ={r['tfm_acc']-r['rf_acc']:+.4f} ({time.time()-ts:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(out_dir / "tuning_per_run.csv", index=False)

        df = pd.DataFrame(cfg_rows)
        if len(df):
            rf, tfm = df["rf_acc"].dropna(), df["tfm_acc"].dropna()
            n = min(len(rf), len(tfm))
            delta = float(tfm.mean() - rf.mean())
            wins = int((df["tfm_acc"] > df["rf_acc"]).sum())
            losses = int((df["tfm_acc"] < df["rf_acc"]).sum())
            summary.append({"config": cfg_name, "n_runs": n,
                            "rf_mean_acc": round(float(rf.mean()), 4),
                            "tfm_mean_acc": round(float(tfm.mean()), 4),
                            "delta_tfm_minus_rf": round(delta, 4),
                            "tfm_wins": wins, "tfm_losses": losses})
            print(f"  → {cfg_name}: RF={rf.mean():.4f} TFM={tfm.mean():.4f} "
                  f"Δ={delta:+.4f} wins={wins} losses={losses}", flush=True)

    sdf = pd.DataFrame(summary)
    sdf.to_csv(out_dir / "tuning_summary.csv", index=False)
    print(f"\n{'='*64}\nSUMMARY (does TabPFN tuning open a TFM>RF gap?)")
    print(sdf.to_string(index=False))
    print(f"\nWall-clock: {(time.time()-t0)/60:.1f} min → {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["hepatitis", "ionosphere", "diabetes", "blood_transfusion"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--max-pipelines", type=int, default=12)
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    main(a.datasets, tuple(a.seeds), a.max_pipelines, a.output_dir)
