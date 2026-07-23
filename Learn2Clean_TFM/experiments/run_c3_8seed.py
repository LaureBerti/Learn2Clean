"""
experiments/run_c3_8seed.py

8-seed driver for C3 (calibration recovery) -> mean +/- 95% CI ECE bands for
Figure 4 (R3-W4 variance). Thin wrapper around run_c3_calibration.main: runs the
four-error-type ECE sweep once per seed and merges into results_per_seed.csv.

C3 is a deterministic greedy-oracle + TabPFN sweep (no RL), so it is cheap.

Usage:
  PYTHONPATH=src:experiments python experiments/run_c3_8seed.py \
      --seeds 42 1 2 3 4 5 6 7 --output-dir outputs/paper_ready/c3_8seed
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import pandas as pd
import run_c3_calibration as C3


def main(seeds, output_dir) -> None:
    base = Path(output_dir); base.mkdir(parents=True, exist_ok=True)
    frames = []
    t0 = time.time()
    for seed in seeds:
        sd = base / f"seed_{seed}"; sd.mkdir(parents=True, exist_ok=True)
        print(f"\n########## C3 seed {seed} ##########", flush=True)
        C3.main(output_dir=str(sd), seed=seed)
        csv = sd / "c3_error_type.csv"
        if csv.exists():
            df = pd.read_csv(csv); df["seed"] = seed; frames.append(df)
            pd.concat(frames, ignore_index=True).to_csv(base / "results_per_seed.csv", index=False)
        print(f"  [seed {seed} done] {(time.time()-t0)/60:.1f} min", flush=True)
    if not frames:
        print("No results."); return
    per = pd.concat(frames, ignore_index=True)
    per.to_csv(base / "results_per_seed.csv", index=False)
    print(f"\nDONE {(time.time()-t0)/60:.1f} min -> {base} ({len(per)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--output-dir", default="outputs/paper_ready/c3_8seed")
    a = ap.parse_args()
    main(a.seeds, a.output_dir)
