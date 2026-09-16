from __future__ import annotations
import argparse, time
from pathlib import Path
import pandas as pd
import run_c4_error_sensitivity as C4


def main(seeds, max_pipelines, output_dir) -> None:
    base = Path(output_dir); base.mkdir(parents=True, exist_ok=True)
    frames = []
    t0 = time.time()
    for seed in seeds:
        sd = base / f"seed_{seed}"; sd.mkdir(parents=True, exist_ok=True)
        print(f"\n########## C4 seed {seed} ##########", flush=True)
        C4.main(output_dir=str(sd), seed=seed, max_pipelines=max_pipelines)
        df = pd.read_csv(sd / "results.csv"); df["seed"] = seed
        frames.append(df)
        pd.concat(frames, ignore_index=True).to_csv(base / "results_per_seed.csv", index=False)
        print(f"  [seed {seed} done] {(time.time()-t0)/60:.1f} min", flush=True)
    per = pd.concat(frames, ignore_index=True)
    per.to_csv(base / "results_per_seed.csv", index=False)
    print(f"\nDONE {(time.time()-t0)/60:.1f} min -> {base} ({len(per)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--max-pipelines", type=int, default=20)
    ap.add_argument("--output-dir", default="outputs/paper_ready/c4_8seed")
    a = ap.parse_args()
    main(a.seeds, a.max_pipelines, a.output_dir)
