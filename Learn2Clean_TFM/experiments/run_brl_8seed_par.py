from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

DS_ORDER = [
    "hepatitis", "heart_statlog", "ionosphere", "blood_transfusion", "diabetes",
    "credit_g", "kr_vs_kp", "phoneme", "adult", "bank_marketing",
]


def _worker_init(threads: int):
    for _v in _THREAD_VARS:
        os.environ[_v] = str(threads)
    try:
        import torch
        torch.set_num_threads(threads)
    except Exception:
        pass
    try:
        import sklearn.ensemble as _ske
        _orig = _ske.RandomForestClassifier.__init__

        def _patched(self, *a, **k):
            k["n_jobs"] = threads
            _orig(self, *a, **k)

        _ske.RandomForestClassifier.__init__ = _patched
    except Exception:
        pass


def _run_cell(args):
    ds_name, seed, n_steps, n_tfm_steps = args
    import run_brl_baselines as BRL
    rows = BRL.run_one_cell(ds_name, seed, n_steps, n_tfm_steps)
    for r in rows:
        r["seed"] = seed
    return ds_name, seed, rows


def _aggregate(per_seed: pd.DataFrame, base: Path):
    def ci95(x):
        x = x.dropna().values
        return float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0

    def _std(x):
        x = x.dropna().values
        return float(x.std(ddof=1)) if len(x) > 1 else 0.0

    aggspec = dict(n_seeds=("seed", "nunique"))
    for m in ("tabpfn_acc", "ece", "f1", "prec", "rec"):
        if m in per_seed.columns:
            aggspec[f"{m}_mean"] = (m, "mean")
            aggspec[f"{m}_std"] = (m, _std)
            aggspec[f"{m}_ci95"] = (m, ci95)
    agg = per_seed.groupby(["dataset", "mode"]).agg(**aggspec).reset_index()
    agg.to_csv(base / "results_aggregated.csv", index=False)
    return agg


def main(seeds, datasets, n_steps, n_tfm_steps, output_dir, workers, threads):
    datasets = datasets or DS_ORDER
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    for _v in _THREAD_VARS:
        os.environ[_v] = str(threads)
    cells = [(ds, s, n_steps, n_tfm_steps) for s in seeds for ds in datasets]
    ncpu = os.cpu_count() or 0
    print(f"[par] {len(cells)} cells ({len(datasets)} datasets x {len(seeds)} seeds) "
          f"across {workers} workers x {threads} threads = {workers*threads} threads "
          f"(machine has {ncpu} logical CPUs)", flush=True)

    all_rows = []
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                             initargs=(threads,)) as ex:
        futs = {ex.submit(_run_cell, c): c for c in cells}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                _, _, rows = fut.result()
                all_rows.extend(rows)
            except Exception as exc:
                print(f"  CELL FAILED {c[0]} seed{c[1]}: {exc}", flush=True)
            done += 1
            pd.DataFrame(all_rows).to_csv(base / "results_per_seed.csv", index=False)
            elapsed = (time.time() - t0) / 60
            rate = elapsed / done
            eta = rate * (len(cells) - done)
            print(f"  [{done}/{len(cells)}] {c[0]} seed{c[1]} | "
                  f"{elapsed:.1f} min elapsed | ETA {eta:.0f} min", flush=True)

    if not all_rows:
        print("No results.")
        return
    per_seed = pd.DataFrame(all_rows)
    per_seed.to_csv(base / "results_per_seed.csv", index=False)
    agg = _aggregate(per_seed, base)
    print(f"\nDONE {(time.time()-t0)/3600:.2f}h -> {base}")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--n-steps", type=int, default=1500)
    ap.add_argument("--n-tfm-steps", type=int, default=100)
    ap.add_argument("--output-dir", default="outputs/paper_ready/brl_8seed_leakfree")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads-per-worker", type=int, default=4)
    a = ap.parse_args()
    main(a.seeds, a.datasets, a.n_steps, a.n_tfm_steps, a.output_dir,
         a.workers, a.threads_per_worker)
