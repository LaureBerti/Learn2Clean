"""
experiments/run_c5_taskeval_par.py

Parallel driver for the C5 task-level experiment: the parameterized vs. discrete
action space, measured on the downstream TabPFN v2 test metric under the held-out
(nested) protocol.

Same computation as run_c5_taskeval.py — it reuses the nested harness
(run_c2_tfm_reward_nested.run_one) and the action-pool builders / k-aware test-prep
from run_c5_taskeval — but it evaluates the independent (dataset, seed, mode) tasks
concurrently. results_per_seed.csv is written after each completed task, and --resume
skips tasks already recorded there. Reported metrics come from the RF-reward-selected
pipeline (the C5 selection criterion).

Usage
-----
  PYTHONPATH=src:experiments python experiments/run_c5_taskeval_par.py \
      --seeds 42 1 2 3 4 5 6 7 --workers 8 \
      --output-dir outputs/paper_ready/c5_taskeval
"""
from __future__ import annotations

# Pin math libraries to a single thread per process before importing numpy/sklearn.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from scipy.stats import wilcoxon

import run_c2_tfm_reward_nested as C2
from run_c5_taskeval import build_discrete, build_param, make_prepare_test


def _cap_threads() -> None:
    """Keep every RandomForest and TabPFN/torch call single-threaded; parallelism is
    provided across tasks, not within a single estimator."""
    import sklearn.ensemble as _ske
    if not getattr(_ske.RandomForestClassifier, "_n_jobs_capped", False):
        _orig = _ske.RandomForestClassifier.__init__
        def _init(self, *a, **k):
            k["n_jobs"] = 1
            _orig(self, *a, **k)
        _ske.RandomForestClassifier.__init__ = _init
        _ske.RandomForestClassifier._n_jobs_capped = True
    try:
        import learn2clean_v3.rewards.multi_objective_reward as _mor
        for _m in getattr(_mor, "_MODELS", {}).values():
            if hasattr(_m, "n_jobs"):
                _m.n_jobs = 1
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


_cap_threads()

_SAMPLE = True
_MAX_PIPELINES = 300


def _run_task(task: Tuple[str, int, str]) -> Optional[Dict]:
    """Run one (dataset, seed, mode): set the mode's action pool on the C2 module
    globals, call run_one, and return the RF-reward-selected TabPFN test metrics."""
    ds, seed, mode = task
    _cap_threads()
    # C5 reports the RF-reward-selected path (rf_*) only. In select_best the RF path is
    # computed independently of the TFM-selection path, so stubbing the per-candidate
    # inner-validation TabPFN scorer leaves every rf_* metric identical while skipping
    # the unused TabPFN selection calls; only the ignored tfm_* columns change.
    C2.inner_val_tabpfn_acc = lambda *a, **k: 0.0
    actions, labels, groups = build_discrete() if mode == "discrete" else build_param()
    C2.ACTION_LABELS = labels
    C2.ACTION_GROUPS = groups
    C2.prepare_test_like_train = make_prepare_test(labels)
    all_pipelines = C2.enumerate_valid_pipelines(max_len=3)
    pipelines = (C2.sample_pipelines(all_pipelines, _MAX_PIPELINES, seed)
                 if _SAMPLE else all_pipelines)
    try:
        r = C2.run_one(ds, seed, pipelines, actions)
    except Exception as exc:  # a single failed cell must not sink the whole sweep
        return {"dataset": ds, "seed": seed, "mode": mode, "error": str(exc)[:200]}
    if r is None:
        return None
    return {"dataset": ds, "seed": seed, "mode": mode,
            "acc": r["rf_acc"], "ece": r["rf_ece"], "f1": r["rf_f1"],
            "pipeline": r["rf_pipeline"]}


def main() -> None:
    global _SAMPLE, _MAX_PIPELINES
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--sample", action="store_true", help="sample pipelines instead of exhaustive")
    ap.add_argument("--max-pipelines", type=int, default=300)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--resume", action="store_true", help="skip tasks already in results_per_seed.csv")
    a = ap.parse_args()

    _SAMPLE = a.sample
    _MAX_PIPELINES = a.max_pipelines
    ds_names = a.datasets or list(C2.BENCHMARK_DATASETS.keys())
    seeds = list(a.seeds)
    out_dir = Path(a.output_dir) if a.output_dir else (
        Path(__file__).parents[1] / "outputs" / "paper_ready" / "c5_taskeval")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed = out_dir / "results_per_seed.csv"

    rows: List[Dict] = []
    done = set()
    if a.resume and per_seed.exists():
        prev = pd.read_csv(per_seed)
        rows = prev.to_dict("records")
        done = {(r["dataset"], int(r["seed"]), r["mode"])
                for r in rows if pd.notna(r.get("acc", None))}
        print(f"[resume] {len(done)} cells already done", flush=True)

    tasks = [(ds, s, m) for m in ("discrete", "param") for ds in ds_names for s in seeds
             if (ds, s, m) not in done]
    print(f"[run] {len(tasks)} tasks | sample={_SAMPLE} max_pipelines={_MAX_PIPELINES} "
          f"| out={out_dir}", flush=True)

    t0 = time.time()
    n_ok = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_run_task, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            t = futs[fut]
            if res is None:
                print(f"  [SKIP] {t}", flush=True)
                continue
            rows.append(res)
            if "error" in res:
                print(f"  [ERR ] {t}: {res['error']}", flush=True)
            else:
                n_ok += 1
                print(f"  [{i}/{len(tasks)}] {res['dataset']:<18} seed{res['seed']} "
                      f"{res['mode']:<8} acc={res['acc']:.4f} ({(time.time()-t0):.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(per_seed, index=False)  # checkpoint after every task

    df = pd.DataFrame(rows)
    df.to_csv(per_seed, index=False)
    if df.empty or "acc" not in df.columns:
        print("No usable results."); return

    good = df[df["acc"].notna()]
    piv = good.groupby(["dataset", "mode"]).acc.mean().unstack("mode")
    if {"param", "discrete"}.issubset(piv.columns):
        piv["delta"] = piv["param"] - piv["discrete"]
    piv.to_csv(out_dir / "results_aggregated.csv")
    print("\n=== C5 TASK-LEVEL (TabPFN v2 test accuracy, RF-reward selection) ===")
    print(piv.round(4).to_string())
    if "delta" in piv.columns:
        d = piv["delta"].dropna()
        print(f"\nmean delta(param - discrete) TabPFN acc = {d.mean():+.4f} over {len(d)} datasets; "
              f"param wins {int((d > 0).sum())}/{len(d)}")
        if len(d) >= 3:
            try:
                stat, p = wilcoxon(piv["param"].dropna().values, piv["discrete"].dropna().values)
                print(f"paired Wilcoxon (param vs discrete test acc): stat={stat:.3f} p={p:.4f}")
            except Exception as e:
                print(f"Wilcoxon failed: {e}")
    print(f"\nok={n_ok} | wall={(time.time()-t0)/60:.1f} min | saved -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
