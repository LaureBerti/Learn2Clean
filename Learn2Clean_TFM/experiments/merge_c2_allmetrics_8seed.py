"""
experiments/merge_c2_allmetrics_8seed.py

Merge the two parallel C2 all-metrics runs into one 8-seed table:
  * c2_allmetrics_4seed        -> seeds 42, 1, 2, 3
  * c2_allmetrics_seeds4567    -> seeds 4, 5, 6, 7

Both were produced by the SAME run_c2_tfm_reward_nested.py (fixed random_state=seed),
so concatenating per-seed rows and re-aggregating is exact.

Usage:
  python experiments/merge_c2_allmetrics_8seed.py \
      --a outputs/paper_ready/c2_allmetrics_4seed/results_per_seed.csv \
      --b outputs/paper_ready/c2_allmetrics_seeds4567/results_per_seed.csv \
      --out outputs/paper_ready/c2_allmetrics_8seed
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = ("acc", "f1", "prec", "rec", "ece")


def main(a_csv, b_csv, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(a_csv), pd.read_csv(b_csv)], ignore_index=True)
    df = df.drop_duplicates(subset=["dataset", "seed"]).sort_values(["dataset", "seed"])
    df.to_csv(out / "results_per_seed.csv", index=False)

    rows = []
    for ds, g in df.groupby("dataset"):
        row = {"dataset": ds, "n_seeds": g["seed"].nunique()}
        for mode in ("rf", "tfm"):
            for m in METRICS:
                vals = g[f"{mode}_{m}"].dropna().values
                row[f"{mode}_{m}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
                row[f"{mode}_{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                row[f"{mode}_{m}_ci95"] = (1.96 * float(np.std(vals, ddof=1)) / np.sqrt(len(vals))
                                           if len(vals) > 1 else 0.0)
        rows.append(row)
    agg = pd.DataFrame(rows)
    agg.to_csv(out / "results_aggregated.csv", index=False)

    print(f"seeds={sorted(df.seed.unique())}  datasets={df.dataset.nunique()}  rows={len(df)}")
    print("\n(mean ± std across seeds)")
    print(f"{'dataset':18s} {'RF acc':>15} {'TFM acc':>15} {'RF f1':>15} {'TFM f1':>15}")
    for _, r in agg.iterrows():
        print(f"{r['dataset']:18s} "
              f"{r.rf_acc_mean:.4f}±{r.rf_acc_std:.4f} {r.tfm_acc_mean:.4f}±{r.tfm_acc_std:.4f} "
              f"{r.rf_f1_mean:.4f}±{r.rf_f1_std:.4f} {r.tfm_f1_mean:.4f}±{r.tfm_f1_std:.4f}")
    print("\nMEAN over datasets (mean ± mean-of-per-dataset-std):")
    for m in METRICS:
        rf, tfm = agg[f"rf_{m}_mean"].mean(), agg[f"tfm_{m}_mean"].mean()
        rfs, tfms = agg[f"rf_{m}_std"].mean(), agg[f"tfm_{m}_std"].mean()
        print(f"  {m:4s}: RF={rf:.4f}±{rfs:.4f}  TFM={tfm:.4f}±{tfms:.4f}  Δ(TFM-RF)={tfm-rf:+.4f}")
    for m in ("acc", "f1", "ece"):
        try:
            s, p = wilcoxon(agg[f"tfm_{m}_mean"], agg[f"rf_{m}_mean"])
            print(f"  Wilcoxon {m} (2-sided, n={len(agg)}): p={p:.3f}")
        except Exception as e:
            print(f"  Wilcoxon {m}: {e}")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="outputs/paper_ready/c2_allmetrics_4seed/results_per_seed.csv")
    ap.add_argument("--b", default="outputs/paper_ready/c2_allmetrics_seeds4567/results_per_seed.csv")
    ap.add_argument("--out", default="outputs/paper_ready/c2_allmetrics_8seed")
    a = ap.parse_args()
    main(a.a, a.b, a.out)
