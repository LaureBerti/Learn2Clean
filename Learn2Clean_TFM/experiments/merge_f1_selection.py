"""Merge the F1-selection RICH shards (saga_richops_a/_b/_rc1/_rc2/_rc3) → full 13-ds 8-seed
multi-metric table. Report R7-vs-R3 under acc-selection (on acc) and F1-selection (on macro-F1),
plus whether F1-selection raises test-F1 over acc-selection — pooled, with Wilcoxon."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

PR = Path(__file__).parents[1] / "outputs/paper_ready"
SHARDS = ["saga_richops_a", "saga_richops_b", "saga_richops_rc1", "saga_richops_rc2", "saga_richops_rc3"]


def wil(d):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    try:
        return wilcoxon(d, alternative="greater").pvalue if np.any(np.abs(d) > 1e-12) else 1.0
    except Exception:
        return float("nan")


def main():
    parts = []
    for s in SHARDS:
        f = PR / s / "saga_richops_per_run.csv"
        if f.exists():
            parts.append(pd.read_csv(f))
    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["dataset", "seed"], keep="first")
    df.to_csv(PR / "saga_richops_F1_MERGED_per_run.csv", index=False)
    ds = sorted(df.dataset.unique())
    print(f"merged {len(df)} runs, {len(ds)} datasets, seeds={sorted(df.seed.unique())}")
    agg = df.groupby("dataset").mean(numeric_only=True)

    print("\n=== RICH pool — R7 (TabPFN-reward) vs R3 (RF-reward) ===")
    for label, dcol in [("ACC-selection → accuracy", "rich_acc_delta"),
                        ("F1-selection  → macro-F1", "rich_f1_delta")]:
        d = agg[dcol]
        print(f"  {label:26}: mean Δ(R7−R3)={d.mean():+.4f}, R7 wins {int((d>0).sum())}/{len(agg)}, p={wil(d):.3f}")

    print("\n=== Does F1-SELECTION raise test-F1 vs ACC-selection (same reward)? ===")
    for rw in ("r3", "r7"):
        g = agg[f"rich_{rw}f1_f1"] - agg[f"rich_{rw}acc_f1"]
        print(f"  {rw}: F1-sel − ACC-sel test-F1 = {g.mean():+.4f}, helps {int((g>0).sum())}/{len(agg)} ds, p={wil(g):.3f}")

    # spotlight the imbalanced datasets
    imb = [d for d in ["blood_transfusion", "adult", "bank_marketing", "AnimalShelter", "hepatitis"] if d in agg.index]
    cols = ["rich_r3f1_f1", "rich_r7f1_f1", "rich_f1_delta"]
    print("\n=== imbalanced datasets, F1-selection (R3-F1 vs R7-F1) ===")
    print(agg.loc[imb, cols].round(4).to_string())
    agg.reset_index().to_csv(PR / "saga_richops_F1_aggregated.csv", index=False)
    print("\nsaved → saga_richops_F1_aggregated.csv")


if __name__ == "__main__":
    main()
