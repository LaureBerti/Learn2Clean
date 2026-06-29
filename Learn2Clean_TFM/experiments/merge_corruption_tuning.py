"""Merge the 4 corruption shards → 13-dataset by-ctype table (verdict ⑦) and the 3 tuning
shards → complete config table (verdict ⑪). Pure pandas; no TabPFN import."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

OUT = Path(__file__).parents[1] / "outputs/paper_ready"


def merge_corruption() -> pd.DataFrame:
    cor = pd.concat([pd.read_csv(OUT / f"corruption_{v}/corruption_per_run.csv") for v in "abcd"],
                    ignore_index=True)
    cor.to_csv(OUT / "corruption_by_ctype_MERGED_perrun.csv", index=False)
    agg = cor.groupby("ctype").agg(
        no_clean=("no_clean_tabpfn", "mean"), R3=("R3_tabpfn", "mean"),
        R7=("R7_tabpfn", "mean"), R7label=("R7label_tabpfn", "mean"),
        label_used=("R7label_uses_label", "mean")).reset_index()
    agg["gap_label_op"] = agg["R7label"] - agg["R7"]
    agg["gap_clean"] = agg["R7"] - agg["no_clean"]
    agg.to_csv(OUT / "corruption_by_ctype_MERGED.csv", index=False)
    print("=== CORRUPTION (13 ds, 5 seeds) — verdict 7 ===")
    print(agg[["ctype", "no_clean", "R7", "R7label", "label_used", "gap_clean", "gap_label_op"]]
          .to_string(index=False))
    print(f"\nmax |gap_clean| = {agg.gap_clean.abs().max():.4f}  (inert if <= ~0.003)")
    print(f"max gap_label_op = {agg.gap_label_op.max():+.4f}")
    print(f"datasets covered: {cor.dataset.nunique()}/13")
    return cor


def merge_tuning() -> None:
    tun = pd.concat([pd.read_csv(OUT / f"tuning_{v}/tuning_per_run.csv") for v in "012"],
                    ignore_index=True)
    tun.to_csv(OUT / "tuning_summary_MERGED_perrun.csv", index=False)
    print("\n=== TUNING per-run columns ===", list(tun.columns))
    # column auto-detect
    rfc = next((c for c in tun.columns if "rf" in c.lower() and "acc" in c.lower()), None)
    tfc = next((c for c in tun.columns if ("tfm" in c.lower() or "tab" in c.lower()) and "acc" in c.lower()), None)
    cfgc = next((c for c in tun.columns if "config" in c.lower()), None)
    print(f"rf={rfc} tfm={tfc} config={cfgc} | rows={len(tun)} ds={tun['dataset'].nunique() if 'dataset' in tun else '?'}")
    if rfc and tfc and cfgc:
        g = tun.groupby(cfgc).apply(lambda d: pd.Series({
            "rf_mean": d[rfc].mean(), "tfm_mean": d[tfc].mean(),
            "delta": d[tfc].mean() - d[rfc].mean(),
            "tfm_wins": int((d[tfc] > d[rfc]).sum()), "n": len(d)}), include_groups=False).reset_index()
        g.to_csv(OUT / "tuning_summary_MERGED.csv", index=False)
        print("\n=== TUNING (3 shards merged) — verdict 11 ===")
        print(g.to_string(index=False))
        print(f"\nALL configs delta(TFM-RF) < 0 ? {bool((g.delta < 0).all())} | "
              f"max delta = {g.delta.max():+.4f}")


if __name__ == "__main__":
    merge_corruption()
    merge_tuning()
