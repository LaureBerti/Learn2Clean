"""
experiments/fill_paper_tables.py

Post-process C1 and C5 experiment results and patch the corresponding
\todo{} markers in paper/main.tex.

Run this after run_c1_reward_benchmark.py and run_c5_param_ablation.py finish:

    PYTHONPATH=src python experiments/fill_paper_tables.py

Requires:
    outputs/paper_ready/c1_reward_benchmark/summary_by_reward.csv
    outputs/paper_ready/c5_param_ablation/results.csv

Edits paper/main.tex in-place; prints a summary of changes made.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parents[1]
PAPER = ROOT / "paper" / "main.tex"

C1_SUMMARY  = ROOT / "outputs" / "paper_ready" / "c1_reward_benchmark" / "summary_by_reward.csv"
C1_DETAILS  = ROOT / "outputs" / "paper_ready" / "c1_reward_benchmark" / "all_results.csv"
C5_RESULTS  = ROOT / "outputs" / "paper_ready" / "c5_param_ablation" / "results.csv"

# ---------------------------------------------------------------------------
# Map: reward class name → LaTeX row label pattern in paper
# Used to find the right row to patch.
# ---------------------------------------------------------------------------
REWARD_ROW_PATTERNS = {
    "CompletenessRetentionReward":       r"CompletenessRetentionReward \(R1\)",
    "AccuracyReward":                    r"AccuracyReward \(R2\)",
    "MultiObjectiveReward":              r"MultiObjectiveReward \(R3\)",
    "DriftPenaltyReward":                r"DriftPenaltyReward \(R4\)",
    "IncrementalGainReward":             r"IncrementalGainReward \(R5\)",
    "DataDistortionPenalty(dist)":       r"DataDistortionPenalty dist \(R6\)",
    "DataDistortionPenalty(acc+dist)":   r"DataDistortionPenalty acc\+dist \(R6\)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_paper() -> str:
    return PAPER.read_text()


def save_paper(text: str) -> None:
    PAPER.write_text(text)


def patch_todo(text: str, pattern: str, replacement: str, count: int = 1) -> tuple[str, int]:
    """Replace *count* occurrences of a \todo{...} near *pattern*."""
    # Find the line containing *pattern* then replace the first \todo{...} on it.
    lines = text.split("\n")
    replaced = 0
    for i, line in enumerate(lines):
        if re.search(pattern, line) and r"\todo{" in line:
            # Replace up to *count* \todo{...} on this line
            new_line, n = re.subn(r"\\todo\{[^{}]*\}", replacement, line, count=count)
            lines[i] = new_line
            replaced += n
            if replaced >= count:
                break
    return "\n".join(lines), replaced


# ---------------------------------------------------------------------------
# C1 — Table 1: reward taxonomy
# ---------------------------------------------------------------------------

def fill_c1_table(text: str) -> tuple[str, list[str]]:
    if not C1_SUMMARY.exists():
        print(f"[SKIP] C1 summary not found: {C1_SUMMARY}")
        return text, []

    df = pd.read_csv(C1_SUMMARY)
    changes: list[str] = []

    for _, row in df.iterrows():
        rf_name = row["reward_fn"]
        if rf_name not in REWARD_ROW_PATTERNS:
            print(f"  [WARN] Unknown reward name in CSV: {rf_name!r}")
            continue

        pattern = REWARD_ROW_PATTERNS[rf_name]
        mean_val = float(row["mean"])
        std_val  = float(row["std"])
        max_val  = float(row["max"])

        mean_str = f"{mean_val:.4f} $\\pm$ {std_val:.4f}"
        max_str  = f"{max_val:.4f}"

        # Patch first \todo (mean ± std), then second \todo (max)
        text, n1 = patch_todo(text, pattern, mean_str, count=1)
        text, n2 = patch_todo(text, pattern, max_str,  count=1)

        if n1 + n2 > 0:
            changes.append(f"  C1 row '{rf_name}': {mean_str} | max={max_str}")

    return text, changes


# ---------------------------------------------------------------------------
# C5 — Table C5: discrete vs parameterized comparison
# ---------------------------------------------------------------------------

def fill_c5_summary(text: str) -> tuple[str, list[str]]:
    """Patch the C5 narrative \todo{Key finding: ...} block."""
    if not C5_RESULTS.exists():
        print(f"[SKIP] C5 results not found: {C5_RESULTS}")
        return text, []

    df = pd.read_csv(C5_RESULTS)
    if df.empty or "mode" not in df.columns:
        return text, []

    disc = df[df["mode"] == "discrete"].set_index("dataset")
    param = df[df["mode"] == "param"].set_index("dataset")

    common = set(disc.index) & set(param.index)
    if not common:
        return text, []

    deltas = {ds: param.loc[ds, "best_score"] - disc.loc[ds, "best_score"]
              for ds in common}
    wins = sum(1 for d in deltas.values() if d > 1e-4)
    mean_delta = float(np.mean(list(deltas.values())))

    best_param_dataset  = max(deltas, key=lambda d: deltas[d])
    best_param_pipeline = param.loc[best_param_dataset, "best_pipeline"]

    finding = (
        f"Parameterized actions win on {wins}/{len(common)} datasets "
        f"(mean $\\Delta$={mean_delta:+.4f}). "
        f"Largest gain on \\texttt{{{best_param_dataset}}}: "
        f"\\texttt{{{best_param_pipeline}}}."
    )

    # Locate the C5 narrative \todo block in the paper
    pattern = r"\\todo\{Key finding: parameterised actions improve"
    new_text, n = re.subn(pattern + r"[^}]*\}", lambda m: finding, text, count=1)

    changes: list[str] = []
    if n:
        changes.append(f"  C5 narrative: wins={wins}/{len(common)}, mean delta={mean_delta:+.4f}")

    return new_text, changes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not PAPER.exists():
        print(f"ERROR: paper not found at {PAPER}")
        return

    text = load_paper()
    all_changes: list[str] = []

    print("=== fill_paper_tables.py ===")

    # C1
    print("\n[C1] Patching Table 1 (reward taxonomy) …")
    text, c1_changes = fill_c1_table(text)
    all_changes.extend(c1_changes)
    if c1_changes:
        for ch in c1_changes:
            print(ch)
    else:
        print("  (no changes made — check CSV or row label patterns)")

    # C5
    print("\n[C5] Patching parameterized ablation narrative …")
    text, c5_changes = fill_c5_summary(text)
    all_changes.extend(c5_changes)
    if c5_changes:
        for ch in c5_changes:
            print(ch)
    else:
        print("  (no changes made)")

    if all_changes:
        save_paper(text)
        print(f"\n✓ paper/main.tex updated ({len(all_changes)} patch(es) applied).")
    else:
        print("\n⚠ No patches applied — experiments may not have completed yet.")
        print("  Expected files:")
        print(f"    {C1_SUMMARY}")
        print(f"    {C5_RESULTS}")


if __name__ == "__main__":
    main()
