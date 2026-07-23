"""Full held-out protocol 8-seed Table 3: 6 methods x 10 datasets, per-metric panels,
each cell mean+-std across seeds, bold = best per dataset/panel, Seeds column.
Sources:
  baselines_t3_8seed/baselines_per_seed.csv  -> B-NC/SP/SC/RAND
  c2_allmetrics_8seed/results_per_seed.csv    -> B-greedy-RF / B-greedy-TFM
Emits venue (Acc/F1/ECE) and extended (Acc/F1/Prec/Rec/ECE) LaTeX."""
import numpy as np, pandas as pd

BL = pd.read_csv("outputs/paper_ready/baselines_t3_8seed/baselines_per_seed.csv")
GR = pd.read_csv("outputs/paper_ready/c2_allmetrics_8seed/results_per_seed.csv")

ORDER = [("hepatitis","hepatitis"),("heart_statlog","heart-statlog"),
         ("ionosphere","ionosphere"),("blood_transfusion","blood-transfusion"),
         ("diabetes","diabetes"),("credit_g","credit-g"),("kr_vs_kp","kr-vs-kp"),
         ("phoneme","phoneme"),("adult","adult"),("bank_marketing","bank-marketing")]
# (display method, source df, column-prefix)
METHODS = [(r"\textbf{B-NC}",  BL, "b0_none"),
           (r"\textbf{B-SP}",  BL, "b1_standard"),
           (r"\textbf{B-SC}",  BL, "b2_full"),
           (r"\textbf{B-RAND}",BL, "b3_random"),
           (r"\textbf{B-greedy-RF}",  GR, "rf"),
           (r"\textbf{B-greedy-TFM}", GR, "tfm")]


def stat(df, prefix, ds, metric):
    g = df[df.dataset == ds][f"{prefix}_{metric}"].dropna().values
    return (float(np.mean(g)), float(np.std(g, ddof=1))) if len(g) else (np.nan, 0.0)


def panel(metric, lower_better, title, dec=3):
    # per-dataset means for bolding
    means = {ds: {mi: stat(df, pre, dso, metric)[0] for mi, (_, df, pre) in enumerate(METHODS)}
             for dso, ds in ORDER}
    best = {ds: (min if lower_better else max)(range(len(METHODS)), key=lambda i: means[ds][i])
            for _, ds in ORDER}
    header = " & ".join([r"Method", "Seeds"] + [rf"\texttt{{{d}}}" for _, d in ORDER] + ["Mean"])
    lines = [rf"\multicolumn{{{len(ORDER)+3}}}{{l}}{{\textit{{{title}}}}} \\", r"\midrule", header + r" \\", r"\midrule"]
    for mi, (disp, df, pre) in enumerate(METHODS):
        cells, ms = [], []
        for dso, ds in ORDER:
            m, s = stat(df, pre, dso, metric)
            ms.append(m)
            c = f"{m:.{dec}f}{{\\tiny$\\pm${s:.{dec}f}}}"
            cells.append(rf"\textbf{{{m:.{dec}f}}}{{\tiny$\pm${s:.{dec}f}}}" if best[ds] == mi else c)
        cells.append(f"{np.nanmean(ms):.{dec}f}")
        lines.append(" & ".join([disp, "8"] + cells) + r" \\")
    return "\n".join(lines)


def build(panels, cap_metrics):
    ncol = len(ORDER) + 3
    colspec = "l c " + "c" * (len(ORDER) + 1)
    out = [r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    for i, (metric, lb, title) in enumerate(panels):
        out.append(panel(metric, lb, title))
        out.append(r"\midrule" if i < len(panels) - 1 else "")
    out += [r"\bottomrule", r"\end{tabular}}"]
    return "\n".join([l for l in out if l != ""])


VENUE = [("acc", False, r"Accuracy ($\uparrow$)"),
         ("f1",  False, r"macro-$F_1$ ($\uparrow$)"),
         ("ece", True,  r"ECE ($\downarrow$)")]
EXT = VENUE[:2] + [("prec", False, r"macro-precision ($\uparrow$)"),
                   ("rec", False, r"macro-recall ($\uparrow$)")] + [VENUE[2]]

print("%%%%% VENUE (Acc / macro-F1 / ECE) %%%%%")
print(build(VENUE, 3))
print("\n\n%%%%% EXTENDED (Acc / F1 / Prec / Rec / ECE) %%%%%")
print(build(EXT, 3))
