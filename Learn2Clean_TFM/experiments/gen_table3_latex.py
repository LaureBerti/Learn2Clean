"""Generate LaTeX for the held-out protocol 8-seed C2 reward-comparison Table 3
(venue: acc/F1/ECE ; extended: acc/F1/prec/rec/ECE), mean+-std across seeds,
bold = better mean per dataset/metric. Reads results_aggregated.csv."""
import pandas as pd

AGG = "outputs/paper_ready/c2_allmetrics_8seed/results_aggregated.csv"
# paper display order (D1..D10) + hyphenated texttt names
ORDER = [("hepatitis","hepatitis"),("heart_statlog","heart-statlog"),
         ("ionosphere","ionosphere"),("blood_transfusion","blood-transfusion"),
         ("diabetes","diabetes"),("credit_g","credit-g"),("kr_vs_kp","kr-vs-kp"),
         ("phoneme","phoneme"),("adult","adult"),("bank_marketing","bank-marketing")]
df = pd.read_csv(AGG).set_index("dataset")


def cell(mean, std, best):
    s = f"{mean:.4f}{{\\tiny$\\pm${std:.4f}}}"
    return f"\\textbf{{{mean:.4f}}}{{\\tiny$\\pm${std:.4f}}}" if best else s


def row(key, disp, metrics):
    parts = [f"\\texttt{{{disp}}}", "8"]
    for m, lower_better in metrics:
        rf_m, rf_s = df.loc[key, f"rf_{m}_mean"], df.loc[key, f"rf_{m}_std"]
        tf_m, tf_s = df.loc[key, f"tfm_{m}_mean"], df.loc[key, f"tfm_{m}_std"]
        rf_best = (rf_m < tf_m) if lower_better else (rf_m > tf_m)
        tf_best = (tf_m < rf_m) if lower_better else (tf_m > rf_m)
        parts += [cell(rf_m, rf_s, rf_best), cell(tf_m, tf_s, tf_best)]
    return " & ".join(parts) + r" \\"


def mean_row(metrics):
    parts = ["Mean", "8"]
    for m, lower_better in metrics:
        rf_m = df[f"rf_{m}_mean"].mean(); rf_s = df[f"rf_{m}_std"].mean()
        tf_m = df[f"tfm_{m}_mean"].mean(); tf_s = df[f"tfm_{m}_std"].mean()
        rf_best = (rf_m < tf_m) if lower_better else (rf_m > tf_m)
        tf_best = (tf_m < rf_m) if lower_better else (tf_m > rf_m)
        parts += [cell(rf_m, rf_s, rf_best), cell(tf_m, tf_s, tf_best)]
    return " & ".join(parts) + r" \\"


def build(metrics, label, headers):
    ncol = 2 + 2 * len(metrics)
    colspec = "l c " + " ".join(["cc"] * len(metrics))
    head_groups = " & ".join([rf"\multicolumn{{2}}{{c}}{{{h}}}" for h in headers])
    cmids = "".join([rf"\cmidrule(lr){{{3+2*i}-{4+2*i}}}" for i in range(len(metrics))])
    subhead = " & ".join(["RF & TFM"] * len(metrics))
    lines = [r"\resizebox{\textwidth}{!}{%",
             rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
             rf"& Seeds & {head_groups} \\", cmids,
             rf"Dataset & & {subhead} \\", r"\midrule"]
    for key, disp in ORDER:
        lines.append(row(key, disp, metrics))
    lines += [r"\midrule", mean_row(metrics), r"\bottomrule",
              r"\end{tabular}}"]
    return "\n".join(lines)


VENUE = [("acc", False), ("f1", False), ("ece", True)]
EXT = [("acc", False), ("f1", False), ("prec", False), ("rec", False), ("ece", True)]

print("%%%%%%%%%%%%%%%%%%%%%% VENUE (acc / F1 / ECE) %%%%%%%%%%%%%%%%%%%%%%")
print(build(VENUE, "tab:c2_main_results",
            [r"Accuracy ($\uparrow$)", r"macro-$F_1$ ($\uparrow$)", r"ECE ($\downarrow$)"]))
print("\n\n%%%%%%%%%%%%%%%%%%%%%% EXTENDED (acc / F1 / prec / rec / ECE) %%%%%%%%%%%%%%%%%%%%%%")
print(build(EXT, "tab:c2_main_results_ext",
            [r"Acc ($\uparrow$)", r"$F_1$ ($\uparrow$)", r"Prec ($\uparrow$)",
             r"Rec ($\uparrow$)", r"ECE ($\downarrow$)"]))
