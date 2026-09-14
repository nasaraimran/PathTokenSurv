import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import setup_publication_style, save_all

def paired_fold_plot(data_dir, outdir, metric, ylabel, stem):
    p = pd.read_csv(Path(data_dir) / "pathway_control_primary_fold_metrics.csv")
    c = pd.read_csv(Path(data_dir) / "control_fold_metrics.csv")
    d = p[["fold", metric]].merge(c[["fold", metric]], on="fold",
                                  suffixes=("_biological", "_permuted"))
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    x = [0, 1]
    for _, r in d.iterrows():
        ax.plot(x, [r[f"{metric}_biological"], r[f"{metric}_permuted"]],
                marker="o", linewidth=1.2)
    ax.set_xticks(x, ["Biological pathway\nassignment",
                      "Feature-permuted\nassignment"])
    ax.set_ylabel(ylabel)
    ax.set_title("Paired outer-fold pathway-token control")
    fig.tight_layout()
    save_all(fig, outdir / stem)

def forest_summary(data_dir, outdir):
    d = pd.read_csv(Path(data_dir) / "pathway_token_control_summary.csv")
    wanted = ["c_index", "cancer_stratified_c_index", "ibs",
              "mean_time_dependent_auc"]
    labels = ["Pooled C-index", "Cancer-stratified C-index",
              "IBS", "Mean tdAUC"]
    s = d.set_index("metric").loc[wanted].reset_index()
    y = np.arange(len(s))
    xerr = np.vstack([
        s["degradation_mean"] - s["degradation_ci95_low"],
        s["degradation_ci95_high"] - s["degradation_mean"],
    ])
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.errorbar(s["degradation_mean"], y, xerr=xerr, fmt="o", capsize=4)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Degradation after pathway-label permutation")
    ax.set_title("Biological pathway-token sensitivity")
    fig.tight_layout()
    save_all(fig, outdir / "Figure6C_Pathway_Control_Metric_Degradation")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--output-dir", default="outputs")
    a = ap.parse_args()
    setup_publication_style()
    out = Path(a.output_dir)
    paired_fold_plot(a.data_dir, out, "cancer_stratified_c_index",
                     "Cancer-stratified C-index",
                     "Figure6A_Pathway_Control_Stratified_CIndex")
    paired_fold_plot(a.data_dir, out, "c_index",
                     "Pooled C-index",
                     "Figure6B_Pathway_Control_Pooled_CIndex")
    forest_summary(a.data_dir, out)

if __name__ == "__main__":
    main()
