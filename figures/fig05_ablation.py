import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import setup_publication_style, save_all, ABLATION_LABELS

def forest(df, metric_prefix, xlabel, stem, outdir):
    value = metric_prefix + "_degradation_mean"
    low = metric_prefix + "_degradation_ci95_low"
    high = metric_prefix + "_degradation_ci95_high"

    order = [
        "no_structured_masking",
        "no_reconstruction_objective",
        "no_subset_consistency",
        "no_cancer_conditioning",
        "no_pathway_bias",
    ]
    d = df.set_index("ablation").loc[order].reset_index()
    labels = [ABLATION_LABELS[x] for x in d["ablation"]]
    y = np.arange(len(d))

    xerr = np.vstack([d[value] - d[low], d[high] - d[value]])
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.errorbar(d[value], y, xerr=xerr, fmt="o", capsize=4)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title("Matched five-fold ablation effects")
    fig.tight_layout()
    save_all(fig, outdir / stem)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--output-dir", default="outputs")
    a = ap.parse_args()
    setup_publication_style()
    d = pd.read_csv(Path(a.data_dir) / "architecture_ablation_publication_table.csv")
    out = Path(a.output_dir)
    forest(d, "c_index",
           "C-index degradation (positive = full model better)",
           "Figure5A_Ablation_Pooled_CIndex_Degradation", out)
    forest(d, "cancer_stratified_c_index",
           "Cancer-stratified C-index degradation",
           "Figure5B_Ablation_Stratified_CIndex_Degradation", out)
    forest(d, "ibs",
           "IBS degradation (positive = ablation worse)",
           "Figure5C_Ablation_IBS_Degradation", out)

if __name__ == "__main__":
    main()
