import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from common import setup_publication_style, save_all, ABLATION_LABELS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--output-dir", default="outputs")
    a = ap.parse_args()
    setup_publication_style()
    d = pd.read_csv(Path(a.data_dir) / "ablation_paired_fold_differences.csv")
    d = d[d["metric"] == "c_index"].copy()
    order = [
        "no_structured_masking",
        "no_reconstruction_objective",
        "no_subset_consistency",
        "no_cancer_conditioning",
        "no_pathway_bias",
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for i, abl in enumerate(order):
        s = d[d["ablation"] == abl]
        ax.scatter(s["degradation_positive_is_worse"], [i]*len(s), s=28)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(range(len(order)), [ABLATION_LABELS[x] for x in order])
    ax.invert_yaxis()
    ax.set_xlabel("Fold-wise C-index degradation")
    ax.set_title("Outer-fold variation in ablation effects")
    fig.tight_layout()
    save_all(fig, Path(a.output_dir) / "Supplementary_Figure_S5_Ablation_Foldwise")

if __name__ == "__main__":
    main()
