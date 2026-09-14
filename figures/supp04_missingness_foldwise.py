import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from common import setup_publication_style, save_all

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--output-dir", default="outputs")
    a = ap.parse_args()
    setup_publication_style()
    d = pd.read_csv(Path(a.data_dir) / "natural_missingness_fold_metrics.csv")
    d = d[(d["group_type"] == "aggregate") & (d["group"].isin(["complete","incomplete"]))]

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for group in ["complete", "incomplete"]:
        s = d[d["group"] == group].sort_values("fold")
        ax.plot(s["fold"], s["cancer_stratified_c_index"], marker="o",
                label=group.capitalize())
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("Cancer-stratified C-index")
    ax.set_xticks([1,2,3,4,5])
    ax.set_title("Natural missingness across outer folds")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_all(fig, Path(a.output_dir) / "Supplementary_Figure_S4_Natural_Missingness_Foldwise")

if __name__ == "__main__":
    main()
