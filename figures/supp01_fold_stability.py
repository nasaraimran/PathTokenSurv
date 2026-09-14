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

    primary = pd.read_csv(Path(a.data_dir) / "outer_fold_metrics.csv")
    baseline = pd.read_csv(Path(a.data_dir) / "baseline_fold_metrics.csv")
    clinical = baseline[baseline["baseline"] == "clinical_only"]
    cancer = baseline[baseline["baseline"] == "cancer_only"]

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ax.plot(primary["fold"], primary["c_index"], marker="o", label="PathTokenSurv")
    ax.plot(clinical["fold"], clinical["c_index"], marker="o", label="Clinical-only")
    ax.plot(cancer["fold"], cancer["c_index"], marker="o", label="Cancer-only")
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("Pooled C-index")
    ax.set_xticks([1,2,3,4,5])
    ax.set_title("Outer-fold discrimination stability")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_all(fig, Path(a.output_dir) / "Supplementary_Figure_S1_Outer_Fold_Stability")

if __name__ == "__main__":
    main()
