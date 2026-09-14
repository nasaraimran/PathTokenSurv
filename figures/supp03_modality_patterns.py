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
    d = pd.read_csv(Path(a.data_dir) / "natural_missingness_summary.csv")
    d = d[d["group_type"] == "exact_pattern"].copy()
    d = d.sort_values("total_patients", ascending=True)

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    bars = ax.barh(d["pattern_name"], d["total_patients"])
    ax.set_xlabel("Patients")
    ax.set_ylabel("Observed modality pattern")
    ax.set_title("Natural modality-availability patterns")
    for bar, value in zip(bars, d["total_patients"]):
        ax.text(bar.get_width(), bar.get_y()+bar.get_height()/2,
                f" {int(value):,}", va="center", fontsize=8)
    fig.tight_layout()
    save_all(fig, Path(a.output_dir) / "Supplementary_Figure_S3_Modality_Availability_Patterns")

if __name__ == "__main__":
    main()
