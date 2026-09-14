import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from common import setup_publication_style, save_all, asymmetric_yerr, CONDITION_LABELS

def controlled(data, outdir):
    d = data[data["panel"] == "controlled_complete_case_dropout"].copy()
    order = ["all_observed", "drop_mrna", "drop_mirna", "drop_cnv", "clinical_only"]
    d["condition"] = pd.Categorical(d["condition"], categories=order, ordered=True)
    d = d.sort_values("condition")
    labels = [CONDITION_LABELS.get(str(x), str(x)) for x in d["condition"]]

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.errorbar(range(len(d)), d["c_index_mean"], yerr=d["c_index_sd"],
                marker="o", capsize=4)
    ax.set_xticks(range(len(d)), labels, rotation=20, ha="right")
    ax.set_ylabel("Pooled C-index")
    ax.set_title("Controlled complete-case modality removal")
    ax.set_ylim(0.755, 0.790)
    fig.tight_layout()
    save_all(fig, outdir / "Figure4A_Controlled_Modality_Dropout_CIndex")

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.errorbar(range(len(d)), d["cancer_stratified_c_index_mean"],
                yerr=d["cancer_stratified_c_index_sd"], marker="o", capsize=4)
    ax.set_xticks(range(len(d)), labels, rotation=20, ha="right")
    ax.set_ylabel("Cancer-stratified C-index")
    ax.set_title("Within-cancer robustness to controlled modality removal")
    ax.set_ylim(0.63, 0.69)
    fig.tight_layout()
    save_all(fig, outdir / "Figure4B_Controlled_Modality_Dropout_Stratified_CIndex")

def natural(data, outdir):
    d = data[data["panel"] == "natural_missingness_observational"].copy()
    order = ["complete", "incomplete"]
    d["condition"] = pd.Categorical(d["condition"], categories=order, ordered=True)
    d = d.sort_values("condition")
    labels = [CONDITION_LABELS.get(str(x), str(x)) for x in d["condition"]]

    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    bars = ax.bar(labels, d["cancer_stratified_c_index_mean"],
                  yerr=d["cancer_stratified_c_index_sd"], capsize=4)
    ax.set_ylabel("Cancer-stratified C-index")
    ax.set_title("Natural missingness: complete vs incomplete patients")
    ax.set_ylim(0.57, 0.70)
    for bar, n in zip(bars, d["total_patients"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                f"n={int(n):,}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save_all(fig, outdir / "Figure4C_Natural_Missingness_Stratified_CIndex")

    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    bars = ax.bar(labels, d["ibs_mean"], yerr=d["ibs_sd"], capsize=4)
    ax.set_ylabel("Integrated Brier score")
    ax.set_title("Prediction error under natural missingness")
    ax.set_ylim(0.11, 0.19)
    for bar, n in zip(bars, d["total_patients"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                f"n={int(n):,}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save_all(fig, outdir / "Figure4D_Natural_Missingness_IBS")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--output-dir", default="outputs")
    a = ap.parse_args()
    setup_publication_style()
    d = pd.read_csv(Path(a.data_dir) / "missingness_robustness_publication_table.csv")
    out = Path(a.output_dir)
    controlled(d, out)
    natural(d, out)

if __name__ == "__main__":
    main()
