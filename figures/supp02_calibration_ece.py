import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from common import setup_publication_style, save_all, asymmetric_yerr, MODEL_LABELS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--output-dir", default="outputs")
    a = ap.parse_args()
    setup_publication_style()
    d = pd.read_csv(Path(a.data_dir) / "outer_cv_summary.csv")
    d = d[d["metric"] == "integrated_ipcw_ece"].copy()
    order = ["cancer_only", "clinical_only", "PathTokenSurv"]
    d["model"] = pd.Categorical(d["model"], categories=order, ordered=True)
    d = d.sort_values("model")
    labels = [MODEL_LABELS.get(str(x), str(x)) for x in d["model"]]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    yerr = asymmetric_yerr(d["mean"], d["ci95_low"], d["ci95_high"])
    bars = ax.bar(labels, d["mean"], yerr=yerr, capsize=4)
    ax.set_ylabel("Integrated IPCW-ECE")
    ax.set_title("Integrated calibration error across outer folds")
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylim(0, 0.11)
    for bar, value in zip(bars, d["mean"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save_all(fig, Path(a.output_dir) / "Supplementary_Figure_S2_Integrated_ECE")

if __name__ == "__main__":
    main()
