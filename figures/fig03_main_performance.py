import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from common import setup_publication_style, save_all, asymmetric_yerr, MODEL_LABELS, METRIC_LABELS

def plot_metric(df, metric, outdir, suffix):
    d = df[df["metric"] == metric].copy()
    order = ["cancer_only", "clinical_only", "PathTokenSurv"]
    d["model"] = pd.Categorical(d["model"], categories=order, ordered=True)
    d = d.sort_values("model")
    labels = [MODEL_LABELS.get(str(x), str(x)) for x in d["model"]]

    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    yerr = asymmetric_yerr(d["mean"], d["ci95_low"], d["ci95_high"])
    bars = ax.bar(labels, d["mean"], yerr=yerr, capsize=4)
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(METRIC_LABELS[metric] + " across five outer folds")
    ax.tick_params(axis="x", rotation=15)

    if metric in ("c_index", "cancer_stratified_c_index", "mean_time_dependent_auc"):
        ax.set_ylim(0.45 if metric == "cancer_stratified_c_index" else 0.68, 0.84)
    elif metric == "ibs":
        ax.set_ylim(0.12, 0.17)

    for bar, value in zip(bars, d["mean"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.3f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save_all(fig, outdir / suffix)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--output-dir", default="outputs")
    a = ap.parse_args()
    setup_publication_style()
    data = pd.read_csv(Path(a.data_dir) / "outer_cv_summary.csv")
    out = Path(a.output_dir)
    plot_metric(data, "c_index", out, "Figure3A_Main_Pooled_CIndex")
    plot_metric(data, "ibs", out, "Figure3B_Main_IBS")
    plot_metric(data, "mean_time_dependent_auc", out, "Figure3C_Main_Mean_tdAUC")
    plot_metric(data, "cancer_stratified_c_index", out, "Figure3D_Main_CancerStratified_CIndex")

if __name__ == "__main__":
    main()
