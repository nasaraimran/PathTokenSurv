
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRIC_LABELS = {
    "c_index": "Pooled C-index",
    "cancer_stratified_c_index": "Cancer-stratified C-index",
    "ibs": "Integrated Brier score",
    "mean_time_dependent_auc": "Mean time-dependent AUC",
    "mean_cancer_stratified_time_dependent_auc": "Cancer-stratified mean tdAUC",
    "integrated_ipcw_ece": "Integrated IPCW-ECE",
}

MODEL_LABELS = {
    "PathTokenSurv": "PathTokenSurv",
    "clinical_only": "Clinical-only",
    "cancer_only": "Cancer-only",
}

ABLATION_LABELS = {
    "no_structured_masking": "No artificial modality drop",
    "no_reconstruction_objective": "No reconstruction objective",
    "no_subset_consistency": "No subset consistency",
    "no_cancer_conditioning": "No cancer conditioning",
    "no_pathway_bias": "No pathway bias",
}

CONDITION_LABELS = {
    "all_observed": "All observed",
    "drop_mrna": "Drop mRNA",
    "drop_mirna": "Drop miRNA",
    "drop_cnv": "Drop CNV",
    "clinical_only": "Clinical only",
    "complete": "Complete",
    "incomplete": "Incomplete",
}

def setup_publication_style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def save_all(fig, outstem):
    outstem = Path(outstem)
    outstem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(outstem.with_suffix("." + ext), bbox_inches="tight")
    plt.close(fig)

def asymmetric_yerr(mean, low, high):
    mean = np.asarray(mean, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    return np.vstack([mean - low, high - mean])
