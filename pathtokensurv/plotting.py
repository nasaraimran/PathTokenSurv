from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_history(history: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["epoch"], history["train_total"], label="Training loss")
    ax.plot(history["epoch"], history["val_survival_loss"], label="Validation survival loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_mean_survival(
    survival: np.ndarray,
    horizons: Sequence[float],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean = survival.mean(axis=0)
    lower = np.quantile(survival, 0.25, axis=0)
    upper = np.quantile(survival, 0.75, axis=0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(horizons, mean, label="Mean predicted survival")
    ax.fill_between(horizons, lower, upper, alpha=0.2, label="Interquartile range")
    ax.set_xlabel("Time")
    ax.set_ylabel("Predicted survival probability")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
