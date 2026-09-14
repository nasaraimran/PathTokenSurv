"""
Optional horizon-specific calibration figure.

Input CSV schema:
    horizon, predicted_survival, observed_survival

One output figure is created per horizon. This avoids inventing calibration
curves from integrated ECE summaries.
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from common import setup_publication_style, save_all

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--output-dir", default="outputs")
    a = ap.parse_args()
    setup_publication_style()
    d = pd.read_csv(a.csv)
    for horizon, s in d.groupby("horizon"):
        s = s.sort_values("predicted_survival")
        fig, ax = plt.subplots(figsize=(4.8, 4.6))
        ax.plot([0,1], [0,1], linestyle="--", linewidth=1, label="Ideal")
        ax.plot(s["predicted_survival"], s["observed_survival"],
                marker="o", label="PathTokenSurv")
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_xlabel("Predicted survival probability")
        ax.set_ylabel("Observed survival probability")
        ax.set_title(f"Calibration at {horizon}")
        ax.legend(frameon=False)
        fig.tight_layout()
        safe = str(horizon).replace(" ","_").replace("/","_")
        save_all(fig, Path(a.output_dir) / f"Supplementary_Calibration_{safe}")

if __name__ == "__main__":
    main()
