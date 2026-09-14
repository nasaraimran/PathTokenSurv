"""
Optional cancer-specific figure.

Input CSV schema:
    cancer_type, c_index, events
Optionally:
    ci95_low, ci95_high

This is intentionally separate because the four supplied aggregate packages do
not contain a single finalized cross-fold cancer-level table.
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
    ap.add_argument("--min-events", type=int, default=20)
    a = ap.parse_args()
    setup_publication_style()
    d = pd.read_csv(a.csv)
    d = d[d["events"] >= a.min_events].sort_values("c_index", ascending=True)

    fig, ax = plt.subplots(figsize=(7.5, max(5.0, 0.23*len(d))))
    ax.barh(d["cancer_type"], d["c_index"])
    ax.axvline(0.5, linewidth=1)
    ax.set_xlabel("C-index")
    ax.set_ylabel("Cancer type")
    ax.set_title(f"Cancer-specific discrimination (≥{a.min_events} events)")
    fig.tight_layout()
    save_all(fig, Path(a.output_dir) / "Supplementary_Figure_Cancer_Specific_CIndex")

if __name__ == "__main__":
    main()
