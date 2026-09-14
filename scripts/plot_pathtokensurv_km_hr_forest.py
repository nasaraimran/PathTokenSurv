from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_stats(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "cancer_type",
        "events",
        "hr_high_vs_low",
        "hr_ci95_low",
        "hr_ci95_high",
        "logrank_p_raw",
        "logrank_p_holm",
        "stable_for_inference",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Statistics file is missing required columns: {sorted(missing)}"
        )

    x = df.copy()
    x["stable_for_inference"] = x["stable_for_inference"].astype(bool)
    return x


def make_forest_plot(df: pd.DataFrame, output_dir: Path, dpi: int = 600) -> None:
    x = df.copy()

    x = x[np.isfinite(x["hr_high_vs_low"])].copy()
    x = x.sort_values(
        ["hr_high_vs_low", "events", "cancer_type"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    y = np.arange(len(x))

    fig_height = max(10, 0.32 * len(x) + 2.5)
    fig, ax = plt.subplots(figsize=(9.5, fig_height))

    ax.hlines(
        y=y,
        xmin=x["hr_ci95_low"],
        xmax=x["hr_ci95_high"],
        linewidth=1.2,
    )
    ax.plot(x["hr_high_vs_low"], y, "o")

    ax.axvline(1.0, linewidth=1.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(x["cancer_type"])
    ax.invert_yaxis()
    ax.set_xlabel("Hazard ratio for high-risk vs low-risk")
    ax.set_ylabel("Cancer type")
    ax.set_title("Cancer-specific hazard ratios for PathTokenSurv risk stratification")

    for yi, (_, row) in enumerate(x.iterrows()):
        label = (
            f'HR={row["hr_high_vs_low"]:.2f} '
            f'({row["hr_ci95_low"]:.2f}–{row["hr_ci95_high"]:.2f}); '
            f'Holm p={row["logrank_p_holm"]:.3g}'
            if np.isfinite(row["logrank_p_holm"])
            else f'HR={row["hr_high_vs_low"]:.2f} '
                 f'({row["hr_ci95_low"]:.2f}–{row["hr_ci95_high"]:.2f})'
        )
        ax.text(
            x["hr_ci95_high"].max() * 1.02,
            yi,
            label,
            va="center",
            fontsize=7.3,
        )

    xmax = float(np.nanmax(x["hr_ci95_high"])) if len(x) else 2.0
    ax.set_xlim(0, xmax * 1.55)

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "Figure_ForestPlot_CancerSpecific_HR"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    summary_cols = [
        "cancer_type", "events", "hr_high_vs_low",
        "hr_ci95_low", "hr_ci95_high",
        "logrank_p_raw", "logrank_p_holm", "stable_for_inference"
    ]
    x[summary_cols].to_csv(output_dir / "forest_plot_input_sorted.csv", index=False)

    print("Forest plot completed.")
    print(f"Output directory: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a forest plot of cancer-specific hazard ratios."
    )
    parser.add_argument("--stats", required=True, help="per_cancer_km_statistics_all33.csv")
    parser.add_argument("--output-dir", default="outputs/tcga_km_main8_supp25")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    df = load_stats(Path(args.stats))
    make_forest_plot(df, Path(args.output_dir), dpi=args.dpi)


if __name__ == "__main__":
    main()
