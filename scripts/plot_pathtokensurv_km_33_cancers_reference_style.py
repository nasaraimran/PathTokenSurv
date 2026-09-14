from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from statsmodels.stats.multitest import multipletests


GROUP_SIZES = [4, 4, 4, 4, 4, 4, 4, 5]


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def load_km_data(path: Path) -> pd.DataFrame:
    """
    Load the OOF patient-level file created by the earlier PathTokenSurv
    Kaplan-Meier workflow.

    Required columns:
        patient_id, time_days, event, cancer_type

    Extra columns such as risk_score/risk_group are allowed and ignored here.
    """
    df = pd.read_csv(path)

    required = {"patient_id", "time_days", "event", "cancer_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    x = df.copy()
    x["time_days"] = pd.to_numeric(x["time_days"], errors="coerce")
    x["event"] = pd.to_numeric(x["event"], errors="coerce")
    x["cancer_type"] = x["cancer_type"].astype(str)

    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.dropna(subset=["time_days", "event", "cancer_type"])
    x = x[x["time_days"] > 0].copy()
    x["event"] = x["event"].astype(int)

    return x


def fixed_groups(cancers: list[str]) -> list[list[str]]:
    """
    Enforce exactly:
        4,4,4,4,4,4,4,5

    Cancers are sorted alphabetically for deterministic output.
    """
    cancers = sorted(cancers)

    if len(cancers) != 33:
        raise ValueError(
            f"Expected exactly 33 cancer types, but found {len(cancers)}."
        )

    groups = []
    start = 0

    for size in GROUP_SIZES:
        groups.append(cancers[start:start + size])
        start += size

    return groups


def panel_logrank(df: pd.DataFrame, cancers: list[str]) -> float:
    """
    Global log-rank test comparing the overall survival distributions of the
    cancer types shown in one subplot.

    IMPORTANT:
    This tests survival differences BETWEEN cancer types. It is not a
    PathTokenSurv high-vs-low risk test.
    """
    sub = df[df["cancer_type"].isin(cancers)].copy()

    if sub["cancer_type"].nunique() < 2:
        return np.nan

    result = multivariate_logrank_test(
        sub["time_days"],
        sub["cancer_type"],
        sub["event"],
    )

    return float(result.p_value)


def plot_group(
    ax,
    df: pd.DataFrame,
    cancers: list[str],
    panel_letter: str,
    raw_p: float,
    holm_p: float,
    max_years: float | None,
    show_pvalues: bool,
) -> None:
    """
    One subplot = 4 cancer-type KM curves, except the final subplot = 5 curves.
    This matches the visual logic of the supplied reference figure.
    """
    for cancer in cancers:
        sub = df[df["cancer_type"] == cancer]

        kmf = KaplanMeierFitter(label=f"{cancer} (n={len(sub)})")
        kmf.fit(
            sub["time_days"] / 365.25,
            event_observed=sub["event"],
        )
        kmf.plot_survival_function(
            ax=ax,
            ci_show=False,
            censor_styles={"ms": 2.2},
            linewidth=1.7,
        )

    ax.set_ylim(0.0, 1.03)

    if max_years is not None:
        ax.set_xlim(0.0, max_years)

    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Survival probability")

    cancer_names = ", ".join(cancers)
    ax.set_title(
        f"({panel_letter}) {cancer_names}",
        fontsize=9.5,
        loc="left",
    )

    if show_pvalues:
        ax.text(
            0.02,
            0.04,
            f"Global log-rank p {fmt_p(raw_p)}\nHolm p {fmt_p(holm_p)}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.3,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
        )

    ax.legend(
        frameon=False,
        fontsize=7.0,
        loc="best",
        handlelength=2.4,
    )


def save_group_list(groups: list[list[str]], outdir: Path) -> None:
    rows = []
    for i, group in enumerate(groups, start=1):
        letter = chr(ord("A") + i - 1)
        for position, cancer in enumerate(group, start=1):
            rows.append({
                "panel_number": i,
                "panel_letter": letter,
                "position": position,
                "cancer_type": cancer,
            })

    pd.DataFrame(rows).to_csv(
        outdir / "KM_reference_style_panel_assignments.csv",
        index=False,
    )


def create_master_figure(
    df: pd.DataFrame,
    groups: list[list[str]],
    panel_stats: pd.DataFrame,
    outdir: Path,
    layout: str,
    max_years: float | None,
    dpi: int,
    show_pvalues: bool,
) -> None:
    """
    Create ONE master figure containing all 8 grouped KM subplots.

    Default:
        4 columns x 2 rows

        A  B  C  D
        E  F  G  H
    """
    if layout == "4x2":
        ncols, nrows = 4, 2
        figsize = (28.0, 12.5)
    elif layout == "2x4":
        ncols, nrows = 2, 4
        figsize = (14.5, 23.0)
    else:
        raise ValueError("layout must be '4x2' or '2x4'")

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        squeeze=False,
    )

    axes = axes.ravel()

    for i, group in enumerate(groups):
        row = panel_stats.iloc[i]
        letter = chr(ord("A") + i)

        plot_group(
            ax=axes[i],
            df=df,
            cancers=group,
            panel_letter=letter,
            raw_p=float(row["global_logrank_p_raw"]),
            holm_p=float(row["global_logrank_p_holm"]),
            max_years=max_years,
            show_pvalues=show_pvalues,
        )

    fig.suptitle(
        "Overall survival across 33 TCGA cancer types",
        fontsize=17,
        y=0.995,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.975], w_pad=1.8, h_pad=2.0)

    stem = outdir / "KM_33_Cancers_8_Subplots_ReferenceStyle"

    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")

    plt.close(fig)


def create_separate_group_figures(
    df: pd.DataFrame,
    groups: list[list[str]],
    panel_stats: pd.DataFrame,
    outdir: Path,
    max_years: float | None,
    dpi: int,
    show_pvalues: bool,
) -> None:
    """
    Optional: also save each of the 8 grouped plots separately.
    Each file contains one axis with 4 cancer curves, except the last with 5.
    """
    separate_dir = outdir / "separate_grouped_plots"
    separate_dir.mkdir(parents=True, exist_ok=True)

    for i, group in enumerate(groups):
        row = panel_stats.iloc[i]
        letter = chr(ord("A") + i)

        fig, ax = plt.subplots(figsize=(7.0, 5.3))

        plot_group(
            ax=ax,
            df=df,
            cancers=group,
            panel_letter=letter,
            raw_p=float(row["global_logrank_p_raw"]),
            holm_p=float(row["global_logrank_p_holm"]),
            max_years=max_years,
            show_pvalues=show_pvalues,
        )

        fig.tight_layout()

        stem = separate_dir / f"KM_Group_{i+1:02d}_of_08"
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")

        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a reference-style Kaplan-Meier figure for all 33 TCGA "
            "cancer types: 8 subplots in one master figure, with cancer counts "
            "4,4,4,4,4,4,4,5. Each subplot overlays the overall KM curves for "
            "the cancer types assigned to that panel."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Patient-level CSV from the earlier KM workflow, e.g. "
            "km_oof_risk_group_assignments.csv."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/tcga_km_33_cancers_reference_style",
    )

    parser.add_argument(
        "--layout",
        choices=["4x2", "2x4"],
        default="4x2",
        help="Master figure layout. Default: 4 columns x 2 rows.",
    )

    parser.add_argument(
        "--max-years",
        type=float,
        default=None,
        help=(
            "Optional common x-axis maximum in years, e.g. --max-years 20. "
            "If omitted, matplotlib uses the observed range."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
    )

    parser.add_argument(
        "--no-pvalues",
        action="store_true",
        help="Do not display global panel log-rank and Holm-adjusted p-values.",
    )

    parser.add_argument(
        "--save-separate",
        action="store_true",
        help="Also save the 8 grouped plots as separate files.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_km_data(input_path)

    cancers = sorted(df["cancer_type"].unique().tolist())
    groups = fixed_groups(cancers)

    # Global log-rank test for each arbitrary 4/5-cancer panel.
    raw_ps = [panel_logrank(df, group) for group in groups]

    valid = np.isfinite(raw_ps)
    adjusted = np.full(len(raw_ps), np.nan, dtype=float)

    if np.any(valid):
        adjusted[np.asarray(valid)] = multipletests(
            np.asarray(raw_ps)[valid],
            method="holm",
        )[1]

    panel_stats = pd.DataFrame({
        "panel_number": np.arange(1, 9),
        "panel_letter": [chr(ord("A") + i) for i in range(8)],
        "cancers": [";".join(g) for g in groups],
        "global_logrank_p_raw": raw_ps,
        "global_logrank_p_holm": adjusted,
    })

    panel_stats.to_csv(
        outdir / "KM_reference_style_panel_statistics.csv",
        index=False,
    )

    save_group_list(groups, outdir)

    create_master_figure(
        df=df,
        groups=groups,
        panel_stats=panel_stats,
        outdir=outdir,
        layout=args.layout,
        max_years=args.max_years,
        dpi=args.dpi,
        show_pvalues=not args.no_pvalues,
    )

    if args.save_separate:
        create_separate_group_figures(
            df=df,
            groups=groups,
            panel_stats=panel_stats,
            outdir=outdir,
            max_years=args.max_years,
            dpi=args.dpi,
            show_pvalues=not args.no_pvalues,
        )

    print("Reference-style 33-cancer Kaplan-Meier figure completed.")
    print("Panel sizes: 4,4,4,4,4,4,4,5")
    print(f"Master layout: {args.layout}")
    print(f"Output directory: {outdir.resolve()}")


if __name__ == "__main__":
    main()
