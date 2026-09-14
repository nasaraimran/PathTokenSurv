from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from statsmodels.stats.multitest import multipletests


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def load_input(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"patient_id", "time_days", "event", "cancer_type", "risk_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input file is missing required columns: {sorted(missing)}"
        )

    x = df.copy()
    x["time_days"] = pd.to_numeric(x["time_days"], errors="coerce")
    x["event"] = pd.to_numeric(x["event"], errors="coerce")
    x["risk_score"] = pd.to_numeric(x["risk_score"], errors="coerce")
    x["cancer_type"] = x["cancer_type"].astype(str)

    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.dropna(subset=["time_days", "event", "cancer_type", "risk_score"])
    x = x[x["time_days"] > 0].copy()
    x["event"] = x["event"].astype(int)

    return x


def assign_risk_groups_within_cancer(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    med = x.groupby("cancer_type")["risk_score"].median().rename("risk_cut")
    x = x.join(med, on="cancer_type")
    x["risk_group_cancer"] = np.where(
        x["risk_score"] >= x["risk_cut"], "High risk", "Low risk"
    )
    return x


def compute_per_cancer_statistics(
    df: pd.DataFrame,
    min_events: int = 10,
    min_group_n: int = 10,
) -> pd.DataFrame:
    rows = []

    for cancer, sub in df.groupby("cancer_type", sort=True):
        low = sub[sub["risk_group_cancer"] == "Low risk"].copy()
        high = sub[sub["risk_group_cancer"] == "High risk"].copy()

        total_events = int(sub["event"].sum())
        stable = (
            len(low) >= min_group_n
            and len(high) >= min_group_n
            and total_events >= min_events
            and int(low["event"].sum()) > 0
            and int(high["event"].sum()) > 0
        )

        p_raw = np.nan
        hr = np.nan
        hr_lo = np.nan
        hr_hi = np.nan

        if stable:
            lr = logrank_test(
                low["time_days"],
                high["time_days"],
                event_observed_A=low["event"],
                event_observed_B=high["event"],
            )
            p_raw = float(lr.p_value)

            cox_df = sub[["time_days", "event", "risk_group_cancer"]].copy()
            cox_df["high_risk"] = (cox_df["risk_group_cancer"] == "High risk").astype(int)

            try:
                cph = CoxPHFitter()
                cph.fit(
                    cox_df[["time_days", "event", "high_risk"]],
                    duration_col="time_days",
                    event_col="event",
                    show_progress=False,
                )
                beta = float(cph.params_["high_risk"])
                hr = float(np.exp(beta))
                ci = cph.confidence_intervals_.loc["high_risk"]
                hr_lo = float(np.exp(ci.iloc[0]))
                hr_hi = float(np.exp(ci.iloc[1]))
            except Exception:
                pass

        rows.append({
            "cancer_type": cancer,
            "n": int(len(sub)),
            "events": total_events,
            "n_low": int(len(low)),
            "events_low": int(low["event"].sum()),
            "n_high": int(len(high)),
            "events_high": int(high["event"].sum()),
            "risk_cut_median": float(sub["risk_cut"].iloc[0]),
            "logrank_p_raw": p_raw,
            "hr_high_vs_low": hr,
            "hr_ci95_low": hr_lo,
            "hr_ci95_high": hr_hi,
            "stable_for_inference": bool(stable),
        })

    out = pd.DataFrame(rows).sort_values("cancer_type").reset_index(drop=True)
    out["logrank_p_holm"] = np.nan

    mask = out["logrank_p_raw"].notna()
    if mask.any():
        out.loc[mask, "logrank_p_holm"] = multipletests(
            out.loc[mask, "logrank_p_raw"].values,
            method="holm",
        )[1]

    return out


def select_main_cancers(stats: pd.DataFrame, n_main: int = 8) -> tuple[list[str], pd.DataFrame]:
    ranked = stats.copy()
    ranked["eligible_for_main"] = ranked["stable_for_inference"].astype(bool)

    eligible = ranked[ranked["eligible_for_main"]].copy()
    eligible = eligible.sort_values(
        ["events", "n", "cancer_type"],
        ascending=[False, False, True],
    )

    selected = eligible.head(n_main).copy()

    if selected.shape[0] < n_main:
        needed = n_main - selected.shape[0]
        fallback = ranked[~ranked["cancer_type"].isin(selected["cancer_type"])].copy()
        fallback = fallback.sort_values(
            ["events", "n", "cancer_type"],
            ascending=[False, False, True],
        )
        selected = pd.concat([selected, fallback.head(needed)], ignore_index=True)

    selected = selected.sort_values(
        ["events", "n", "cancer_type"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    reason_rows = []
    for i, row in selected.iterrows():
        reason_rows.append({
            "rank_for_main_figure": i + 1,
            "cancer_type": row["cancer_type"],
            "n": int(row["n"]),
            "events": int(row["events"]),
            "stable_for_inference": bool(row["stable_for_inference"]),
            "selection_rule": "Selected by descending observed event count, independent of model significance.",
        })

    return selected["cancer_type"].tolist(), pd.DataFrame(reason_rows)


def plot_single_cancer_validation(ax, sub: pd.DataFrame, stat_row: pd.Series) -> None:
    for label in ["Low risk", "High risk"]:
        g = sub[sub["risk_group_cancer"] == label]
        if len(g) == 0:
            continue

        km = KaplanMeierFitter(label=f"{label} (n={len(g)})")
        km.fit(g["time_days"] / 365.25, event_observed=g["event"])
        km.plot_survival_function(
            ax=ax,
            ci_show=True,
            censor_styles={"ms": 2.4},
            linewidth=1.6,
        )

    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Overall survival probability")
    ax.set_title(
        f'{stat_row["cancer_type"]} (n={int(stat_row["n"])}, events={int(stat_row["events"])})',
        fontsize=10,
    )

    if bool(stat_row["stable_for_inference"]):
        txt = f'log-rank p = {fmt_p(float(stat_row["logrank_p_raw"]))}'
        if np.isfinite(stat_row["hr_high_vs_low"]):
            txt += (
                f'\nHR = {float(stat_row["hr_high_vs_low"]):.2f} '
                f'({float(stat_row["hr_ci95_low"]):.2f}–{float(stat_row["hr_ci95_high"]):.2f})'
            )
    else:
        txt = "Descriptive only\ninsufficient events/group size"

    ax.text(
        0.98,
        0.98,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    ax.legend(frameon=False, fontsize=8, loc="best")


def save_main_figure(df: pd.DataFrame, stats: pd.DataFrame, main_cancers: list[str], outdir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5), squeeze=False)
    axes = axes.ravel()

    for i, cancer in enumerate(main_cancers):
        ax = axes[i]
        sub = df[df["cancer_type"] == cancer]
        stat_row = stats[stats["cancer_type"] == cancer].iloc[0]
        plot_single_cancer_validation(ax, sub, stat_row)
        ax.text(
            0.02, 0.98, f"({chr(ord('A') + i)})",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=11, fontweight="bold",
        )

    fig.suptitle(
        "Cancer-specific Kaplan–Meier validation of PathTokenSurv risk stratification",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975], w_pad=1.6, h_pad=1.8)

    stem = outdir / "Figure_Main_CancerSpecific_KM_Top8_ByEventCount"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def split_remaining_into_groups(remaining: list[str], group_size: int = 5) -> list[list[str]]:
    return [remaining[i:i + group_size] for i in range(0, len(remaining), group_size)]


def save_supplementary_figures(
    df: pd.DataFrame,
    stats: pd.DataFrame,
    supp_groups: list[list[str]],
    outdir: Path,
    dpi: int,
) -> None:
    supp_dir = outdir / "supplementary_figures"
    supp_dir.mkdir(parents=True, exist_ok=True)

    for fig_idx, group in enumerate(supp_groups, start=1):
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.8), squeeze=False)
        axes = axes.ravel()

        for i, cancer in enumerate(group):
            ax = axes[i]
            sub = df[df["cancer_type"] == cancer]
            stat_row = stats[stats["cancer_type"] == cancer].iloc[0]
            plot_single_cancer_validation(ax, sub, stat_row)
            ax.text(
                0.02, 0.98, f"({chr(ord('A') + i)})",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=11, fontweight="bold",
            )

        for j in range(len(group), len(axes)):
            axes[j].axis("off")

        fig.suptitle(
            f"Supplementary cancer-specific Kaplan–Meier validation ({fig_idx}/{len(supp_groups)})",
            fontsize=15,
            y=0.995,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.975], w_pad=1.4, h_pad=1.7)

        stem = supp_dir / f"Supplementary_KM_CancerSpecific_{fig_idx:02d}_of_{len(supp_groups):02d}"
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def save_assignments_and_group_lists(
    patient_df: pd.DataFrame,
    stats: pd.DataFrame,
    main_cancers: list[str],
    supp_groups: list[list[str]],
    outdir: Path,
) -> None:
    patient_df.to_csv(outdir / "per_cancer_km_patient_assignments.csv", index=False)
    stats.to_csv(outdir / "per_cancer_km_statistics_all33.csv", index=False)

    main_rows = []
    for i, cancer in enumerate(main_cancers, start=1):
        row = stats[stats["cancer_type"] == cancer].iloc[0]
        main_rows.append({
            "main_figure_position": i,
            "panel_letter": chr(ord("A") + i - 1),
            "cancer_type": cancer,
            "n": int(row["n"]),
            "events": int(row["events"]),
            "selection_rule": "Top 8 cancers by observed event count, independent of significance.",
        })
    pd.DataFrame(main_rows).to_csv(outdir / "main_figure_selected_cancers.csv", index=False)

    supp_rows = []
    for fig_idx, group in enumerate(supp_groups, start=1):
        for pos, cancer in enumerate(group, start=1):
            row = stats[stats["cancer_type"] == cancer].iloc[0]
            supp_rows.append({
                "supp_figure_number": fig_idx,
                "position_within_figure": pos,
                "cancer_type": cancer,
                "n": int(row["n"]),
                "events": int(row["events"]),
            })
    pd.DataFrame(supp_rows).to_csv(outdir / "supplementary_figure_cancer_assignments.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select 8 cancer types for the main manuscript by observed event count, "
            "then create cancer-specific Kaplan–Meier validation figures. "
            "The remaining 25 cancers are sent to supplementary figures."
        )
    )
    parser.add_argument("--input", required=True, help="Input patient-level OOF risk file.")
    parser.add_argument("--output-dir", default="outputs/tcga_km_main8_supp25")
    parser.add_argument("--min-events", type=int, default=10)
    parser.add_argument("--min-group-n", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_input(Path(args.input))
    df = assign_risk_groups_within_cancer(df)

    stats = compute_per_cancer_statistics(
        df,
        min_events=args.min_events,
        min_group_n=args.min_group_n,
    )

    if stats.shape[0] != 33:
        print(f"Warning: expected 33 cancer types, found {stats.shape[0]}.")

    main_cancers, selection_table = select_main_cancers(stats, n_main=8)
    selection_table.to_csv(outdir / "main_selection_rule_summary.csv", index=False)

    remaining = (
        stats.loc[~stats["cancer_type"].isin(main_cancers)]
        .sort_values(["events", "n", "cancer_type"], ascending=[False, False, True])["cancer_type"]
        .tolist()
    )
    supp_groups = split_remaining_into_groups(remaining, group_size=5)

    save_assignments_and_group_lists(df, stats, main_cancers, supp_groups, outdir)
    save_main_figure(df, stats, main_cancers, outdir, dpi=args.dpi)
    save_supplementary_figures(df, stats, supp_groups, outdir, dpi=args.dpi)

    print("Completed PathTokenSurv cancer-specific KM figure generation.")
    print(f"Selected main cancers: {', '.join(main_cancers)}")
    print(f"Supplementary groups: {len(supp_groups)}")
    print(f"Output directory: {outdir.resolve()}")


if __name__ == "__main__":
    main()
