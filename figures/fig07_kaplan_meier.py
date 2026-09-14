"""
Generate the final Kaplan-Meier figure after the OOF KM analysis has been run.

Expected file:
    <km-dir>/km_oof_risk_group_assignments.csv

The input must contain:
    time_days, event, risk_group

Optional:
    cancer_type (used for the stratified Cox test)

This script deliberately does not create risk groups from survival outcomes.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import setup_publication_style, save_all

def fmt_p(p):
    return "<0.001" if p < 0.001 else f"={p:.3f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--km-dir", required=True)
    ap.add_argument("--output-dir", default="outputs")
    a = ap.parse_args()

    try:
        from lifelines import KaplanMeierFitter, CoxPHFitter
        from lifelines.plotting import add_at_risk_counts
        from lifelines.statistics import multivariate_logrank_test
    except ImportError as e:
        raise SystemExit("Install lifelines first: python -m pip install lifelines") from e

    setup_publication_style()
    f = Path(a.km_dir) / "km_oof_risk_group_assignments.csv"
    d = pd.read_csv(f)
    required = {"time_days","event","risk_group"}
    if not required.issubset(d.columns):
        raise ValueError(f"{f} must contain {sorted(required)}")

    labels = ["Q1 low", "Q2", "Q3", "Q4 high"]
    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    fitters = []
    for label in labels:
        s = d[d["risk_group"].astype(str) == label]
        km = KaplanMeierFitter(label=f"{label} (n={len(s):,})")
        km.fit(s["time_days"]/365.25, event_observed=s["event"])
        km.plot_survival_function(ax=ax, ci_show=True, censor_styles={"ms":3})
        fitters.append(km)

    lr = multivariate_logrank_test(
        d["time_days"], d["risk_group"].astype(str), d["event"]
    )
    txt = f"Global log-rank p {fmt_p(float(lr.p_value))}"

    if "cancer_type" in d.columns:
        x = d[["time_days","event","cancer_type","risk_group"]].copy()
        dum = pd.get_dummies(x["risk_group"].astype(str), prefix="risk",
                             drop_first=True, dtype=float)
        x = pd.concat([x.drop(columns=["risk_group"]), dum], axis=1)
        cph = CoxPHFitter()
        cph.fit(x, duration_col="time_days", event_col="event",
                strata=["cancer_type"])
        p = float(cph.log_likelihood_ratio_test().p_value)
        txt += f"\nCancer-stratified Cox global p {fmt_p(p)}"

    ax.text(0.98, 0.98, txt, transform=ax.transAxes, ha="right", va="top")
    ax.set_xlabel("Time since diagnosis (years)")
    ax.set_ylabel("Overall survival probability")
    ax.set_ylim(0, 1.02)
    ax.set_title("Kaplan–Meier survival by PathTokenSurv OOF risk quartile")
    add_at_risk_counts(*fitters, ax=ax, rows_to_show=["At risk"])
    fig.tight_layout()
    save_all(fig, Path(a.output_dir) / "Figure7_Kaplan_Meier_OOF_Risk_Quartiles")

if __name__ == "__main__":
    main()
