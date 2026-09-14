
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import multivariate_logrank_test, logrank_test

try:
    from statsmodels.stats.multitest import multipletests
except Exception:
    multipletests = None

def pick_column(df, candidates, required=True):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise KeyError(f"Expected one of {candidates}; found {list(df.columns)}")
    return None

def fmt_p(p):
    if not np.isfinite(p):
        return "NA"
    return "<0.001" if p < 0.001 else f"={p:.3f}"

def load_oof_predictions(fold_dirs):
    frames = []
    for i, fold_dir in enumerate(fold_dirs, start=1):
        fold_dir = Path(fold_dir)
        pred_file = next((p for p in [
            fold_dir / "test_predictions_extended.csv",
            fold_dir / "test_predictions.csv",
        ] if p.exists()), None)
        if pred_file is None:
            raise FileNotFoundError(f"No prediction file found in {fold_dir}")
        df = pd.read_csv(pred_file)

        patient_col = pick_column(df, ["patient_id", "patient", "case_id"])
        time_col = pick_column(df, ["time", "observed_time", "survival_time", "os_time"])
        event_col = pick_column(df, ["event", "event_indicator", "status", "os_event"])
        cancer_col = pick_column(df, ["cancer_type", "cancer", "project_id", "cancer_label"])
        risk_col = pick_column(df, ["evaluation_risk_score", "risk_score", "risk"])

        frames.append(pd.DataFrame({
            "patient_id": df[patient_col].astype(str),
            "time_days": pd.to_numeric(df[time_col], errors="coerce"),
            "event": pd.to_numeric(df[event_col], errors="coerce"),
            "cancer_type": df[cancer_col].astype(str),
            "risk_score": pd.to_numeric(df[risk_col], errors="coerce"),
            "outer_fold": i,
        }))

    x = pd.concat(frames, ignore_index=True)
    x = x.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["time_days", "event", "risk_score", "cancer_type"]
    )
    x = x[x["time_days"] > 0].copy()
    x["event"] = x["event"].astype(int)
    if x["patient_id"].duplicated().any():
        raise ValueError("Duplicate OOF patient IDs detected.")
    return x

def assign_within_cancer_quartiles(df):
    x = df.copy()
    x["within_cancer_risk_percentile"] = (
        x.groupby("cancer_type")["risk_score"].rank(method="average", pct=True)
    )
    x["risk_group"] = pd.cut(
        x["within_cancer_risk_percentile"],
        bins=[0, .25, .50, .75, 1.0],
        labels=["Q1 low", "Q2", "Q3", "Q4 high"],
        include_lowest=True,
    )
    return x

def stratified_cox(df):
    x = df[["time_days", "event", "cancer_type", "risk_group"]].copy()
    dummies = pd.get_dummies(
        x["risk_group"].astype(str), prefix="risk", drop_first=True, dtype=float
    )
    x = pd.concat([x.drop(columns=["risk_group"]), dummies], axis=1)
    cph = CoxPHFitter()
    cph.fit(x, duration_col="time_days", event_col="event",
            strata=["cancer_type"], show_progress=False)
    lrt = cph.log_likelihood_ratio_test()

    hr = lo = hi = np.nan
    q4 = [c for c in cph.params_.index if "Q4 high" in c]
    if q4:
        c = q4[0]
        hr = float(np.exp(cph.params_[c]))
        ci = cph.confidence_intervals_.loc[c]
        lo, hi = float(np.exp(ci.iloc[0])), float(np.exp(ci.iloc[1]))
    return float(lrt.p_value), hr, lo, hi

def pairwise_tests(df):
    groups = ["Q1 low", "Q2", "Q3", "Q4 high"]
    rows = []
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            a = df[df["risk_group"].astype(str) == groups[i]]
            b = df[df["risk_group"].astype(str) == groups[j]]
            r = logrank_test(a["time_days"], b["time_days"],
                             event_observed_A=a["event"],
                             event_observed_B=b["event"])
            rows.append({"group_a": groups[i], "group_b": groups[j],
                         "p_raw": float(r.p_value)})
    out = pd.DataFrame(rows)
    if multipletests is not None and not out.empty:
        out["p_holm"] = multipletests(out["p_raw"], method="holm")[1]
    return out

def numbers_at_risk(df, years=(0,1,2,3,5)):
    rows = []
    for group, sub in df.groupby("risk_group", observed=True):
        for year in years:
            rows.append({
                "risk_group": str(group),
                "time_years": year,
                "n_at_risk": int((sub["time_days"] >= year*365.25).sum())
            })
    return pd.DataFrame(rows)

def make_plot(df, outdir):
    labels = ["Q1 low", "Q2", "Q3", "Q4 high"]
    fitters = []
    fig, ax = plt.subplots(figsize=(8.3, 6.6))
    for label in labels:
        sub = df[df["risk_group"].astype(str) == label]
        kmf = KaplanMeierFitter(label=f"{label} (n={len(sub):,})")
        kmf.fit(sub["time_days"]/365.25, event_observed=sub["event"])
        kmf.plot_survival_function(ax=ax, ci_show=True, censor_styles={"ms":3})
        fitters.append(kmf)

    lr = multivariate_logrank_test(
        df["time_days"], df["risk_group"].astype(str), df["event"]
    )
    p_strat, hr, lo, hi = stratified_cox(df)

    ax.set_xlabel("Time since diagnosis (years)")
    ax.set_ylabel("Overall survival probability")
    ax.set_ylim(0, 1.02)
    ax.set_title("Kaplan–Meier survival by PathTokenSurv out-of-fold risk quartile")

    txt = (
        f"Pooled log-rank p {fmt_p(float(lr.p_value))}\n"
        f"Cancer-stratified Cox global p {fmt_p(p_strat)}"
    )
    if np.isfinite(hr):
        txt += f"\nQ4 vs Q1 HR {hr:.2f} (95% CI {lo:.2f}–{hi:.2f})"
    ax.text(0.98, 0.98, txt, transform=ax.transAxes,
            ha="right", va="top",
            bbox={"boxstyle":"round", "alpha":0.08})

    add_at_risk_counts(*fitters, ax=ax, rows_to_show=["At risk"])
    fig.tight_layout()
    fig.savefig(outdir/"KM_PathTokenSurv_OOF_Risk_Quartiles.pdf", bbox_inches="tight")
    fig.savefig(outdir/"KM_PathTokenSurv_OOF_Risk_Quartiles.png", dpi=600, bbox_inches="tight")
    fig.savefig(outdir/"KM_PathTokenSurv_OOF_Risk_Quartiles.svg", bbox_inches="tight")
    plt.close(fig)

    return {
        "pooled_logrank_p": float(lr.p_value),
        "cancer_stratified_cox_global_p": p_strat,
        "q4_vs_q1_stratified_hr": hr,
        "q4_vs_q1_stratified_hr_ci_low": lo,
        "q4_vs_q1_stratified_hr_ci_high": hi,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-dirs", nargs="+", required=True)
    ap.add_argument("--output-dir", default="outputs/tcga_kaplan_meier_oof")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = assign_within_cancer_quartiles(load_oof_predictions(args.fold_dirs))
    summary = make_plot(df, outdir)

    df.to_csv(outdir/"km_oof_risk_group_assignments.csv", index=False)
    numbers_at_risk(df).to_csv(outdir/"km_numbers_at_risk.csv", index=False)
    pairwise_tests(df).to_csv(outdir/"km_pairwise_logrank_holm.csv", index=False)

    group_summary = df.groupby("risk_group", observed=True).agg(
        n=("patient_id","size"),
        events=("event","sum"),
        median_risk=("risk_score","median"),
        median_followup_days=("time_days","median"),
    ).reset_index()
    group_summary.to_csv(outdir/"km_group_summary.csv", index=False)

    summary.update({
        "n_patients": int(len(df)),
        "n_events": int(df["event"].sum()),
        "group_rule": (
            "Quartiles of out-of-fold PathTokenSurv evaluation risk score ranked "
            "within cancer type; outcomes were not used to define risk groups."
        ),
    })
    (outdir/"km_statistical_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Outputs: {outdir}")

if __name__ == "__main__":
    main()
