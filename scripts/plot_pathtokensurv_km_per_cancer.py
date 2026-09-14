
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from statsmodels.stats.multitest import multipletests

def fmt_p(p):
    if not np.isfinite(p): return "NA"
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def load_data(path):
    df = pd.read_csv(path)
    need = {"patient_id","time_days","event","cancer_type","risk_score"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns: {sorted(miss)}")
    df = df.dropna(subset=list(need)).copy()
    df = df[df["time_days"] > 0].copy()
    df["event"] = df["event"].astype(int)
    return df

def assign_groups(df):
    x = df.copy()
    cuts = x.groupby("cancer_type")["risk_score"].median().rename("risk_cut")
    x = x.join(cuts, on="cancer_type")
    x["risk_group"] = np.where(x["risk_score"] >= x["risk_cut"], "High risk", "Low risk")
    return x

def summarize(df, min_events=10, min_group_n=10):
    rows = []
    for cancer, s in df.groupby("cancer_type", sort=True):
        lo = s[s["risk_group"]=="Low risk"]
        hi = s[s["risk_group"]=="High risk"]
        stable = (len(lo)>=min_group_n and len(hi)>=min_group_n and
                  int(s["event"].sum())>=min_events and
                  int(lo["event"].sum())>0 and int(hi["event"].sum())>0)
        p = hr = cil = cih = np.nan
        if stable:
            p = float(logrank_test(lo["time_days"], hi["time_days"],
                                  event_observed_A=lo["event"],
                                  event_observed_B=hi["event"]).p_value)
            c = s[["time_days","event","risk_group"]].copy()
            c["high"] = (c["risk_group"]=="High risk").astype(int)
            try:
                m = CoxPHFitter()
                m.fit(c[["time_days","event","high"]], "time_days", "event")
                hr = float(np.exp(m.params_["high"]))
                ci = m.confidence_intervals_.loc["high"]
                cil, cih = float(np.exp(ci.iloc[0])), float(np.exp(ci.iloc[1]))
            except Exception:
                pass
        rows.append(dict(
            cancer_type=cancer, n=len(s), events=int(s["event"].sum()),
            n_low=len(lo), events_low=int(lo["event"].sum()),
            n_high=len(hi), events_high=int(hi["event"].sum()),
            logrank_p_raw=p, hr_high_vs_low=hr,
            hr_ci95_low=cil, hr_ci95_high=cih,
            stable_for_inference=stable
        ))
    out = pd.DataFrame(rows)
    out["logrank_p_holm"] = np.nan
    mask = out["logrank_p_raw"].notna()
    if mask.any():
        out.loc[mask,"logrank_p_holm"] = multipletests(
            out.loc[mask,"logrank_p_raw"], method="holm")[1]
    return out

def draw(ax, s, r):
    for label in ["Low risk","High risk"]:
        g = s[s["risk_group"]==label]
        km = KaplanMeierFitter(label=f"{label} (n={len(g)})")
        km.fit(g["time_days"]/365.25, event_observed=g["event"])
        km.plot_survival_function(ax=ax, ci_show=True, censor_styles={"ms":2.5})
    ax.set_ylim(0,1.02)
    ax.set_xlabel("Years")
    ax.set_ylabel("OS probability")
    ax.set_title(f'{r.cancer_type} (n={int(r.n)}, events={int(r.events)})', fontsize=9)
    if r.stable_for_inference:
        txt = f"log-rank p={fmt_p(r.logrank_p_raw)}\nHolm p={fmt_p(r.logrank_p_holm)}"
        if np.isfinite(r.hr_high_vs_low):
            txt += f"\nHR={r.hr_high_vs_low:.2f} ({r.hr_ci95_low:.2f}–{r.hr_ci95_high:.2f})"
    else:
        txt = "Descriptive only\ninsufficient events/group size"
    ax.text(0.98,0.98,txt,transform=ax.transAxes,ha="right",va="top",fontsize=7,
            bbox={"boxstyle":"round","alpha":0.08})
    ax.legend(frameon=False, fontsize=7)

def make_panels(df, stats, outdir, n_figures):
    chunks = [list(a) for a in np.array_split(stats["cancer_type"].tolist(), n_figures)]
    for i, chunk in enumerate(chunks, 1):
        n = len(chunk)
        if n <= 4:
            nr,nc,fs = 2,2,(10,8)
        elif n <= 6:
            nr,nc,fs = 2,3,(12,8)
        else:
            nr,nc,fs = 3,3,(12,10.5)
        fig, axes = plt.subplots(nr,nc,figsize=fs)
        axes = np.ravel(axes)
        for j,cancer in enumerate(chunk):
            s = df[df["cancer_type"]==cancer]
            r = stats[stats["cancer_type"]==cancer].iloc[0]
            draw(axes[j], s, r)
        for ax in axes[len(chunk):]:
            ax.axis("off")
        fig.suptitle(f"PathTokenSurv cancer-specific Kaplan–Meier analysis ({i}/{len(chunks)})")
        fig.tight_layout(rect=[0,0,1,0.97])
        stem = outdir/f"KM_PerCancer_Panel_{i:02d}_of_{len(chunks):02d}"
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
        plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="km_oof_risk_group_assignments.csv from the overall OOF KM analysis")
    ap.add_argument("--output-dir", default="outputs/tcga_km_per_cancer")
    ap.add_argument("--panel-set", type=int, choices=[4,8], default=8)
    ap.add_argument("--min-events", type=int, default=10)
    ap.add_argument("--min-group-n", type=int, default=10)
    args = ap.parse_args()

    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    df = assign_groups(load_data(args.input))
    stats = summarize(df, args.min_events, args.min_group_n)

    df.to_csv(outdir/"per_cancer_km_patient_assignments.csv", index=False)
    stats.to_csv(outdir/"per_cancer_km_statistics.csv", index=False)
    make_panels(df, stats, outdir, args.panel_set)

    print(f"Completed {len(stats)} cancer types.")
    print(f"Stable inferential analyses: {int(stats['stable_for_inference'].sum())}")
    print(outdir.resolve())

if __name__ == "__main__":
    main()
