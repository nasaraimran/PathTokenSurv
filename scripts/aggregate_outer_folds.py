from __future__ import annotations

# Allow direct execution from a source checkout.
import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from pathtokensurv.utils.io import save_json


CORE_METRICS = [
    "c_index",
    "cancer_stratified_c_index",
    "ibs",
    "mean_time_dependent_auc",
    "mean_cancer_stratified_time_dependent_auc",
    "integrated_ipcw_ece",
]


def _mean_ci(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        half = float(stats.t.ppf(0.975, df=n - 1) * sd / np.sqrt(n))
    else:
        half = np.nan
    return {
        "n": int(n),
        "mean": mean,
        "sd": sd,
        "ci95_low": float(mean - half) if np.isfinite(half) else np.nan,
        "ci95_high": float(mean + half) if np.isfinite(half) else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate completed PathTokenSurv outer folds.")
    parser.add_argument("--fold-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/tcga_outer_cv_v1.5.7_summary")
    args = parser.parse_args()

    rows = []
    baseline_rows = []
    for fold_dir_raw in args.fold_dirs:
        fold_dir = Path(fold_dir_raw)
        manifest_path = fold_dir / "experiment_manifest.json"
        metric_path = fold_dir / "test_metrics_extended.json"
        if not metric_path.exists():
            raise FileNotFoundError(f"Missing extended metrics: {metric_path}")
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        row = {
            "fold": int(manifest.get("outer_fold", len(rows) + 1)),
            "artifact_dir": str(fold_dir),
            **{key: metrics.get(key, np.nan) for key in CORE_METRICS},
        }
        rows.append(row)
        for baseline in ["cancer_only", "clinical_only"]:
            path = fold_dir / "baselines" / baseline / "test_metrics_extended.json"
            if not path.exists():
                continue
            bm = json.loads(path.read_text(encoding="utf-8"))
            baseline_rows.append(
                {
                    "fold": row["fold"],
                    "baseline": baseline,
                    **{key: bm.get(key, np.nan) for key in CORE_METRICS},
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_frame = pd.DataFrame(rows).sort_values("fold")
    fold_frame.to_csv(output_dir / "outer_fold_metrics.csv", index=False)

    summaries = []
    for metric in CORE_METRICS:
        summary = _mean_ci(fold_frame[metric].to_numpy(dtype=float))
        summaries.append({"model": "PathTokenSurv", "metric": metric, **summary})
    baseline_frame = pd.DataFrame(baseline_rows)
    if not baseline_frame.empty:
        baseline_frame.to_csv(output_dir / "baseline_fold_metrics.csv", index=False)
        for baseline, block in baseline_frame.groupby("baseline"):
            for metric in CORE_METRICS:
                summary = _mean_ci(block[metric].to_numpy(dtype=float))
                summaries.append({"model": baseline, "metric": metric, **summary})

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output_dir / "outer_cv_summary.csv", index=False)
    save_json(
        {
            "completed_folds": fold_frame["fold"].astype(int).tolist(),
            "fold_count": int(len(fold_frame)),
            "ci_method": "two-sided 95% Student-t interval across outer-fold point estimates",
            "note": "This is a cross-fold interval, not the later multiple-seed ensemble confidence interval.",
        },
        output_dir / "aggregation_manifest.json",
    )
    print(summary_frame.to_string(index=False))
    print(f"\nAggregated outer-fold artifacts: {output_dir}")


if __name__ == "__main__":
    main()
