from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from pathtokensurv.data.preprocessing import ProcessedCohort
from pathtokensurv.metrics import (
    KaplanMeier,
    cumulative_dynamic_auc_components,
    evaluate_survival_predictions,
    harrell_c_components,
    risk_from_survival,
    stratified_harrell_c_index,
)
from pathtokensurv.trainer import PredictionBundle
from pathtokensurv.utils.io import save_json


def _supported_horizon_indices(
    train_times: np.ndarray, train_events: np.ndarray, horizons: np.ndarray
) -> np.ndarray:
    """Mirror the support rule used by evaluate_survival_predictions."""
    censor_km = KaplanMeier.fit(train_times, 1 - train_events)
    censor_survival = censor_km.predict(horizons)
    upper_follow_up = float(np.quantile(train_times, 0.90))
    supported = (horizons <= upper_follow_up) & (censor_survival >= 0.05)
    idx = np.flatnonzero(supported)
    if idx.size == 0:
        idx = np.arange(max(1, min(len(horizons) - 1, len(horizons))))
    return idx


def _summary_auc(metrics: Dict[str, float]) -> float:
    vals = [float(v) for k, v in metrics.items() if k.startswith("auc_") and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _train_arrays(train: ProcessedCohort) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        train.times.detach().cpu().numpy().astype(float),
        train.events.detach().cpu().numpy().astype(int),
        train.cancers.detach().cpu().numpy().astype(int),
    )


def evaluate_per_cancer(
    train: ProcessedCohort,
    predictions: PredictionBundle,
    horizons: Sequence[float],
    calibration_bins: int,
) -> pd.DataFrame:
    """Evaluate each cancer using cancer-specific training censoring references.

    Small strata are retained in the table; metrics may be NaN when there are
    too few comparable patients/events. This is preferable to silently dropping
    difficult cancer types.
    """
    horizons_arr = np.asarray(horizons, dtype=float)
    rows = []
    train_times, train_events, train_cancers = _train_arrays(train)

    supported_idx = _supported_horizon_indices(train_times, train_events, horizons_arr)
    pooled_supported_horizons = horizons_arr[supported_idx]
    pooled_supported_survival = predictions.survival[:, supported_idx]
    pooled_risk = risk_from_survival(pooled_supported_survival, pooled_supported_horizons)

    for cancer_id in sorted(set(predictions.cancers.astype(int).tolist())):
        test_mask = predictions.cancers.astype(int) == cancer_id
        train_mask = train_cancers == cancer_id
        label = (
            train.cancer_labels[cancer_id]
            if 0 <= cancer_id < len(train.cancer_labels)
            else f"cancer_{cancer_id}"
        )
        concordant, comparable = harrell_c_components(
            predictions.times[test_mask], predictions.events[test_mask], pooled_risk[test_mask]
        )
        row = {
            "cancer_id": int(cancer_id),
            "cancer_type": label,
            "train_patients": int(train_mask.sum()),
            "train_events": int(train_events[train_mask].sum()),
            "test_patients": int(test_mask.sum()),
            "test_events": int(predictions.events[test_mask].sum()),
            "comparable_pairs": int(comparable),
        }
        if train_mask.sum() >= 2 and test_mask.sum() >= 2:
            try:
                metrics = evaluate_survival_predictions(
                    train_times=train_times[train_mask],
                    train_events=train_events[train_mask],
                    test_times=predictions.times[test_mask],
                    test_events=predictions.events[test_mask],
                    survival=predictions.survival[test_mask],
                    horizons=horizons_arr,
                    calibration_bins=calibration_bins,
                )
                row.update(
                    {
                        "c_index": (
                            float(concordant / comparable)
                            if comparable > 0
                            else float("nan")
                        ),
                        "ibs": metrics.get("ibs", float("nan")),
                        "mean_time_dependent_auc": metrics.get(
                            "mean_time_dependent_auc", _summary_auc(metrics)
                        ),
                        "evaluation_horizon_max": metrics.get(
                            "evaluation_horizon_max", float("nan")
                        ),
                    }
                )
            except (ValueError, ZeroDivisionError, FloatingPointError):
                row.update(
                    {
                        "c_index": float("nan"),
                        "ibs": float("nan"),
                        "mean_time_dependent_auc": float("nan"),
                        "evaluation_horizon_max": float("nan"),
                    }
                )
        else:
            row.update(
                {
                    "c_index": float("nan"),
                    "ibs": float("nan"),
                    "mean_time_dependent_auc": float("nan"),
                    "evaluation_horizon_max": float("nan"),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def stratified_discrimination_summary(
    train: ProcessedCohort,
    predictions: PredictionBundle,
    horizons: Sequence[float],
) -> Dict[str, float]:
    """Summarize pooled versus within-cancer Harrell concordance.

    The risk score uses exactly the same supported discrete-time horizons as the
    pooled test metric. The stratified C-index excludes every cross-cancer pair.
    """
    train_times, train_events, _ = _train_arrays(train)
    horizons_arr = np.asarray(horizons, dtype=float)
    supported_idx = _supported_horizon_indices(train_times, train_events, horizons_arr)
    supported_horizons = horizons_arr[supported_idx]
    risk = risk_from_survival(predictions.survival[:, supported_idx], supported_horizons)
    summary = stratified_harrell_c_index(
        predictions.times,
        predictions.events,
        risk,
        predictions.cancers,
    )
    pooled_concordant, pooled_pairs = harrell_c_components(
        predictions.times, predictions.events, risk
    )
    return {
        "pooled_c_index": (
            float(pooled_concordant / pooled_pairs) if pooled_pairs > 0 else float("nan")
        ),
        "cancer_stratified_c_index": float(summary["c_index"]),
        "macro_per_cancer_c_index": float(summary["macro_c_index"]),
        "within_cancer_comparable_pairs": int(summary["within_strata_comparable_pairs"]),
        "pooled_comparable_pairs": int(summary["pooled_comparable_pairs"]),
        "within_cancer_pair_fraction": float(summary["within_strata_pair_fraction"]),
        "cross_cancer_c_index": float(summary["cross_strata_c_index"]),
        "cross_cancer_comparable_pairs": int(summary["cross_strata_comparable_pairs"]),
        "evaluable_cancers_for_c_index": int(summary["evaluable_strata"]),
        "risk_horizon_max": float(supported_horizons[-1]),
    }


def stratified_auc_table(
    train: ProcessedCohort,
    predictions: PredictionBundle,
    horizons: Sequence[float],
) -> pd.DataFrame:
    """Cancer-stratified cumulative/dynamic AUC at supported horizons.

    For each cancer and horizon, IPCW uses that cancer's training censoring
    distribution. Weighted Mann--Whitney numerators/denominators are then pooled
    across cancers, so no case-control comparison crosses cancer type.
    """
    train_times, train_events, train_cancers = _train_arrays(train)
    horizons_arr = np.asarray(horizons, dtype=float)
    supported_idx = _supported_horizon_indices(train_times, train_events, horizons_arr)
    rows = []
    test_cancers = predictions.cancers.astype(int)

    for h_idx in supported_idx:
        horizon = float(horizons_arr[h_idx])
        total_num = 0.0
        total_den = 0.0
        per_cancer_auc = []
        evaluable = 0
        for cancer_id in np.unique(test_cancers):
            train_mask = train_cancers == cancer_id
            test_mask = test_cancers == cancer_id
            if train_mask.sum() < 2 or test_mask.sum() < 2:
                continue
            numerator, denominator = cumulative_dynamic_auc_components(
                train_times[train_mask],
                train_events[train_mask],
                predictions.times[test_mask],
                predictions.events[test_mask],
                1.0 - predictions.survival[test_mask, h_idx],
                horizon,
            )
            if denominator <= 0:
                continue
            evaluable += 1
            total_num += numerator
            total_den += denominator
            per_cancer_auc.append(float(numerator / denominator))
        rows.append(
            {
                "horizon": horizon,
                "cancer_stratified_auc": (
                    float(total_num / total_den) if total_den > 0 else float("nan")
                ),
                "macro_per_cancer_auc": (
                    float(np.mean(per_cancer_auc)) if per_cancer_auc else float("nan")
                ),
                "evaluable_cancers": int(evaluable),
                "ipcw_pair_weight_denominator": float(total_den),
            }
        )
    return pd.DataFrame(rows)


def ipcw_calibration_table(
    train_times: Sequence[float],
    train_events: Sequence[int],
    test_times: Sequence[float],
    test_events: Sequence[int],
    survival: np.ndarray,
    horizons: Sequence[float],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Create IPCW calibration summaries at supported discrete-time horizons.

    Patients censored before a horizon receive zero weight. Event cases at or
    before the horizon are weighted by 1/G(T-), while patients surviving beyond
    the horizon are weighted by 1/G(horizon), where G is the training censoring
    Kaplan--Meier estimate.
    """
    train_times = np.asarray(train_times, dtype=float)
    train_events = np.asarray(train_events, dtype=int)
    test_times = np.asarray(test_times, dtype=float)
    test_events = np.asarray(test_events, dtype=int)
    survival = np.asarray(survival, dtype=float)
    horizons_arr = np.asarray(horizons, dtype=float)
    supported_idx = _supported_horizon_indices(train_times, train_events, horizons_arr)
    censor_km = KaplanMeier.fit(train_times, 1 - train_events)

    rows = []
    for h_idx in supported_idx:
        horizon = float(horizons_arr[h_idx])
        pred = survival[:, h_idx]
        target = (test_times > horizon).astype(float)
        weights = np.zeros_like(test_times, dtype=float)

        survived = test_times > horizon
        g_h = float(np.clip(censor_km.predict(horizon)[0], 1e-6, None))
        weights[survived] = 1.0 / g_h

        failed = (test_times <= horizon) & (test_events == 1)
        if failed.any():
            g_event = np.clip(
                censor_km.predict(test_times[failed], left_limit=True), 1e-6, None
            )
            weights[failed] = 1.0 / g_event

        usable = weights > 0
        if usable.sum() < 2:
            continue
        p = pred[usable]
        y = target[usable]
        w = weights[usable]

        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 2:
            groups = np.zeros(len(p), dtype=int)
        else:
            groups = np.digitize(p, edges[1:-1], right=True)

        for group in np.unique(groups):
            mask = groups == group
            wg = w[mask]
            denom = float(wg.sum())
            if denom <= 0:
                continue
            rows.append(
                {
                    "horizon": horizon,
                    "bin": int(group) + 1,
                    "patients": int(mask.sum()),
                    "weight_sum": denom,
                    "mean_predicted_survival": float(np.sum(wg * p[mask]) / denom),
                    "ipcw_observed_survival": float(np.sum(wg * y[mask]) / denom),
                }
            )
    return pd.DataFrame(rows)


def summarize_ipcw_calibration(table: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, float]]:
    """Compute censoring-weighted expected calibration error (IPCW-ECE).

    At each horizon, absolute bin calibration gaps are weighted by the IPCW bin
    mass. The time-integrated summary uses trapezoidal integration divided by
    the supported time span; a simple mean/median are also reported explicitly.
    """
    columns = [
        "horizon",
        "ipcw_ece",
        "ipcw_signed_error",
        "total_weight",
        "calibration_bins",
    ]
    if table.empty:
        empty = pd.DataFrame(columns=columns)
        return empty, {
            "mean_ipcw_ece": float("nan"),
            "median_ipcw_ece": float("nan"),
            "max_ipcw_ece": float("nan"),
            "integrated_ipcw_ece": float("nan"),
            "supported_horizons": 0,
        }

    rows = []
    for horizon, block in table.groupby("horizon", sort=True):
        w = block["weight_sum"].to_numpy(dtype=float)
        gap = (
            block["mean_predicted_survival"].to_numpy(dtype=float)
            - block["ipcw_observed_survival"].to_numpy(dtype=float)
        )
        denom = float(w.sum())
        rows.append(
            {
                "horizon": float(horizon),
                "ipcw_ece": (
                    float(np.sum(w * np.abs(gap)) / denom) if denom > 0 else float("nan")
                ),
                "ipcw_signed_error": (
                    float(np.sum(w * gap) / denom) if denom > 0 else float("nan")
                ),
                "total_weight": denom,
                "calibration_bins": int(len(block)),
            }
        )
    summary_table = pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)
    finite = summary_table[np.isfinite(summary_table["ipcw_ece"])].copy()
    if finite.empty:
        integrated = float("nan")
        mean = median = maximum = float("nan")
    else:
        h = finite["horizon"].to_numpy(dtype=float)
        e = finite["ipcw_ece"].to_numpy(dtype=float)
        mean = float(np.mean(e))
        median = float(np.median(e))
        maximum = float(np.max(e))
        if len(e) == 1 or h[-1] <= h[0]:
            integrated = mean
        else:
            integrated = float(np.trapezoid(e, h) / (h[-1] - h[0]))
    summary = {
        "mean_ipcw_ece": mean,
        "median_ipcw_ece": median,
        "max_ipcw_ece": maximum,
        "integrated_ipcw_ece": integrated,
        "supported_horizons": int(len(finite)),
        "evaluation_horizon_max": (
            float(finite["horizon"].max()) if not finite.empty else float("nan")
        ),
    }
    return summary_table, summary


def save_calibration_plot(
    table: pd.DataFrame, output_path: str | Path, max_panels: int = 4
) -> None:
    """Plot a compact set of representative IPCW calibration horizons."""
    if table.empty:
        return
    import matplotlib.pyplot as plt

    horizons = np.sort(table["horizon"].unique())
    if len(horizons) > max_panels:
        positions = np.linspace(0, len(horizons) - 1, max_panels).round().astype(int)
        horizons = horizons[np.unique(positions)]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Ideal")
    for horizon in horizons:
        block = table[table["horizon"] == horizon].sort_values(
            "mean_predicted_survival"
        )
        ax.plot(
            block["mean_predicted_survival"],
            block["ipcw_observed_survival"],
            marker="o",
            linewidth=1.25,
            label=f"t={horizon:.0f}",
        )
    ax.set_xlabel("Mean predicted survival")
    ax.set_ylabel("IPCW observed survival")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(title="Horizon")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_scientific_evaluation(
    train: ProcessedCohort,
    predictions: PredictionBundle,
    horizons: Sequence[float],
    output_dir: str | Path,
    calibration_bins: int,
) -> Dict[str, float]:
    """Save publication-oriented held-out evaluation artifacts.

    v1.5.7 adds within-cancer discrimination, cancer-stratified time-dependent
    AUC, and explicit IPCW calibration summaries without changing model fitting
    or the frozen checkpoint-selection rule.
    """
    output_dir = Path(output_dir)
    per_cancer = evaluate_per_cancer(train, predictions, horizons, calibration_bins)
    per_cancer.to_csv(output_dir / "per_cancer_metrics.csv", index=False)

    discrimination = stratified_discrimination_summary(train, predictions, horizons)
    save_json(discrimination, output_dir / "stratified_discrimination.json")

    stratified_auc = stratified_auc_table(train, predictions, horizons)
    stratified_auc.to_csv(output_dir / "stratified_auc_by_horizon.csv", index=False)
    strat_auc_values = stratified_auc["cancer_stratified_auc"].to_numpy(dtype=float)
    macro_auc_values = stratified_auc["macro_per_cancer_auc"].to_numpy(dtype=float)
    discrimination.update(
        {
            "mean_cancer_stratified_time_dependent_auc": (
                float(np.nanmean(strat_auc_values))
                if np.isfinite(strat_auc_values).any()
                else float("nan")
            ),
            "mean_macro_per_cancer_time_dependent_auc": (
                float(np.nanmean(macro_auc_values))
                if np.isfinite(macro_auc_values).any()
                else float("nan")
            ),
        }
    )

    train_times, train_events, _ = _train_arrays(train)
    calibration = ipcw_calibration_table(
        train_times=train_times,
        train_events=train_events,
        test_times=predictions.times,
        test_events=predictions.events,
        survival=predictions.survival,
        horizons=horizons,
        n_bins=calibration_bins,
    )
    calibration.to_csv(output_dir / "calibration_ipcw.csv", index=False)
    calibration_summary_table, calibration_summary = summarize_ipcw_calibration(calibration)
    calibration_summary_table.to_csv(
        output_dir / "calibration_ipcw_summary.csv", index=False
    )
    save_json(calibration_summary, output_dir / "calibration_ipcw_summary.json")
    save_calibration_plot(calibration, output_dir / "calibration_ipcw.png")

    extended = {**discrimination, **calibration_summary}
    save_json(extended, output_dir / "scientific_evaluation_summary.json")
    return extended
