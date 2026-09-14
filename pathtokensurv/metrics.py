from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np


@dataclass
class KaplanMeier:
    times: np.ndarray
    survival: np.ndarray

    @classmethod
    def fit(cls, times: Sequence[float], event_indicators: Sequence[int]) -> "KaplanMeier":
        t = np.asarray(times, dtype=float)
        e = np.asarray(event_indicators, dtype=int)
        order = np.argsort(t)
        t = t[order]
        e = e[order]
        unique_times = np.unique(t)
        survival_values = []
        current = 1.0
        for time in unique_times:
            at_risk = np.sum(t >= time)
            events = np.sum((t == time) & (e == 1))
            if at_risk > 0:
                current *= 1.0 - events / at_risk
            survival_values.append(current)
        return cls(times=unique_times, survival=np.asarray(survival_values, dtype=float))

    def predict(self, query: Sequence[float] | float, left_limit: bool = False) -> np.ndarray:
        q = np.atleast_1d(np.asarray(query, dtype=float))
        side = "left" if left_limit else "right"
        indices = np.searchsorted(self.times, q, side=side) - 1
        result = np.ones_like(q, dtype=float)
        valid = indices >= 0
        result[valid] = self.survival[indices[valid]]
        return result


def harrell_c_components(
    times: Sequence[float], events: Sequence[int], risk: Sequence[float]
) -> tuple[float, int]:
    """Return concordant credit and comparable-pair count for Harrell C.

    Larger risk means earlier event. Ties in predicted risk receive half credit,
    matching :func:`harrell_c_index`. Exposing the numerator and denominator
    allows cancer-stratified concordance to be pooled without introducing
    cross-cancer comparable pairs.
    """

    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    r = np.asarray(risk, dtype=float)
    concordant = 0.0
    comparable = 0
    for i in np.flatnonzero(e == 1):
        mask = t > t[i]
        n_comparable = int(mask.sum())
        if n_comparable == 0:
            continue
        differences = r[i] - r[mask]
        concordant += float(np.sum(differences > 0) + 0.5 * np.sum(differences == 0))
        comparable += n_comparable
    return float(concordant), int(comparable)


def harrell_c_index(times: Sequence[float], events: Sequence[int], risk: Sequence[float]) -> float:
    """Harrell concordance index; larger risk means earlier event."""

    concordant, comparable = harrell_c_components(times, events, risk)
    return float(concordant / comparable) if comparable > 0 else float("nan")


def stratified_harrell_c_index(
    times: Sequence[float],
    events: Sequence[int],
    risk: Sequence[float],
    strata: Sequence[int] | Sequence[str],
) -> Dict[str, float]:
    """Cancer/stratum-restricted Harrell concordance summary.

    Comparable pairs are formed only within the same stratum. The returned
    ``c_index`` is comparable-pair weighted (equivalently, numerator/denominator
    pooled across strata). ``macro_c_index`` is the unweighted mean across
    strata with at least one comparable pair. The function also reports the
    cross-stratum component of the ordinary pooled C-index for transparency.
    """

    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    r = np.asarray(risk, dtype=float)
    g = np.asarray(strata)
    if not (len(t) == len(e) == len(r) == len(g)):
        raise ValueError("times, events, risk, and strata must have equal length.")

    pooled_concordant, pooled_comparable = harrell_c_components(t, e, r)
    within_concordant = 0.0
    within_comparable = 0
    per_stratum = []
    for value in np.unique(g):
        mask = g == value
        concordant, comparable = harrell_c_components(t[mask], e[mask], r[mask])
        if comparable > 0:
            per_stratum.append(float(concordant / comparable))
            within_concordant += concordant
            within_comparable += comparable

    cross_concordant = pooled_concordant - within_concordant
    cross_comparable = pooled_comparable - within_comparable
    return {
        "c_index": (
            float(within_concordant / within_comparable)
            if within_comparable > 0
            else float("nan")
        ),
        "macro_c_index": float(np.mean(per_stratum)) if per_stratum else float("nan"),
        "within_strata_comparable_pairs": int(within_comparable),
        "pooled_comparable_pairs": int(pooled_comparable),
        "within_strata_pair_fraction": (
            float(within_comparable / pooled_comparable)
            if pooled_comparable > 0
            else float("nan")
        ),
        "cross_strata_c_index": (
            float(cross_concordant / cross_comparable)
            if cross_comparable > 0
            else float("nan")
        ),
        "cross_strata_comparable_pairs": int(cross_comparable),
        "evaluable_strata": int(len(per_stratum)),
    }


def risk_from_survival(survival: np.ndarray, right_edges: Sequence[float]) -> np.ndarray:
    """Use negative restricted mean survival time as a continuous risk score."""

    s = np.asarray(survival, dtype=float)
    edges = np.asarray(right_edges, dtype=float)
    left = np.concatenate([[0.0], edges[:-1]])
    widths = edges - left
    rmst = np.sum(s * widths[None, :], axis=1)
    return -rmst


def ipcw_brier_score(
    train_times: Sequence[float],
    train_events: Sequence[int],
    test_times: Sequence[float],
    test_events: Sequence[int],
    survival_at_horizon: Sequence[float],
    horizon: float,
) -> float:
    train_times = np.asarray(train_times, dtype=float)
    train_events = np.asarray(train_events, dtype=int)
    test_times = np.asarray(test_times, dtype=float)
    test_events = np.asarray(test_events, dtype=int)
    pred = np.asarray(survival_at_horizon, dtype=float)

    censor_km = KaplanMeier.fit(train_times, 1 - train_events)
    g_t = float(np.clip(censor_km.predict(horizon)[0], 1e-6, None))
    weights = np.zeros_like(test_times, dtype=float)
    target = (test_times > horizon).astype(float)

    survived = test_times > horizon
    weights[survived] = 1.0 / g_t

    failed = (test_times <= horizon) & (test_events == 1)
    if failed.any():
        g_event = np.clip(censor_km.predict(test_times[failed], left_limit=True), 1e-6, None)
        weights[failed] = 1.0 / g_event

    return float(np.mean(weights * (target - pred) ** 2))


def integrated_brier_score(
    train_times: Sequence[float],
    train_events: Sequence[int],
    test_times: Sequence[float],
    test_events: Sequence[int],
    survival: np.ndarray,
    horizons: Sequence[float],
) -> float:
    horizons_arr = np.asarray(horizons, dtype=float)
    if survival.shape[1] != len(horizons_arr):
        raise ValueError("Survival columns must match horizons.")
    scores = np.asarray(
        [
            ipcw_brier_score(
                train_times,
                train_events,
                test_times,
                test_events,
                survival[:, idx],
                horizon,
            )
            for idx, horizon in enumerate(horizons_arr)
        ]
    )
    if len(scores) == 1:
        return float(scores[0])
    span = horizons_arr[-1] - horizons_arr[0]
    if span <= 0:
        return float(np.mean(scores))
    return float(np.trapezoid(scores, horizons_arr) / span)


def cumulative_dynamic_auc_components(
    train_times: Sequence[float],
    train_events: Sequence[int],
    test_times: Sequence[float],
    test_events: Sequence[int],
    risk_at_horizon: Sequence[float],
    horizon: float,
) -> tuple[float, float]:
    """Return IPCW AUC numerator and denominator at one horizon.

    Exposing components permits a cancer-stratified cumulative/dynamic AUC in
    which case-control comparisons are formed only within cancer type and the
    weighted Mann--Whitney numerators/denominators are pooled across cancers.
    """

    train_times = np.asarray(train_times, dtype=float)
    train_events = np.asarray(train_events, dtype=int)
    test_times = np.asarray(test_times, dtype=float)
    test_events = np.asarray(test_events, dtype=int)
    risk = np.asarray(risk_at_horizon, dtype=float)

    cases = (test_times <= horizon) & (test_events == 1)
    controls = test_times > horizon
    if cases.sum() == 0 or controls.sum() == 0:
        return 0.0, 0.0

    censor_km = KaplanMeier.fit(train_times, 1 - train_events)
    case_weights = 1.0 / np.clip(
        censor_km.predict(test_times[cases], left_limit=True), 1e-6, None
    )
    control_weight = 1.0 / float(
        np.clip(censor_km.predict(horizon)[0], 1e-6, None)
    )

    case_risk = risk[cases]
    control_risk = risk[controls]
    control_weights = np.full(control_risk.shape, control_weight, dtype=float)

    # Weighted Mann--Whitney statistic without a cases-by-controls matrix.
    order = np.argsort(control_risk, kind="mergesort")
    sorted_risk = control_risk[order]
    sorted_weights = control_weights[order]
    cumulative = np.concatenate([[0.0], np.cumsum(sorted_weights)])
    numerator = 0.0
    for value, case_weight in zip(case_risk, case_weights):
        left = np.searchsorted(sorted_risk, value, side="left")
        right = np.searchsorted(sorted_risk, value, side="right")
        lower_weight = cumulative[left]
        tie_weight = cumulative[right] - cumulative[left]
        numerator += case_weight * (lower_weight + 0.5 * tie_weight)
    denominator = float(case_weights.sum() * control_weights.sum())
    return float(numerator), float(denominator)


def cumulative_dynamic_auc(
    train_times: Sequence[float],
    train_events: Sequence[int],
    test_times: Sequence[float],
    test_events: Sequence[int],
    risk_at_horizon: Sequence[float],
    horizon: float,
) -> float:
    numerator, denominator = cumulative_dynamic_auc_components(
        train_times,
        train_events,
        test_times,
        test_events,
        risk_at_horizon,
        horizon,
    )
    return float(numerator / denominator) if denominator > 0 else float("nan")


def calibration_error(
    times: Sequence[float],
    events: Sequence[int],
    predicted_survival: Sequence[float],
    horizon: float,
    n_bins: int = 10,
) -> float:
    """Simple bin-based calibration error.

    This diagnostic uses binary survival status among patients whose status is
    known at the horizon. IPCW calibration should be used in the final paper.
    """

    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    p = np.asarray(predicted_survival, dtype=float)
    known = (t > horizon) | ((t <= horizon) & (e == 1))
    if known.sum() < 2:
        return float("nan")
    y = (t[known] > horizon).astype(float)
    p = p[known]
    quantile_edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(quantile_edges) < 2:
        return float(abs(y.mean() - p.mean()))
    bins = np.digitize(p, quantile_edges[1:-1], right=True)
    error = 0.0
    for b in np.unique(bins):
        mask = bins == b
        error += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(error)


def evaluate_survival_predictions(
    train_times: Sequence[float],
    train_events: Sequence[int],
    test_times: Sequence[float],
    test_events: Sequence[int],
    survival: np.ndarray,
    horizons: Sequence[float],
    calibration_bins: int = 10,
) -> Dict[str, float]:
    horizons_arr = np.asarray(horizons, dtype=float)
    survival_arr = np.asarray(survival, dtype=float)
    train_times_arr = np.asarray(train_times, dtype=float)
    train_events_arr = np.asarray(train_events, dtype=int)

    # Avoid unstable IPCW estimates beyond the supported follow-up range.
    censor_km = KaplanMeier.fit(train_times_arr, 1 - train_events_arr)
    censor_survival = censor_km.predict(horizons_arr)
    upper_follow_up = float(np.quantile(train_times_arr, 0.90))
    supported = (horizons_arr <= upper_follow_up) & (censor_survival >= 0.05)
    supported_indices = np.flatnonzero(supported)
    if supported_indices.size == 0:
        supported_indices = np.arange(max(1, min(len(horizons_arr) - 1, len(horizons_arr))))
    supported_horizons = horizons_arr[supported_indices]
    supported_survival = survival_arr[:, supported_indices]

    risk = risk_from_survival(supported_survival, supported_horizons)
    metrics: Dict[str, float] = {
        "c_index": harrell_c_index(test_times, test_events, risk),
        "ibs": integrated_brier_score(
            train_times,
            train_events,
            test_times,
            test_events,
            supported_survival,
            supported_horizons,
        ),
        "evaluation_horizon_max": float(supported_horizons[-1]),
    }
    for local_idx, horizon in enumerate(supported_horizons):
        metrics[f"auc_{float(horizon):.4g}"] = cumulative_dynamic_auc(
            train_times,
            train_events,
            test_times,
            test_events,
            1.0 - supported_survival[:, local_idx],
            float(horizon),
        )
        metrics[f"calibration_error_{float(horizon):.4g}"] = calibration_error(
            test_times,
            test_events,
            supported_survival[:, local_idx],
            float(horizon),
            n_bins=calibration_bins,
        )
    auc_values = [
        float(value)
        for key, value in metrics.items()
        if key.startswith("auc_") and np.isfinite(value)
    ]
    metrics["mean_time_dependent_auc"] = (
        float(np.mean(auc_values)) if auc_values else float("nan")
    )
    return metrics
