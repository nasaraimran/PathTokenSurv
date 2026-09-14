from pathlib import Path
import json

import numpy as np
import pandas as pd

from pathtokensurv.missingness_aggregation import aggregate_missingness_robustness


def _make_fold(root: Path, fold: int, offset: float = 0.0) -> Path:
    d = root / f"fold_{fold}"
    d.mkdir(parents=True)
    (d / "experiment_manifest.json").write_text(json.dumps({"outer_fold": fold}))
    natural = pd.DataFrame(
        [
            {
                "group_type": "exact_pattern",
                "group": "1111",
                "pattern_name": "clinical + mrna + mirna + cnv",
                "patients": 80,
                "events": 20,
                "c_index": 0.70 + offset,
                "ibs": 0.12 + offset,
                "mean_time_dependent_auc": 0.75 + offset,
                "cancer_stratified_c_index": 0.65 + offset,
                "within_cancer_comparable_pairs": 100,
            },
            {
                "group_type": "exact_pattern",
                "group": "1001",
                "pattern_name": "clinical + cnv",
                "patients": 20,
                "events": 10,
                "c_index": 0.68 + offset,
                "ibs": 0.15 + offset,
                "mean_time_dependent_auc": 0.72 + offset,
                "cancer_stratified_c_index": 0.60 + offset,
                "within_cancer_comparable_pairs": 20,
            },
            {
                "group_type": "aggregate",
                "group": "complete",
                "pattern_name": "complete",
                "patients": 80,
                "events": 20,
                "c_index": 0.70 + offset,
                "ibs": 0.12 + offset,
                "mean_time_dependent_auc": 0.75 + offset,
                "cancer_stratified_c_index": 0.65 + offset,
                "within_cancer_comparable_pairs": 100,
            },
            {
                "group_type": "aggregate",
                "group": "incomplete",
                "pattern_name": "incomplete",
                "patients": 20,
                "events": 10,
                "c_index": 0.68 + offset,
                "ibs": 0.15 + offset,
                "mean_time_dependent_auc": 0.72 + offset,
                "cancer_stratified_c_index": 0.60 + offset,
                "within_cancer_comparable_pairs": 20,
            },
        ]
    )
    natural.to_csv(d / "natural_missingness_metrics.csv", index=False)
    rows = []
    for condition, dc, ds, di, da in [
        ("all_observed", 0.00, 0.00, 0.00, 0.00),
        ("drop_mrna", -0.01, -0.02, 0.01, -0.01),
        ("drop_mirna", -0.005, -0.01, 0.005, -0.005),
        ("drop_cnv", -0.02, -0.03, 0.02, -0.02),
        ("clinical_only", -0.04, -0.05, 0.03, -0.04),
    ]:
        rows.append(
            {
                "condition": condition,
                "dropped_modalities": "none" if condition == "all_observed" else condition,
                "patients": 80,
                "events": 20,
                "c_index": 0.70 + offset + dc,
                "cancer_stratified_c_index": 0.65 + offset + ds,
                "ibs": 0.12 + offset + di,
                "mean_time_dependent_auc": 0.75 + offset + da,
            }
        )
    pd.DataFrame(rows).to_csv(d / "complete_case_modality_drop_metrics.csv", index=False)
    return d


def test_aggregate_missingness_robustness_outputs_and_signs(tmp_path):
    folds = [_make_fold(tmp_path, 1, 0.0), _make_fold(tmp_path, 2, 0.01)]
    out = tmp_path / "summary"
    result = aggregate_missingness_robustness(folds, out)
    assert (out / "missingness_robustness_publication_table.csv").exists()
    assert (out / "MISSINGNESS_ROBUSTNESS_SUMMARY.md").exists()
    drop = result["dropout_summary"].set_index("condition")
    assert np.isclose(drop.loc["drop_mrna", "degradation_c_index_mean"], 0.01)
    assert np.isclose(drop.loc["drop_mrna", "degradation_ibs_mean"], 0.01)
    assert np.isclose(drop.loc["all_observed", "degradation_c_index_mean"], 0.0)
    natural_gaps = result["natural_gap_summary"].set_index("contrast")
    assert np.isclose(
        natural_gaps.loc["complete_minus_incomplete_c_index", "mean"], 0.02
    )
    assert np.isclose(
        natural_gaps.loc["incomplete_minus_complete_ibs", "mean"], 0.03
    )


def test_controlled_all_observed_must_match_natural_complete(tmp_path):
    fold = _make_fold(tmp_path, 1, 0.0)
    drop = pd.read_csv(fold / "complete_case_modality_drop_metrics.csv")
    drop.loc[drop.condition == "all_observed", "c_index"] = 0.9
    drop.to_csv(fold / "complete_case_modality_drop_metrics.csv", index=False)
    try:
        aggregate_missingness_robustness([fold], tmp_path / "out")
    except ValueError as exc:
        assert "does not match natural complete" in str(exc)
    else:
        raise AssertionError("Expected a consistency-check failure")
