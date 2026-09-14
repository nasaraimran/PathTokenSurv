import numpy as np
import pandas as pd

from pathtokensurv.metrics import stratified_harrell_c_index
from pathtokensurv.scientific_evaluation import summarize_ipcw_calibration


def test_stratified_c_excludes_cross_cancer_pairs():
    # Cancer A is uniformly lower risk than cancer B, so pooled concordance is
    # driven by between-cancer separation. Within each cancer, risk ties imply C=0.5.
    times = np.array([1.0, 2.0, 10.0, 20.0])
    events = np.array([1, 1, 1, 1])
    risk = np.array([2.0, 2.0, 1.0, 1.0])
    cancer = np.array([0, 0, 1, 1])
    result = stratified_harrell_c_index(times, events, risk, cancer)
    assert np.isclose(result["c_index"], 0.5)
    assert result["within_strata_comparable_pairs"] == 2
    assert result["pooled_comparable_pairs"] == 6
    assert np.isclose(result["within_strata_pair_fraction"], 1 / 3)


def test_ipcw_calibration_summary_is_weighted_by_bin_mass():
    table = pd.DataFrame(
        {
            "horizon": [10.0, 10.0, 20.0, 20.0],
            "bin": [1, 2, 1, 2],
            "patients": [5, 5, 5, 5],
            "weight_sum": [1.0, 3.0, 2.0, 2.0],
            "mean_predicted_survival": [0.2, 0.8, 0.3, 0.7],
            "ipcw_observed_survival": [0.1, 0.7, 0.2, 0.5],
        }
    )
    per_horizon, summary = summarize_ipcw_calibration(table)
    first = per_horizon.loc[per_horizon.horizon == 10.0, "ipcw_ece"].iloc[0]
    second = per_horizon.loc[per_horizon.horizon == 20.0, "ipcw_ece"].iloc[0]
    assert np.isclose(first, 0.1)
    assert np.isclose(second, 0.15)
    assert np.isclose(summary["mean_ipcw_ece"], 0.125)
    assert np.isclose(summary["integrated_ipcw_ece"], 0.125)
