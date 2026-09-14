from __future__ import annotations

import numpy as np
import torch

from pathtokensurv.data.preprocessing import ProcessedCohort
from pathtokensurv.scientific_evaluation import evaluate_per_cancer, ipcw_calibration_table
from pathtokensurv.trainer import PredictionBundle


def _cohort() -> ProcessedCohort:
    return ProcessedCohort(
        patient_ids=np.array(["a", "b", "c", "d"]),
        molecular={
            "mrna": torch.zeros(4, 1),
            "mirna": torch.zeros(4, 1),
            "cnv": torch.zeros(4, 1),
        },
        clinical_continuous=torch.zeros(4, 1),
        clinical_categorical={},
        availability=torch.ones(4, 4, dtype=torch.bool),
        times=torch.tensor([5.0, 8.0, 6.0, 10.0]),
        events=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        cancers=torch.tensor([1, 1, 2, 2]),
        cancer_labels=["UNKNOWN", "A", "B"],
        feature_names={"mrna": ["G"], "mirna": ["M"], "cnv": ["G"]},
    )


def test_ipcw_calibration_table_has_valid_probabilities():
    train_t = np.array([4.0, 5.0, 7.0, 8.0, 9.0, 10.0])
    train_e = np.array([1, 0, 1, 0, 1, 0])
    test_t = np.array([5.0, 6.0, 9.0, 10.0])
    test_e = np.array([1, 0, 1, 0])
    surv = np.array([
        [0.8, 0.6, 0.4],
        [0.9, 0.7, 0.5],
        [0.95, 0.85, 0.7],
        [0.98, 0.9, 0.8],
    ])
    table = ipcw_calibration_table(train_t, train_e, test_t, test_e, surv, [4.0, 7.0, 9.0], n_bins=2)
    assert not table.empty
    assert table["mean_predicted_survival"].between(0, 1).all()
    assert table["ipcw_observed_survival"].between(0, 1).all()


def test_per_cancer_table_retains_all_test_cancers():
    train = _cohort()
    pred = PredictionBundle(
        patient_ids=["x", "y", "z", "w"],
        times=np.array([4.0, 9.0, 5.0, 11.0]),
        events=np.array([1, 0, 1, 0]),
        cancers=np.array([1, 1, 2, 2]),
        survival=np.array([
            [0.7, 0.4],
            [0.9, 0.8],
            [0.6, 0.3],
            [0.95, 0.85],
        ]),
        fused=np.zeros((4, 2)),
    )
    table = evaluate_per_cancer(train, pred, [5.0, 9.0], calibration_bins=2)
    assert set(table["cancer_type"]) == {"A", "B"}
    assert table["test_patients"].sum() == 4
