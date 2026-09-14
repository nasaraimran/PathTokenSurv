from __future__ import annotations

import numpy as np
import pandas as pd

from pathtokensurv.config import DataConfig
from pathtokensurv.data.raw import RawCohort
from pathtokensurv.data.qc import (
    patient_availability_table,
    modality_pattern_summary,
    cancer_specific_missingness,
    variance_diagnostics,
    annotation_validation,
    pathway_coverage_diagnostics,
)


def _raw():
    ids = np.asarray(["P1", "P2", "P3", "P4"])
    outcomes = pd.DataFrame({
        "time": [10, 20, 30, 40],
        "event": [1, 0, 1, 0],
        "cancer_type": ["A", "A", "B", "B"],
    }, index=ids)
    clinical = pd.DataFrame({
        "age": [50, 60, 70, 65],
        "sex": ["F", "M", "F", "M"],
        "race": ["R1", "R1", "R2", "R2"],
        "stage": ["Stage I", "Stage II", "Stage III", "Stage IV"],
        "histology": ["H1", "H1", "H2", "H2"],
    }, index=ids)
    mrna = pd.DataFrame({"7157": [0.0, 10.0, 20.0, np.nan], "1956": [1, 2, 3, np.nan]}, index=ids)
    mirna = pd.DataFrame({"hsa-miR-1": [1.0, np.nan, 3.0, 4.0], "hsa-miR-2": [2.0, np.nan, 4.0, 5.0]}, index=ids)
    cnv = pd.DataFrame({"TP53": [0.1, 0.2, np.nan, 0.4], "EGFR": [0.0, 0.3, np.nan, 0.6]}, index=ids)
    return RawCohort(ids, outcomes, clinical, {"mrna": mrna, "mirna": mirna, "cnv": cnv})


def test_availability_and_missingness_tables():
    raw = _raw()
    cfg = DataConfig(clinical_continuous=["age"], clinical_categorical=["sex", "race", "stage", "histology"])
    av = patient_availability_table(raw, cfg)
    assert len(av) == 4
    assert set(av["mask"]) == {"1111", "1101", "1110", "1011"}
    patterns = modality_pattern_summary(av)
    assert patterns["patients"].sum() == 4
    cm = cancer_specific_missingness(av)
    assert set(cm["cancer_type"]) == {"A", "B"}


def test_variance_and_annotation_qc():
    raw = _raw()
    cfg = DataConfig(
        variance_thresholds={"mrna": 5.0, "mirna": 0.0, "cnv": 0.0},
        min_features_per_modality=1,
        min_features_per_pathway=1,
        min_modalities_per_pathway=1,
        clinical_continuous=["age"], clinical_categorical=["sex", "race", "stage", "histology"],
    )
    summary, tables = variance_diagnostics(raw, cfg, [0, 1, 2])
    assert set(summary["modality"]) == {"mrna", "mirna", "cnv"}
    assert tables["mrna"].iloc[0]["variance"] >= tables["mrna"].iloc[-1]["variance"]

    mapping = {
        "pathways": ["P53", "EGFR"],
        "modality_features": {
            "mrna": {"P53": ["7157.0"], "EGFR": ["1956"]},
            "mirna": {"P53": ["HSA-MIR-1"], "EGFR": ["hsa-miR-2"]},
            "cnv": {"P53": ["tp53"], "EGFR": ["EGFR"]},
        },
        "adjacency": [[0, 1], [1, 0]],
    }
    selected = {m: tables[m].loc[tables[m]["selected"] == 1, "feature"].tolist() for m in tables}
    ann = annotation_validation(raw, mapping, selected)
    assert int(ann.loc[ann.modality == "mrna", "matched_dataset_features"].iloc[0]) == 2
    psummary, psizes, spec = pathway_coverage_diagnostics(mapping, selected, cfg)
    assert spec.num_pathways >= 1
    assert not psummary.empty
    assert not psizes.empty
