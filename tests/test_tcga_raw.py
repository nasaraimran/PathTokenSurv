from __future__ import annotations

import numpy as np
import pandas as pd

from pathtokensurv.config import DataConfig
from pathtokensurv.data.raw import load_raw_cohort
from pathtokensurv.data.tcga import harmonize_stage, normalize_tcga_patient_barcode, tcga_sample_type_code


def _write_matrix(path, feature_col, features, samples, values):
    frame = pd.DataFrame(values, columns=samples)
    frame.insert(0, feature_col, features)
    frame.to_csv(path, sep="\t", index=False)


def test_tcga_barcode_and_stage_helpers():
    assert normalize_tcga_patient_barcode("TCGA-OR-A5J1-01A") == "TCGA-OR-A5J1"
    assert tcga_sample_type_code("TCGA-OR-A5J1-01A") == "01"
    assert harmonize_stage("Stage IIIB", "") == "Stage III"
    assert harmonize_stage("[Not Available]", "Stage IIA") == "Stage II"


def test_native_tcga_loader(tmp_path):
    clinical = pd.DataFrame({
        "bcr_patient_barcode": ["TCGA-AA-0001", "TCGA-AA-0002", "TCGA-AA-0003", "TCGA-AA-0004"],
        "type": ["BRCA", "BRCA", "LUAD", "LUAD"],
        "age_at_initial_pathologic_diagnosis": [50, 60, 70, 55],
        "gender": ["FEMALE", "MALE", "FEMALE", "MALE"],
        "race": ["WHITE", "BLACK OR AFRICAN AMERICAN", "WHITE", "ASIAN"],
        "ajcc_pathologic_tumor_stage": ["Stage IIA", "[Not Available]", "Stage IV", "Stage I"],
        "clinical_stage": ["", "Stage IIIB", "", ""],
        "histological_type": ["H1", "H2", "H3", "H4"],
        "OS": [1, 0, 1, 0],
        "OS.time": [100, 200, 0, 400],
    })
    with pd.ExcelWriter(tmp_path / "PanCancer_Clinical.xlsx", engine="openpyxl") as writer:
        clinical.to_excel(writer, sheet_name="TCGA-CDR", index=False)

    samples = ["TCGA-AA-0001-01A", "TCGA-AA-0001-01B", "TCGA-AA-0002-01A", "TCGA-AA-0003-01A", "TCGA-AA-0004-11A"]
    _write_matrix(tmp_path/"PanCancer_mRNA.txt", "sample", ["7157", "1956"], samples,
                  [[1, 3, 5, 7, 99], [2, 4, 6, 8, 99]])
    _write_matrix(tmp_path/"PanCancer_miRNA.txt", "sample", ["hsa-let-7a-5p", "hsa-miR-1-3p"], samples,
                  [[2, 4, 6, 8, 99], [3, 5, 7, 9, 99]])
    _write_matrix(tmp_path/"PanCancer_CNV.txt", "Sample", ["TP53", "EGFR"], samples,
                  [[-1, 1, 0, 0.5, 99], [0, 2, 1, 0.2, 99]])

    cfg = DataConfig(
        data_dir=str(tmp_path), source_format="tcga_pancancer_raw",
        clinical_file="PanCancer_Clinical.xlsx",
        modality_files={"mrna":"PanCancer_mRNA.txt", "mirna":"PanCancer_miRNA.txt", "cnv":"PanCancer_CNV.txt"},
        clinical_continuous=["age"], clinical_categorical=["sex","race","stage","histology"],
    )
    raw = load_raw_cohort(cfg, require_outcomes=True)
    # Patient 3 has zero OS.time and patient 4 only has a normal (11) molecular sample.
    assert raw.patient_ids.tolist() == ["TCGA-AA-0001", "TCGA-AA-0002", "TCGA-AA-0004"]
    assert raw.clinical.loc["TCGA-AA-0002", "stage"] == "Stage III"
    # Duplicate primary-tumor samples for patient 1 are averaged: (1 + 3) / 2 = 2.
    assert np.isclose(raw.modalities["mrna"].loc["TCGA-AA-0001", "7157"], 2.0)
    # Normal sample type 11 is ignored, leaving patient 4 unavailable for mRNA.
    assert raw.modalities["mrna"].loc["TCGA-AA-0004"].isna().all()
    assert raw.metadata["clinical_qc"]["zero_survival_time"] == 1
