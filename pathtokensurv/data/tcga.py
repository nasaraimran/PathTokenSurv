from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from pathtokensurv.config import DataConfig
from pathtokensurv.constants import MOLECULAR_MODALITIES


_MISSING_TEXT = {
    "", "na", "nan", "n/a", "none", "null", "unknown", "not available",
    "not evaluated", "not reported", "[not available]", "[not evaluated]",
    "[unknown]", "[discrepancy]", "--", "[completed]",
}


def normalize_tcga_patient_barcode(value: object, length: int = 12) -> str:
    """Return the patient-level TCGA barcode (normally the first 12 characters)."""
    text = str(value).strip().upper()
    if text in {"", "NAN", "NONE"}:
        return ""
    return text[:length]


def tcga_sample_type_code(value: object) -> str | None:
    """Extract the two-digit TCGA sample-type code from a sample barcode."""
    text = str(value).strip().upper()
    parts = text.split("-")
    if len(parts) < 4:
        return None
    match = re.match(r"(\d{2})", parts[3])
    return match.group(1) if match else None


def _clean_category(value: object) -> str:
    if pd.isna(value):
        return "<MISSING>"
    text = str(value).strip()
    if text.lower() in _MISSING_TEXT:
        return "<MISSING>"
    return text


def harmonize_stage(pathologic: object, clinical: object) -> str:
    """Collapse TCGA pathologic/clinical stage to Stage 0/I/II/III/IV when possible.

    Pathologic stage is preferred. Clinical stage is used only when pathologic
    stage is unavailable. Uninterpretable values are returned as ``<MISSING>``.
    """
    raw = _clean_category(pathologic)
    if raw == "<MISSING>":
        raw = _clean_category(clinical)
    if raw == "<MISSING>":
        return raw

    s = raw.upper().replace("STAGE", "").strip()
    s = s.replace(" ", "")
    # Common TCGA variants: IIA, IIIB, IVA, 1, 2A, IVB, etc.
    if re.match(r"^0", s):
        return "Stage 0"
    if re.match(r"^IV", s) or re.match(r"^4", s):
        return "Stage IV"
    if re.match(r"^III", s) or re.match(r"^3", s):
        return "Stage III"
    if re.match(r"^II", s) or re.match(r"^2", s):
        return "Stage II"
    if re.match(r"^I", s) or re.match(r"^1", s):
        return "Stage I"
    return "<MISSING>"


def _normalize_sex(value: object) -> str:
    text = _clean_category(value)
    if text == "<MISSING>":
        return text
    key = text.strip().lower()
    if key in {"female", "f"}:
        return "Female"
    if key in {"male", "m"}:
        return "Male"
    return text.title()


def _normalize_race(value: object) -> str:
    text = _clean_category(value)
    if text == "<MISSING>":
        return text
    return " ".join(word.capitalize() for word in text.replace("_", " ").split())


def load_tcga_clinical_tables(
    cfg: DataConfig,
    require_outcomes: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read TCGA-CDR and return canonical patient-level clinical/outcome tables."""
    path = Path(cfg.data_dir) / cfg.clinical_file
    if not path.exists():
        raise FileNotFoundError(f"TCGA clinical workbook not found: {path}")

    frame = pd.read_excel(path, sheet_name=cfg.clinical_sheet, engine="openpyxl")
    required = [
        cfg.tcga_clinical_patient_id_col,
        cfg.tcga_clinical_cancer_col,
        cfg.tcga_age_col,
        cfg.tcga_sex_col,
        cfg.tcga_race_col,
        cfg.tcga_pathologic_stage_col,
        cfg.tcga_clinical_stage_col,
        cfg.tcga_histology_col,
    ]
    if require_outcomes:
        required += [cfg.tcga_clinical_time_col, cfg.tcga_clinical_event_col]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"TCGA clinical sheet '{cfg.clinical_sheet}' is missing columns: {missing}")

    patient_ids = frame[cfg.tcga_clinical_patient_id_col].map(
        lambda x: normalize_tcga_patient_barcode(x, cfg.tcga_patient_barcode_length)
    )
    if (patient_ids == "").any():
        frame = frame.loc[patient_ids != ""].copy()
        patient_ids = patient_ids.loc[patient_ids != ""]
    frame = frame.copy()
    frame["__patient_id"] = patient_ids.values
    if frame["__patient_id"].duplicated().any():
        dup = frame.loc[frame["__patient_id"].duplicated(), "__patient_id"].head().tolist()
        raise ValueError(f"Duplicate patient barcodes in TCGA-CDR: {dup}")
    frame = frame.set_index("__patient_id")

    stage = [
        harmonize_stage(p, c)
        for p, c in zip(
            frame[cfg.tcga_pathologic_stage_col],
            frame[cfg.tcga_clinical_stage_col],
        )
    ]
    clinical = pd.DataFrame(index=frame.index.copy())
    clinical["age"] = pd.to_numeric(frame[cfg.tcga_age_col], errors="coerce")
    clinical["sex"] = frame[cfg.tcga_sex_col].map(_normalize_sex)
    clinical["race"] = frame[cfg.tcga_race_col].map(_normalize_race)
    clinical["stage"] = stage
    clinical["histology"] = frame[cfg.tcga_histology_col].map(_clean_category)

    cancer = frame[cfg.tcga_clinical_cancer_col].map(_clean_category)
    outcomes = pd.DataFrame(index=frame.index.copy())
    outcomes[cfg.cancer_col] = cancer
    if require_outcomes:
        outcomes[cfg.time_col] = pd.to_numeric(frame[cfg.tcga_clinical_time_col], errors="coerce")
        outcomes[cfg.event_col] = pd.to_numeric(frame[cfg.tcga_clinical_event_col], errors="coerce")
    else:
        outcomes[cfg.time_col] = np.nan
        outcomes[cfg.event_col] = np.nan

    qc = {
        "clinical_file": str(path),
        "clinical_sheet": cfg.clinical_sheet,
        "clinical_rows": int(len(frame)),
        "clinical_unique_patients": int(frame.index.nunique()),
        "cancer_types": int(cancer[cancer != "<MISSING>"].nunique()),
    }

    if require_outcomes:
        time = pd.to_numeric(outcomes[cfg.time_col], errors="coerce")
        event = pd.to_numeric(outcomes[cfg.event_col], errors="coerce")
        valid_time = time.notna()
        if cfg.tcga_exclude_zero_survival_time:
            valid_time &= time > 0
        else:
            valid_time &= time >= 0
        valid = valid_time & event.isin([0, 1]) & (outcomes[cfg.cancer_col] != "<MISSING>")
        qc.update({
            "missing_survival_time": int(time.isna().sum()),
            "zero_survival_time": int((time == 0).sum()),
            "invalid_event": int((~event.isin([0, 1])).sum()),
            "eligible_patients": int(valid.sum()),
            "events": int((event.loc[valid] == 1).sum()),
            "censored": int((event.loc[valid] == 0).sum()),
        })
        outcomes = outcomes.loc[valid].copy()
        clinical = clinical.loc[outcomes.index].copy()

    return outcomes, clinical, qc


def _resolve_feature_column(columns: Sequence[str], requested: str | None) -> str:
    columns = list(map(str, columns))
    if requested and requested in columns:
        return requested
    if requested:
        lower = {c.lower(): c for c in columns}
        if requested.lower() in lower:
            return lower[requested.lower()]
    if not columns:
        raise ValueError("Molecular file has no columns.")
    return columns[0]


def _read_tcga_molecular_matrix(
    path: Path,
    modality: str,
    cfg: DataConfig,
    target_patients: Sequence[str],
) -> tuple[pd.DataFrame, dict]:
    """Read a native features-by-samples TCGA matrix as patients-by-features.

    Only columns whose sample type is allowed and whose patient barcode occurs in
    the target clinical cohort are loaded. Numeric columns are read as float32 by
    default to limit memory use for the large mRNA matrix.
    """
    if not path.exists():
        raise FileNotFoundError(f"TCGA {modality} file not found: {path}")

    header = pd.read_csv(path, sep=cfg.delimiter, nrows=0)
    feature_col = _resolve_feature_column(
        header.columns,
        cfg.tcga_molecular_feature_columns.get(modality),
    )
    target_set = set(map(str, target_patients))
    # The configured list is also the preference order. This matters in pan-cancer
    # data because solid tumors commonly use 01, blood cancers can use 03/09, and
    # some TCGA cohorts (notably SKCM) can contain only metastatic tumor samples 06.
    sample_type_priority = [str(x).zfill(2) for x in cfg.tcga_allowed_sample_types]
    allowed_types = set(sample_type_priority)

    candidates: Dict[str, list[tuple[str, str | None]]] = {}
    sample_type_counts: Dict[str, int] = {}
    for col in map(str, header.columns):
        if col == feature_col:
            continue
        patient = normalize_tcga_patient_barcode(col, cfg.tcga_patient_barcode_length)
        sample_type = tcga_sample_type_code(col)
        if sample_type is not None:
            sample_type_counts[sample_type] = sample_type_counts.get(sample_type, 0) + 1
        if patient not in target_set:
            continue
        if allowed_types and sample_type not in allowed_types:
            continue
        candidates.setdefault(patient, []).append((col, sample_type))

    sample_cols: list[str] = []
    patient_for_sample: dict[str, str] = {}
    selected_sample_type_counts: Dict[str, int] = {}
    for patient in target_patients:
        choices = candidates.get(str(patient), [])
        if not choices:
            continue
        selected = []
        for preferred_type in sample_type_priority:
            selected = [(col, typ) for col, typ in choices if typ == preferred_type]
            if selected:
                break
        if not selected and not sample_type_priority:
            selected = choices
        for col, typ in selected:
            sample_cols.append(col)
            patient_for_sample[col] = str(patient)
            key = typ or "UNKNOWN"
            selected_sample_type_counts[key] = selected_sample_type_counts.get(key, 0) + 1

    if not sample_cols:
        raise ValueError(
            f"No usable {modality} sample columns were found in {path}. "
            f"Allowed sample types in priority order={sample_type_priority} and "
            f"target clinical patients={len(target_set)}."
        )

    dtype_name = str(cfg.tcga_raw_float_dtype).lower()
    dtype = np.float32 if dtype_name == "float32" else np.float64
    dtype_map = {c: dtype for c in sample_cols}
    frame = pd.read_csv(
        path,
        sep=cfg.delimiter,
        usecols=[feature_col] + sample_cols,
        dtype=dtype_map,
        low_memory=False,
    )
    features = frame[feature_col].astype(str).str.strip()
    numeric = frame.drop(columns=[feature_col])
    numeric.index = features
    duplicate_features = int(numeric.index.duplicated().sum())
    if duplicate_features:
        strategy = cfg.tcga_duplicate_feature_strategy.lower()
        if strategy == "mean":
            numeric = numeric.groupby(level=0, sort=False).mean()
        elif strategy == "first":
            numeric = numeric.loc[~numeric.index.duplicated(keep="first")]
        else:
            raise ValueError("tcga_duplicate_feature_strategy must be 'mean' or 'first'.")

    # Convert samples to patients. Multiple primary-tumor aliquots can occur for
    # one patient; average them by default so that a patient appears only once.
    sample_by_feature = numeric.T
    sample_by_feature.insert(
        0,
        "__patient_id",
        [patient_for_sample[str(sample)] for sample in sample_by_feature.index],
    )
    counts = sample_by_feature["__patient_id"].value_counts()
    duplicate_patient_samples = int((counts - 1).clip(lower=0).sum())
    strategy = cfg.tcga_duplicate_patient_strategy.lower()
    if strategy == "mean":
        patient_by_feature = sample_by_feature.groupby("__patient_id", sort=False).mean(numeric_only=True)
    elif strategy == "first":
        patient_by_feature = sample_by_feature.drop_duplicates("__patient_id", keep="first").set_index("__patient_id")
    else:
        raise ValueError("tcga_duplicate_patient_strategy must be 'mean' or 'first'.")

    patient_by_feature = patient_by_feature.astype(dtype, copy=False)
    patient_by_feature = patient_by_feature.reindex(list(map(str, target_patients)))

    qc = {
        "file": str(path),
        "modality": modality,
        "raw_sample_columns": int(len(header.columns) - 1),
        "loaded_sample_columns": int(len(sample_cols)),
        "unique_loaded_patients": int(counts.index.nunique()),
        "duplicate_patient_samples_collapsed": duplicate_patient_samples,
        "raw_feature_rows": int(len(frame)),
        "final_unique_features": int(patient_by_feature.shape[1]),
        "duplicate_feature_rows_collapsed": duplicate_features,
        "matched_patients_with_modality": int(patient_by_feature.notna().any(axis=1).sum()),
        "allowed_sample_types_in_priority_order": sample_type_priority,
        "sample_type_counts_in_header": sample_type_counts,
        "selected_sample_type_counts": selected_sample_type_counts,
    }
    return patient_by_feature, qc


def load_tcga_pancancer_cohort(cfg: DataConfig, require_outcomes: bool = True):
    """Load the user's native PanCancer clinical and molecular files directly."""
    # Local import avoids a module cycle with raw.py.
    from pathtokensurv.data.raw import RawCohort

    outcomes, clinical, clinical_qc = load_tcga_clinical_tables(cfg, require_outcomes=require_outcomes)
    patient_ids = outcomes.index.astype(str).tolist()
    modalities: Dict[str, pd.DataFrame] = {}
    modality_qc: Dict[str, dict] = {}
    for modality in MOLECULAR_MODALITIES:
        filename = cfg.modality_files.get(modality)
        if not filename:
            raise ValueError(f"No raw file configured for modality '{modality}'.")
        frame, qc = _read_tcga_molecular_matrix(
            Path(cfg.data_dir) / filename,
            modality,
            cfg,
            patient_ids,
        )
        modalities[modality] = frame
        modality_qc[modality] = qc

    metadata = {
        "source_format": "tcga_pancancer_raw",
        "clinical_qc": clinical_qc,
        "modality_qc": modality_qc,
        "final_patient_count": int(len(patient_ids)),
    }
    return RawCohort(
        patient_ids=np.asarray(patient_ids, dtype=str),
        outcomes=outcomes,
        clinical=clinical,
        modalities=modalities,
        metadata=metadata,
    )
