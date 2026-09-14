from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from pathtokensurv.config import DataConfig
from pathtokensurv.constants import MOLECULAR_MODALITIES


@dataclass
class RawCohort:
    """Aligned patient-level tables before fold-specific preprocessing."""

    patient_ids: np.ndarray
    outcomes: pd.DataFrame
    clinical: pd.DataFrame
    modalities: Dict[str, pd.DataFrame]
    metadata: Dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.patient_ids)

    def subset(self, indices: Iterable[int]) -> "RawCohort":
        idx = np.asarray(list(indices), dtype=int)
        ids = self.patient_ids[idx]
        return RawCohort(
            patient_ids=ids.copy(),
            outcomes=self.outcomes.loc[ids].copy(),
            clinical=self.clinical.loc[ids].copy(),
            modalities={name: frame.loc[ids].copy() for name, frame in self.modalities.items()},
            metadata=dict(self.metadata),
        )


def _read_indexed(path: Path, cfg: DataConfig) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    frame = pd.read_csv(path, sep=cfg.delimiter)
    if cfg.patient_id_col not in frame.columns:
        raise ValueError(f"{path} must contain a '{cfg.patient_id_col}' column.")
    frame[cfg.patient_id_col] = frame[cfg.patient_id_col].astype(str)
    if frame[cfg.patient_id_col].duplicated().any():
        duplicates = frame.loc[frame[cfg.patient_id_col].duplicated(), cfg.patient_id_col].head().tolist()
        raise ValueError(f"Duplicate patient IDs in {path}: {duplicates}")
    return frame.set_index(cfg.patient_id_col)


def load_raw_cohort(cfg: DataConfig, require_outcomes: bool = True) -> RawCohort:
    if cfg.source_format.lower() in {"tcga_pancancer_raw", "tcga_raw", "pancancer_raw"}:
        from pathtokensurv.data.tcga import load_tcga_pancancer_cohort
        return load_tcga_pancancer_cohort(cfg, require_outcomes=require_outcomes)

    # Prepared patient-wide format. Molecular tables may omit a patient or
    # contain an all-NaN row when the modality is unavailable.
    root = Path(cfg.data_dir)
    clinical = _read_indexed(root / cfg.clinical_file, cfg)

    outcomes_path = root / cfg.outcomes_file
    if outcomes_path.exists():
        outcomes = _read_indexed(outcomes_path, cfg)
    elif require_outcomes:
        raise FileNotFoundError(f"Required outcomes file not found: {outcomes_path}")
    else:
        outcomes = pd.DataFrame(index=clinical.index.copy())
        outcomes[cfg.time_col] = np.nan
        outcomes[cfg.event_col] = np.nan
        if cfg.cancer_col in clinical.columns:
            outcomes[cfg.cancer_col] = clinical[cfg.cancer_col]
        else:
            outcomes[cfg.cancer_col] = "UNKNOWN"

    required_outcome_cols = [cfg.time_col, cfg.event_col, cfg.cancer_col]
    if require_outcomes:
        missing = [col for col in required_outcome_cols if col not in outcomes.columns]
        if missing:
            raise ValueError(f"Outcomes table is missing columns: {missing}")

    patient_ids = outcomes.index.intersection(clinical.index, sort=False)
    if len(patient_ids) == 0:
        raise ValueError("No patient IDs are shared by the outcomes and clinical tables.")

    outcomes = outcomes.loc[patient_ids].copy()
    clinical = clinical.loc[patient_ids].copy()

    if require_outcomes:
        valid = (
            pd.to_numeric(outcomes[cfg.time_col], errors="coerce").notna()
            & pd.to_numeric(outcomes[cfg.event_col], errors="coerce").isin([0, 1])
            & (pd.to_numeric(outcomes[cfg.time_col], errors="coerce") > 0)
            & outcomes[cfg.cancer_col].notna()
        )
        outcomes = outcomes.loc[valid].copy()
        clinical = clinical.loc[outcomes.index].copy()
        patient_ids = outcomes.index

    modalities: Dict[str, pd.DataFrame] = {}
    for name in MOLECULAR_MODALITIES:
        filename = cfg.modality_files.get(name)
        if filename is None:
            raise ValueError(f"No file configured for modality '{name}'.")
        frame = _read_indexed(root / filename, cfg)
        modalities[name] = frame.reindex(patient_ids)

    return RawCohort(
        patient_ids=np.asarray(patient_ids.astype(str)),
        outcomes=outcomes,
        clinical=clinical,
        modalities=modalities,
        metadata={"source_format": "patient_wide"},
    )
