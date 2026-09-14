from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from pathtokensurv.config import DataConfig
from pathtokensurv.constants import MODALITIES, MOLECULAR_MODALITIES, MODALITY_TO_INDEX
from pathtokensurv.data.raw import RawCohort


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return numeric data without expensive per-column conversion when possible."""
    if all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        return frame
    return frame.apply(pd.to_numeric, errors="coerce")


@dataclass
class MolecularTransform:
    selected_features: List[str]
    medians: np.ndarray
    minimums: np.ndarray
    maximums: np.ndarray


@dataclass
class ClinicalTransform:
    continuous_medians: Dict[str, float]
    continuous_minimums: Dict[str, float]
    continuous_maximums: Dict[str, float]
    category_maps: Dict[str, Dict[str, int]]


@dataclass
class ProcessedCohort:
    patient_ids: np.ndarray
    molecular: Dict[str, torch.Tensor]
    clinical_continuous: torch.Tensor
    clinical_categorical: Dict[str, torch.Tensor]
    availability: torch.Tensor
    times: torch.Tensor
    events: torch.Tensor
    cancers: torch.Tensor
    cancer_labels: List[str]
    feature_names: Dict[str, List[str]]

    def __len__(self) -> int:
        return len(self.patient_ids)

    def subset(self, indices: Sequence[int]) -> "ProcessedCohort":
        idx_np = np.asarray(indices, dtype=int)
        idx_t = torch.as_tensor(idx_np, dtype=torch.long)
        return ProcessedCohort(
            patient_ids=self.patient_ids[idx_np],
            molecular={k: v.index_select(0, idx_t) for k, v in self.molecular.items()},
            clinical_continuous=self.clinical_continuous.index_select(0, idx_t),
            clinical_categorical={k: v.index_select(0, idx_t) for k, v in self.clinical_categorical.items()},
            availability=self.availability.index_select(0, idx_t),
            times=self.times.index_select(0, idx_t),
            events=self.events.index_select(0, idx_t),
            cancers=self.cancers.index_select(0, idx_t),
            cancer_labels=list(self.cancer_labels),
            feature_names={k: list(v) for k, v in self.feature_names.items()},
        )


@dataclass
class FoldPreprocessor:
    """Fit all transformations on one training partition only."""

    config: DataConfig
    clinical_transform: ClinicalTransform | None = None
    molecular_transforms: Dict[str, MolecularTransform] = field(default_factory=dict)
    cancer_map: Dict[str, int] = field(default_factory=dict)
    fitted: bool = False

    def fit(self, cohort: RawCohort, train_indices: Sequence[int]) -> "FoldPreprocessor":
        train_ids = cohort.patient_ids[np.asarray(train_indices, dtype=int)]
        train_clinical = cohort.clinical.loc[train_ids]

        continuous_medians: Dict[str, float] = {}
        continuous_minimums: Dict[str, float] = {}
        continuous_maximums: Dict[str, float] = {}
        for column in self.config.clinical_continuous:
            values = pd.to_numeric(train_clinical[column], errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median)
            continuous_medians[column] = median
            continuous_minimums[column] = float(filled.min())
            continuous_maximums[column] = float(filled.max())

        category_maps: Dict[str, Dict[str, int]] = {}
        for column in self.config.clinical_categorical:
            values = train_clinical[column].fillna("<MISSING>").astype(str)
            categories = sorted(values.unique().tolist())
            category_maps[column] = {value: idx + 1 for idx, value in enumerate(categories)}

        self.clinical_transform = ClinicalTransform(
            continuous_medians=continuous_medians,
            continuous_minimums=continuous_minimums,
            continuous_maximums=continuous_maximums,
            category_maps=category_maps,
        )

        self.molecular_transforms = {}
        for modality in MOLECULAR_MODALITIES:
            frame = cohort.modalities[modality].loc[train_ids]
            numeric = _numeric_frame(frame)
            row_available = ~numeric.isna().all(axis=1)
            if not bool(row_available.any()):
                raise ValueError(f"No observed training rows for modality '{modality}'.")

            # v1.4: compute median-imputed variances in feature chunks. This avoids
            # creating a second full copy of very large TCGA matrices during fold
            # preprocessing while preserving the exact selection rule.
            chunk_size = max(1, int(self.config.preprocessing_feature_chunk_size))
            median_parts = []
            variance_parts = []
            columns = list(numeric.columns)
            for start in range(0, len(columns), chunk_size):
                chunk_cols = columns[start:start + chunk_size]
                block = numeric.loc[row_available, chunk_cols]
                block_medians = block.median(axis=0).fillna(0.0)
                block_imputed = block.fillna(block_medians)
                median_parts.append(block_medians)
                variance_parts.append(block_imputed.var(axis=0, ddof=0))
            medians_series = pd.concat(median_parts).reindex(columns).fillna(0.0)
            variances = pd.concat(variance_parts).reindex(columns).fillna(0.0)

            threshold = float(self.config.variance_thresholds.get(modality, 0.0))
            selected = variances[variances > threshold].sort_values(ascending=False).index.tolist()
            minimum_count = min(self.config.min_features_per_modality, len(columns))
            if len(selected) < minimum_count:
                selected = variances.sort_values(ascending=False).head(minimum_count).index.tolist()
            if not selected:
                raise ValueError(f"Feature filtering removed every feature from '{modality}'.")

            selected_frame = numeric.loc[row_available, selected]
            selected_imputed = selected_frame.fillna(medians_series[selected])
            medians = medians_series[selected].to_numpy(dtype=np.float32, copy=True)
            minimums = selected_imputed.min(axis=0).to_numpy(dtype=np.float32, copy=True)
            maximums = selected_imputed.max(axis=0).to_numpy(dtype=np.float32, copy=True)
            self.molecular_transforms[modality] = MolecularTransform(
                selected_features=[str(x) for x in selected],
                medians=np.ascontiguousarray(medians),
                minimums=np.ascontiguousarray(minimums),
                maximums=np.ascontiguousarray(maximums),
            )

        cancer_values = cohort.outcomes.loc[train_ids, self.config.cancer_col].fillna("UNKNOWN").astype(str)
        cancers = sorted(cancer_values.unique().tolist())
        self.cancer_map = {value: idx + 1 for idx, value in enumerate(cancers)}
        self.fitted = True
        return self

    @property
    def category_cardinalities(self) -> Dict[str, int]:
        self._check_fitted()
        assert self.clinical_transform is not None
        return {
            column: len(mapping) + 1
            for column, mapping in self.clinical_transform.category_maps.items()
        }

    @property
    def num_cancers(self) -> int:
        self._check_fitted()
        return len(self.cancer_map) + 1

    def transform(self, cohort: RawCohort, indices: Sequence[int]) -> ProcessedCohort:
        self._check_fitted()
        assert self.clinical_transform is not None

        idx_np = np.asarray(indices, dtype=int)
        ids = cohort.patient_ids[idx_np]
        clinical = cohort.clinical.loc[ids]
        outcomes = cohort.outcomes.loc[ids]

        continuous_cols = []
        for column in self.config.clinical_continuous:
            values = pd.to_numeric(clinical[column], errors="coerce")
            values = values.fillna(self.clinical_transform.continuous_medians[column])
            lower = self.clinical_transform.continuous_minimums[column]
            upper = self.clinical_transform.continuous_maximums[column]
            scaled = (values.to_numpy(dtype=np.float32) - lower) / (upper - lower + 1e-8)
            if self.config.clip_scaled_values:
                scaled = np.clip(scaled, 0.0, 1.0)
            continuous_cols.append(scaled)
        continuous = np.stack(continuous_cols, axis=1).astype(np.float32)

        categorical: Dict[str, torch.Tensor] = {}
        for column in self.config.clinical_categorical:
            mapping = self.clinical_transform.category_maps[column]
            values = clinical[column].fillna("<MISSING>").astype(str)
            # pandas may return a read-only NumPy view when Copy-on-Write is enabled.
            # PyTorch requires writable arrays for safe zero-copy conversion.
            encoded = values.map(lambda value: mapping.get(value, 0)).to_numpy(
                dtype=np.int64, copy=True
            )
            categorical[column] = torch.from_numpy(np.ascontiguousarray(encoded))

        availability = np.zeros((len(ids), len(MODALITIES)), dtype=np.bool_)
        availability[:, MODALITY_TO_INDEX["clinical"]] = True
        molecular: Dict[str, torch.Tensor] = {}
        feature_names: Dict[str, List[str]] = {}

        for modality in MOLECULAR_MODALITIES:
            transform = self.molecular_transforms[modality]
            frame = cohort.modalities[modality].loc[ids]
            numeric = _numeric_frame(frame.reindex(columns=transform.selected_features))
            row_available = ~numeric.isna().all(axis=1)
            row_available_np = row_available.to_numpy(dtype=np.bool_, copy=True)
            availability[:, MODALITY_TO_INDEX[modality]] = row_available_np
            # copy=True is required because we impute missing cells in-place below.
            array = numeric.to_numpy(dtype=np.float32, copy=True)
            missing_cells = np.isnan(array)
            if missing_cells.any():
                array[missing_cells] = np.take(transform.medians, np.where(missing_cells)[1])
            scaled = (array - transform.minimums) / (transform.maximums - transform.minimums + 1e-8)
            if self.config.clip_scaled_values:
                scaled = np.clip(scaled, 0.0, 1.0)
            scaled[~row_available_np, :] = 0.0
            scaled = np.ascontiguousarray(scaled, dtype=np.float32)
            molecular[modality] = torch.from_numpy(scaled)
            feature_names[modality] = list(transform.selected_features)

        if not availability.any(axis=1).all():
            raise ValueError("Every patient must have at least one available modality.")

        time_series = (
            outcomes[self.config.time_col]
            if self.config.time_col in outcomes.columns
            else pd.Series(np.nan, index=outcomes.index)
        )
        event_series = (
            outcomes[self.config.event_col]
            if self.config.event_col in outcomes.columns
            else pd.Series(0, index=outcomes.index)
        )
        cancer_series = (
            outcomes[self.config.cancer_col]
            if self.config.cancer_col in outcomes.columns
            else pd.Series("UNKNOWN", index=outcomes.index)
        )
        time_values = pd.to_numeric(time_series, errors="coerce").to_numpy(
            dtype=np.float32, copy=True
        )
        event_values = pd.to_numeric(event_series, errors="coerce").fillna(0).to_numpy(
            dtype=np.float32, copy=True
        )
        cancer_values = cancer_series.fillna("UNKNOWN").astype(str)
        cancer_ids = cancer_values.map(lambda value: self.cancer_map.get(value, 0)).to_numpy(
            dtype=np.int64, copy=True
        )

        cancer_labels = ["UNKNOWN"] + [name for name, _ in sorted(self.cancer_map.items(), key=lambda item: item[1])]
        return ProcessedCohort(
            patient_ids=np.asarray(ids.astype(str)),
            molecular=molecular,
            clinical_continuous=torch.from_numpy(np.ascontiguousarray(continuous)),
            clinical_categorical=categorical,
            availability=torch.from_numpy(np.ascontiguousarray(availability)),
            times=torch.from_numpy(np.ascontiguousarray(time_values)),
            events=torch.from_numpy(np.ascontiguousarray(event_values)),
            cancers=torch.from_numpy(np.ascontiguousarray(cancer_ids)),
            cancer_labels=cancer_labels,
            feature_names=feature_names,
        )

    def fit_transform(
        self,
        cohort: RawCohort,
        train_indices: Sequence[int],
        transform_indices: Sequence[int] | None = None,
    ) -> ProcessedCohort:
        self.fit(cohort, train_indices)
        return self.transform(cohort, train_indices if transform_indices is None else transform_indices)

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "FoldPreprocessor":
        with Path(path).open("rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, found {type(obj).__name__}.")
        return obj

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("FoldPreprocessor.fit must be called before this operation.")
