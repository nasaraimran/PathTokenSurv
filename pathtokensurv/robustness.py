from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pathtokensurv.constants import MODALITIES, MODALITY_TO_INDEX
from pathtokensurv.data.dataset import MultiModalSurvivalDataset
from pathtokensurv.data.raw import load_raw_cohort
from pathtokensurv.inference import load_inference_model
from pathtokensurv.metrics import evaluate_survival_predictions
from pathtokensurv.trainer import PredictionBundle
from pathtokensurv.utils.device import move_batch_to_device, resolve_device


def _slice_batch(batch: dict, keep: torch.Tensor) -> dict:
    indices = torch.nonzero(keep, as_tuple=False).squeeze(1)
    sliced = {}
    for key, value in batch.items():
        if key == "patient_id":
            sliced[key] = [value[i] for i in indices.tolist()]
        elif isinstance(value, dict):
            sliced[key] = {nested: tensor.index_select(0, indices) for nested, tensor in value.items()}
        else:
            sliced[key] = value.index_select(0, indices)
    return sliced


@torch.no_grad()
def predict_with_mask_function(
    model,
    loader: DataLoader,
    device: torch.device,
    mask_function: Callable[[torch.Tensor], torch.Tensor],
) -> PredictionBundle:
    model.eval()
    patient_ids: List[str] = []
    times: List[np.ndarray] = []
    events: List[np.ndarray] = []
    cancers: List[np.ndarray] = []
    survival: List[np.ndarray] = []
    fused: List[np.ndarray] = []

    for batch in loader:
        natural = batch["availability"].bool()
        effective = mask_function(natural.clone()).bool()
        valid = effective.any(dim=1)
        if not valid.any():
            continue
        if not valid.all():
            batch = _slice_batch(batch, valid)
            effective = effective[valid]
        patient_ids.extend(list(batch["patient_id"]))
        batch = move_batch_to_device(batch, device)
        effective = effective.to(device)
        output = model(batch, effective_mask=effective)
        times.append(batch["time"].cpu().numpy())
        events.append(batch["event"].cpu().numpy())
        cancers.append(batch["cancer"].cpu().numpy())
        survival.append(output["survival"].cpu().numpy())
        fused.append(output["fused"].cpu().numpy())

    if not survival:
        raise ValueError("The selected modality mask left no evaluable patients.")
    return PredictionBundle(
        patient_ids=patient_ids,
        times=np.concatenate(times),
        events=np.concatenate(events),
        cancers=np.concatenate(cancers),
        survival=np.concatenate(survival),
        fused=np.concatenate(fused),
    )


def evaluate_missingness(
    artifact_dir: str | Path,
    data_dir: str | Path,
    rates: Sequence[float],
    repeats: int = 5,
    seed: int = 123,
    device_name: str = "auto",
) -> pd.DataFrame:
    device = resolve_device(device_name)
    model, config, preprocessor, discretizer = load_inference_model(artifact_dir, device)
    config.data.data_dir = str(data_dir)
    raw = load_raw_cohort(config.data, require_outcomes=True)
    processed = preprocessor.transform(raw, np.arange(len(raw)))
    loader = DataLoader(
        MultiModalSurvivalDataset(processed),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    reference = np.load(Path(artifact_dir) / "training_reference.npz")

    rows = []
    for rate in rates:
        if not 0.0 <= rate < 1.0:
            raise ValueError("Missingness rates must lie in [0, 1).")
        for repeat in range(repeats):
            generator = torch.Generator().manual_seed(seed + repeat + int(rate * 10000))

            def mask_fn(natural: torch.Tensor, p=float(rate), gen=generator) -> torch.Tensor:
                keep = torch.rand(natural.shape, generator=gen) > p
                effective = natural & keep
                empty = ~effective.any(dim=1)
                # Clinical data are retained as the fallback because the study cohort requires it.
                effective[empty, MODALITY_TO_INDEX["clinical"]] = natural[
                    empty, MODALITY_TO_INDEX["clinical"]
                ]
                return effective

            predictions = predict_with_mask_function(model, loader, device, mask_fn)
            metrics = evaluate_survival_predictions(
                reference["times"],
                reference["events"],
                predictions.times,
                predictions.events,
                predictions.survival,
                discretizer.right_edges,
                calibration_bins=config.evaluation.calibration_bins,
            )
            rows.append({"missing_rate": rate, "repeat": repeat, **metrics})
    return pd.DataFrame(rows)


def evaluate_modality_subsets(
    artifact_dir: str | Path,
    data_dir: str | Path,
    include_all_nonempty: bool = True,
    device_name: str = "auto",
) -> pd.DataFrame:
    device = resolve_device(device_name)
    model, config, preprocessor, discretizer = load_inference_model(artifact_dir, device)
    config.data.data_dir = str(data_dir)
    raw = load_raw_cohort(config.data, require_outcomes=True)
    processed = preprocessor.transform(raw, np.arange(len(raw)))
    loader = DataLoader(
        MultiModalSurvivalDataset(processed),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    reference = np.load(Path(artifact_dir) / "training_reference.npz")

    subsets: List[tuple[str, ...]] = []
    if include_all_nonempty:
        for size in range(1, len(MODALITIES) + 1):
            subsets.extend(combinations(MODALITIES, size))
    rows = []
    for subset in subsets:
        allowed = torch.zeros(len(MODALITIES), dtype=torch.bool)
        for modality in subset:
            allowed[MODALITY_TO_INDEX[modality]] = True

        def mask_fn(natural: torch.Tensor, allowed_mask=allowed) -> torch.Tensor:
            return natural & allowed_mask.unsqueeze(0)

        try:
            predictions = predict_with_mask_function(model, loader, device, mask_fn)
        except ValueError:
            continue
        metrics = evaluate_survival_predictions(
            reference["times"],
            reference["events"],
            predictions.times,
            predictions.events,
            predictions.survival,
            discretizer.right_edges,
            calibration_bins=config.evaluation.calibration_bins,
        )
        rows.append(
            {
                "modalities": "+".join(subset),
                "n_patients": len(predictions.patient_ids),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _availability_mask_strings(availability: torch.Tensor) -> np.ndarray:
    arr = availability.detach().cpu().numpy().astype(bool)
    return np.asarray(["".join("1" if value else "0" for value in row) for row in arr], dtype=object)


def _availability_pattern_name(mask: str) -> str:
    names = [name for bit, name in zip(mask, MODALITIES) if bit == "1"]
    return " + ".join(names) if names else "none"


def _subset_prediction_bundle(bundle: PredictionBundle, indices: np.ndarray) -> PredictionBundle:
    idx = np.asarray(indices, dtype=int)
    return PredictionBundle(
        patient_ids=[bundle.patient_ids[int(i)] for i in idx],
        times=bundle.times[idx],
        events=bundle.events[idx],
        cancers=bundle.cancers[idx],
        survival=bundle.survival[idx],
        fused=bundle.fused[idx],
    )


def evaluate_natural_missingness_groups(
    train: ProcessedCohort,
    test: ProcessedCohort,
    predictions: PredictionBundle,
    horizons: Sequence[float],
    calibration_bins: int,
) -> pd.DataFrame:
    """Evaluate naturally observed TCGA modality-availability groups.

    Exact availability patterns plus complete/incomplete aggregate groups are
    evaluated on the held-out test set only. The pooled censoring reference is
    always the outer-fold training cohort. Cancer-stratified C is reported in
    parallel to expose within-cancer discrimination.
    """
    from pathtokensurv.scientific_evaluation import stratified_discrimination_summary

    if list(map(str, test.patient_ids)) != list(map(str, predictions.patient_ids)):
        raise ValueError("Prediction order does not match the processed test cohort.")
    patterns = _availability_mask_strings(test.availability)
    train_times = train.times.detach().cpu().numpy()
    train_events = train.events.detach().cpu().numpy()
    horizons_arr = np.asarray(horizons, dtype=float)

    groups: List[tuple[str, str, np.ndarray]] = []
    for mask in sorted(set(patterns.tolist())):
        groups.append(("exact_pattern", mask, np.flatnonzero(patterns == mask)))
    complete = np.flatnonzero(patterns == "1111")
    incomplete = np.flatnonzero(patterns != "1111")
    groups.extend([
        ("aggregate", "complete", complete),
        ("aggregate", "incomplete", incomplete),
    ])

    rows = []
    for group_type, label, indices in groups:
        if len(indices) == 0:
            continue
        subset = _subset_prediction_bundle(predictions, indices)
        row = {
            "group_type": group_type,
            "group": label,
            "pattern_name": (
                _availability_pattern_name(label) if group_type == "exact_pattern" else label
            ),
            "patients": int(len(indices)),
            "events": int(subset.events.sum()),
        }
        try:
            metrics = evaluate_survival_predictions(
                train_times,
                train_events,
                subset.times,
                subset.events,
                subset.survival,
                horizons_arr,
                calibration_bins=calibration_bins,
            )
            row.update(
                {
                    "c_index": metrics.get("c_index", float("nan")),
                    "ibs": metrics.get("ibs", float("nan")),
                    "mean_time_dependent_auc": metrics.get(
                        "mean_time_dependent_auc", float("nan")
                    ),
                }
            )
            stratified = stratified_discrimination_summary(train, subset, horizons_arr)
            row.update(
                {
                    "cancer_stratified_c_index": stratified["cancer_stratified_c_index"],
                    "within_cancer_comparable_pairs": stratified[
                        "within_cancer_comparable_pairs"
                    ],
                }
            )
        except (ValueError, ZeroDivisionError, FloatingPointError):
            row.update(
                {
                    "c_index": float("nan"),
                    "ibs": float("nan"),
                    "mean_time_dependent_auc": float("nan"),
                    "cancer_stratified_c_index": float("nan"),
                    "within_cancer_comparable_pairs": 0,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_complete_case_modality_dropouts(
    model,
    train: ProcessedCohort,
    test: ProcessedCohort,
    horizons: Sequence[float],
    calibration_bins: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> pd.DataFrame:
    """Deliberately remove molecular modalities from the same complete cases.

    Every condition uses the identical naturally complete held-out patients.
    Clinical data are retained because clinical availability is mandatory in the
    TCGA study cohort. This isolates test-time robustness to missing molecular
    sources without changing patient composition between conditions.
    """
    from pathtokensurv.scientific_evaluation import stratified_discrimination_summary

    complete = test.availability.bool().all(dim=1).detach().cpu().numpy()
    indices = np.flatnonzero(complete)
    if len(indices) < 2:
        return pd.DataFrame()
    cohort = test.subset(indices)
    loader = DataLoader(
        MultiModalSurvivalDataset(cohort),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    conditions = {
        "all_observed": (),
        "drop_mrna": ("mrna",),
        "drop_mirna": ("mirna",),
        "drop_cnv": ("cnv",),
        "clinical_only": ("mrna", "mirna", "cnv"),
    }
    train_times = train.times.detach().cpu().numpy()
    train_events = train.events.detach().cpu().numpy()
    horizons_arr = np.asarray(horizons, dtype=float)
    rows = []
    for condition, dropped in conditions.items():
        dropped_indices = [MODALITY_TO_INDEX[name] for name in dropped]

        def mask_fn(natural: torch.Tensor, positions=dropped_indices) -> torch.Tensor:
            effective = natural.clone()
            if positions:
                effective[:, positions] = False
            return effective

        predictions = predict_with_mask_function(model, loader, device, mask_fn)
        metrics = evaluate_survival_predictions(
            train_times,
            train_events,
            predictions.times,
            predictions.events,
            predictions.survival,
            horizons_arr,
            calibration_bins=calibration_bins,
        )
        stratified = stratified_discrimination_summary(train, predictions, horizons_arr)
        rows.append(
            {
                "condition": condition,
                "dropped_modalities": "+".join(dropped) if dropped else "none",
                "patients": int(len(predictions.patient_ids)),
                "events": int(predictions.events.sum()),
                "c_index": metrics.get("c_index", float("nan")),
                "cancer_stratified_c_index": stratified["cancer_stratified_c_index"],
                "ibs": metrics.get("ibs", float("nan")),
                "mean_time_dependent_auc": metrics.get(
                    "mean_time_dependent_auc", float("nan")
                ),
            }
        )
    frame = pd.DataFrame(rows)
    reference = frame.loc[frame["condition"] == "all_observed"].iloc[0]
    for metric in [
        "c_index",
        "cancer_stratified_c_index",
        "ibs",
        "mean_time_dependent_auc",
    ]:
        frame[f"delta_{metric}_vs_all_observed"] = frame[metric] - float(reference[metric])
    return frame
