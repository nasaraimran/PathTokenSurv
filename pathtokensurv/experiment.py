from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pathtokensurv.config import ExperimentConfig
from pathtokensurv.data.dataset import MultiModalSurvivalDataset
from pathtokensurv.data.pathways import build_pathway_spec, load_pathway_mapping, validate_selected_annotation_coverage
from pathtokensurv.data.preprocessing import FoldPreprocessor, ProcessedCohort
from pathtokensurv.data.raw import RawCohort
from pathtokensurv.data.time import TimeDiscretizer
from pathtokensurv.metrics import evaluate_survival_predictions, risk_from_survival
from pathtokensurv.models.model import PathTokenSurv
from pathtokensurv.plotting import plot_mean_survival
from pathtokensurv.scientific_evaluation import save_scientific_evaluation
from pathtokensurv.robustness import (
    evaluate_complete_case_modality_dropouts,
    evaluate_natural_missingness_groups,
)
from pathtokensurv.trainer import PredictionBundle, Trainer
from pathtokensurv.utils.device import resolve_device
from pathtokensurv.utils.io import save_json
from pathtokensurv.utils.seed import seed_everything


@dataclass
class ExperimentResult:
    metrics: Dict[str, float]
    predictions: PredictionBundle
    output_dir: Path


def make_loader(
    cohort: ProcessedCohort,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        MultiModalSurvivalDataset(cohort),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def save_predictions(
    predictions: PredictionBundle,
    right_edges: np.ndarray,
    output_path: str | Path,
    cancer_labels: Sequence[str] | None = None,
    evaluation_horizon_max: float | None = None,
) -> None:
    payload = {
        "patient_id": predictions.patient_ids,
        "time": predictions.times,
        "event": predictions.events,
        "cancer_id": predictions.cancers,
        # Legacy/full-grid score is retained for backward compatibility.
        "risk_score": risk_from_survival(predictions.survival, right_edges),
    }
    if evaluation_horizon_max is not None:
        edges = np.asarray(right_edges, dtype=float)
        supported = np.flatnonzero(edges <= float(evaluation_horizon_max) + 1e-6)
        if supported.size:
            payload["evaluation_risk_score"] = risk_from_survival(
                predictions.survival[:, supported], edges[supported]
            )
    if cancer_labels is not None:
        labels = list(cancer_labels)
        payload["cancer_type"] = [
            labels[int(idx)] if 0 <= int(idx) < len(labels) else "UNKNOWN"
            for idx in predictions.cancers
        ]
    frame = pd.DataFrame(payload)
    for idx, horizon in enumerate(right_edges):
        frame[f"survival_{float(horizon):.6g}"] = predictions.survival[:, idx]
    frame.to_csv(output_path, index=False)


def run_single_split(
    raw: RawCohort,
    config: ExperimentConfig,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    test_indices: Sequence[int],
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    run_extended_robustness: bool = False,
) -> ExperimentResult:
    config.validate()
    seed_everything(config.training.seed)
    output_dir = Path(output_dir or config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.json")

    preprocessor = FoldPreprocessor(config.data).fit(raw, train_indices)
    train = preprocessor.transform(raw, train_indices)
    val = preprocessor.transform(raw, val_indices)
    test = preprocessor.transform(raw, test_indices)

    pathway_mapping = load_pathway_mapping(Path(config.data.data_dir) / config.data.pathway_file)
    validate_selected_annotation_coverage(
        pathway_mapping,
        train.feature_names,
        config.data.min_selected_annotation_coverage_pct,
        conservative_mirna_alias_harmonization=config.data.mirna_conservative_alias_harmonization,
    )
    pathway_spec = build_pathway_spec(
        pathway_mapping,
        train.feature_names,
        min_features_per_pathway=config.data.min_features_per_pathway,
        max_features_per_pathway=config.data.max_features_per_pathway,
        max_features_per_pathway_by_modality=config.data.max_features_per_pathway_by_modality,
        conservative_mirna_alias_harmonization=config.data.mirna_conservative_alias_harmonization,
        min_modalities_per_pathway=config.data.min_modalities_per_pathway,
        max_pathways=config.data.max_pathways,
        balance_pathway_sources=config.data.balance_pathway_sources,
    )
    pathway_spec.save(output_dir / "pathway_spec.json")
    preprocessor.save(output_dir / "preprocessor.pkl")

    discretizer = TimeDiscretizer().fit(
        train.times.numpy(),
        train.events.numpy(),
        config.model.num_time_bins,
    )
    # Quantile boundaries can merge when event times repeat.
    config.model.num_time_bins = discretizer.num_bins
    config.save(output_dir / "config.json")
    discretizer.save(output_dir / "time_discretizer.json")
    np.savez(
        output_dir / "training_reference.npz",
        times=train.times.numpy(),
        events=train.events.numpy(),
    )

    model = PathTokenSurv(
        model_config=config.model,
        pathway_spec=pathway_spec,
        category_cardinalities=preprocessor.category_cardinalities,
        num_continuous_clinical=len(config.data.clinical_continuous),
        num_cancers=preprocessor.num_cancers,
    )
    device = resolve_device(config.training.device)
    trainer = Trainer(model, discretizer, config.training, device, output_dir)

    train_loader = make_loader(train, config.training.batch_size, True, config.training.num_workers)
    val_loader = make_loader(val, config.training.batch_size, False, config.training.num_workers)
    test_loader = make_loader(test, config.training.batch_size, False, config.training.num_workers)

    trainer.fit(
        train_loader,
        val_loader,
        train_times=train.times.numpy(),
        train_events=train.events.numpy(),
        resume_from=resume_from,
    )
    predictions = trainer.predict(test_loader)
    metrics = evaluate_survival_predictions(
        train_times=train.times.numpy(),
        train_events=train.events.numpy(),
        test_times=predictions.times,
        test_events=predictions.events,
        survival=predictions.survival,
        horizons=discretizer.right_edges,
        calibration_bins=config.evaluation.calibration_bins,
    )
    save_json(metrics, output_dir / "test_metrics.json")
    save_predictions(
        predictions,
        discretizer.right_edges,
        output_dir / "test_predictions.csv",
        cancer_labels=test.cancer_labels,
        evaluation_horizon_max=metrics.get("evaluation_horizon_max"),
    )
    scientific_summary = save_scientific_evaluation(
        train=train,
        predictions=predictions,
        horizons=discretizer.right_edges,
        output_dir=output_dir,
        calibration_bins=config.evaluation.calibration_bins,
    )
    save_json(
        {**metrics, **scientific_summary},
        output_dir / "test_metrics_extended.json",
    )

    if run_extended_robustness:
        natural_missingness = evaluate_natural_missingness_groups(
            train=train,
            test=test,
            predictions=predictions,
            horizons=discretizer.right_edges,
            calibration_bins=config.evaluation.calibration_bins,
        )
        natural_missingness.to_csv(
            output_dir / "natural_missingness_metrics.csv", index=False
        )
        complete_case_dropouts = evaluate_complete_case_modality_dropouts(
            model=trainer.model,
            train=train,
            test=test,
            horizons=discretizer.right_edges,
            calibration_bins=config.evaluation.calibration_bins,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            device=device,
        )
        complete_case_dropouts.to_csv(
            output_dir / "complete_case_modality_drop_metrics.csv", index=False
        )

    plot_mean_survival(
        predictions.survival,
        discretizer.right_edges,
        output_dir / "test_mean_survival.png",
    )
    torch.save(
        {
            "model_state": trainer.model.state_dict(),
            "model_config": config.model.__dict__,
            "category_cardinalities": preprocessor.category_cardinalities,
            "num_continuous_clinical": len(config.data.clinical_continuous),
            "num_cancers": preprocessor.num_cancers,
        },
        output_dir / "model_for_inference.pt",
    )
    return ExperimentResult(metrics=metrics, predictions=predictions, output_dir=output_dir)
