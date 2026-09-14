from __future__ import annotations

from pathlib import Path
import shutil
from typing import Sequence

import numpy as np
import torch

from pathtokensurv.config import ExperimentConfig
from pathtokensurv.data.dataset import MultiModalSurvivalDataset
from pathtokensurv.data.pathways import PathwaySpec
from pathtokensurv.data.preprocessing import FoldPreprocessor
from pathtokensurv.data.raw import RawCohort
from pathtokensurv.data.time import TimeDiscretizer
from pathtokensurv.experiment import ExperimentResult, make_loader, save_predictions
from pathtokensurv.metrics import evaluate_survival_predictions
from pathtokensurv.models.model import PathTokenSurv
from pathtokensurv.plotting import plot_mean_survival
from pathtokensurv.robustness import evaluate_complete_case_modality_dropouts, evaluate_natural_missingness_groups
from pathtokensurv.scientific_evaluation import save_scientific_evaluation
from pathtokensurv.trainer import Trainer
from pathtokensurv.utils.device import resolve_device
from pathtokensurv.utils.io import save_json
from pathtokensurv.utils.seed import seed_everything


def run_frozen_artifact_ablation(
    raw: RawCohort,
    config: ExperimentConfig,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    test_indices: Sequence[int],
    primary_artifacts: str | Path,
    output_dir: str | Path,
    resume_from: str | Path | None = None,
    run_extended_robustness: bool = True,
    pathway_spec_override: PathwaySpec | None = None,
) -> ExperimentResult:
    """Train a frozen-split experiment using primary preprocessing/time artifacts.

    ``pathway_spec_override`` is used only by controlled secondary experiments such
    as the v1.6.0 feature-permuted pathway-token analysis. Architecture ablations
    leave it as ``None`` and therefore reuse the primary pathway specification.
    """
    config.validate()
    seed_everything(config.training.seed)
    primary_artifacts = Path(primary_artifacts)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required = ["preprocessor.pkl", "pathway_spec.json", "time_discretizer.json"]
    for name in required:
        if not (primary_artifacts / name).exists():
            raise FileNotFoundError(primary_artifacts / name)

    preprocessor = FoldPreprocessor.load(primary_artifacts / "preprocessor.pkl")
    primary_pathway_spec = PathwaySpec.load(primary_artifacts / "pathway_spec.json")
    pathway_spec = pathway_spec_override if pathway_spec_override is not None else primary_pathway_spec
    discretizer = TimeDiscretizer.load(primary_artifacts / "time_discretizer.json")

    train = preprocessor.transform(raw, train_indices)
    val = preprocessor.transform(raw, val_indices)
    test = preprocessor.transform(raw, test_indices)

    for modality, feature_names in pathway_spec.feature_names.items():
        if list(feature_names) != list(train.feature_names[modality]):
            raise ValueError(
                f"Frozen pathway_spec feature order for {modality} does not match the frozen preprocessor transform."
            )

    config.model.num_time_bins = discretizer.num_bins
    config.save(output_dir / "config.json")
    # Keep every ablation directory self-contained and auditable.
    shutil.copy2(primary_artifacts / "preprocessor.pkl", output_dir / "preprocessor.pkl")
    if pathway_spec_override is None:
        shutil.copy2(primary_artifacts / "pathway_spec.json", output_dir / "pathway_spec.json")
    else:
        pathway_spec.save(output_dir / "pathway_spec.json")
    shutil.copy2(primary_artifacts / "time_discretizer.json", output_dir / "time_discretizer.json")
    np.savez(output_dir / "training_reference.npz", times=train.times.numpy(), events=train.events.numpy())

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
    save_json({**metrics, **scientific_summary}, output_dir / "test_metrics_extended.json")

    if run_extended_robustness:
        natural = evaluate_natural_missingness_groups(
            train=train,
            test=test,
            predictions=predictions,
            horizons=discretizer.right_edges,
            calibration_bins=config.evaluation.calibration_bins,
        )
        natural.to_csv(output_dir / "natural_missingness_metrics.csv", index=False)
        controlled = evaluate_complete_case_modality_dropouts(
            model=trainer.model,
            train=train,
            test=test,
            horizons=discretizer.right_edges,
            calibration_bins=config.evaluation.calibration_bins,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            device=device,
        )
        controlled.to_csv(output_dir / "complete_case_modality_drop_metrics.csv", index=False)

    plot_mean_survival(predictions.survival, discretizer.right_edges, output_dir / "test_mean_survival.png")
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
