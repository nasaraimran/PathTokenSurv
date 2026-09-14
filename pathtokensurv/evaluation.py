from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from pathtokensurv.data.dataset import MultiModalSurvivalDataset
from pathtokensurv.data.raw import load_raw_cohort
from pathtokensurv.experiment import save_predictions
from pathtokensurv.inference import load_inference_model
from pathtokensurv.metrics import evaluate_survival_predictions
from pathtokensurv.trainer import PredictionBundle
from pathtokensurv.utils.device import move_batch_to_device, resolve_device
from pathtokensurv.utils.io import save_json


@torch.no_grad()
def evaluate_artifact(
    artifact_dir: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
) -> dict:
    artifact_dir = Path(artifact_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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

    patient_ids = []
    times, events, cancers, survival, fused = [], [], [], [], []
    for batch in loader:
        patient_ids.extend(list(batch["patient_id"]))
        batch = move_batch_to_device(batch, device)
        output = model(batch)
        times.append(batch["time"].cpu().numpy())
        events.append(batch["event"].cpu().numpy())
        cancers.append(batch["cancer"].cpu().numpy())
        survival.append(output["survival"].cpu().numpy())
        fused.append(output["fused"].cpu().numpy())

    predictions = PredictionBundle(
        patient_ids=patient_ids,
        times=np.concatenate(times),
        events=np.concatenate(events),
        cancers=np.concatenate(cancers),
        survival=np.concatenate(survival),
        fused=np.concatenate(fused),
    )
    reference = np.load(artifact_dir / "training_reference.npz")
    metrics = evaluate_survival_predictions(
        train_times=reference["times"],
        train_events=reference["events"],
        test_times=predictions.times,
        test_events=predictions.events,
        survival=predictions.survival,
        horizons=discretizer.right_edges,
        calibration_bins=config.evaluation.calibration_bins,
    )
    save_json(metrics, output_dir / "metrics.json")
    save_predictions(predictions, discretizer.right_edges, output_dir / "predictions.csv")
    return metrics
