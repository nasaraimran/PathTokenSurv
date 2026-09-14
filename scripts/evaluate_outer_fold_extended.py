from __future__ import annotations

# Allow direct execution from a source checkout.
import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pathtokensurv import __version__
from pathtokensurv.data.dataset import MultiModalSurvivalDataset
from pathtokensurv.data.raw import load_raw_cohort
from pathtokensurv.inference import load_inference_model
from pathtokensurv.experiment import save_predictions
from pathtokensurv.metrics import evaluate_survival_predictions
from pathtokensurv.robustness import (
    evaluate_complete_case_modality_dropouts,
    evaluate_natural_missingness_groups,
    predict_with_mask_function,
)
from pathtokensurv.scientific_evaluation import save_scientific_evaluation
from pathtokensurv.utils.device import resolve_device
from pathtokensurv.utils.io import save_json


def _indices_from_assignments(raw, assignments: pd.DataFrame, split: str) -> np.ndarray:
    wanted = assignments.loc[assignments["split"] == split, "patient_id"].astype(str).tolist()
    position = {str(pid): idx for idx, pid in enumerate(raw.patient_ids)}
    missing = [pid for pid in wanted if pid not in position]
    if missing:
        raise ValueError(f"{len(missing)} {split} patient IDs are absent from the loaded cohort; first={missing[:3]}")
    return np.asarray([position[pid] for pid in wanted], dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc v1.5.7 evaluation for an existing outer-fold artifact directory. "
            "No model parameter is updated and the held-out patient split is unchanged."
        )
    )
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    artifact_dir = Path(args.artifacts)
    device = resolve_device(args.device)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA was required for post-hoc evaluation but is unavailable.")

    model, config, preprocessor, discretizer = load_inference_model(artifact_dir, device)
    if args.data_dir is not None:
        config.data.data_dir = str(args.data_dir)
    raw = load_raw_cohort(config.data, require_outcomes=True)

    assignments_path = artifact_dir / "split_assignments.csv"
    if not assignments_path.exists():
        raise FileNotFoundError(f"Missing split assignments: {assignments_path}")
    assignments = pd.read_csv(assignments_path)
    train_idx = _indices_from_assignments(raw, assignments, "train")
    test_idx = _indices_from_assignments(raw, assignments, "test")
    train = preprocessor.transform(raw, train_idx)
    test = preprocessor.transform(raw, test_idx)

    loader = DataLoader(
        MultiModalSurvivalDataset(test),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
    )
    predictions = predict_with_mask_function(model, loader, device, lambda natural: natural)
    metrics = evaluate_survival_predictions(
        train.times.detach().cpu().numpy(),
        train.events.detach().cpu().numpy(),
        predictions.times,
        predictions.events,
        predictions.survival,
        discretizer.right_edges,
        calibration_bins=config.evaluation.calibration_bins,
    )
    scientific = save_scientific_evaluation(
        train,
        predictions,
        discretizer.right_edges,
        artifact_dir,
        config.evaluation.calibration_bins,
    )
    save_predictions(
        predictions,
        discretizer.right_edges,
        artifact_dir / "test_predictions_extended.csv",
        cancer_labels=test.cancer_labels,
        evaluation_horizon_max=metrics.get("evaluation_horizon_max"),
    )
    save_json({**metrics, **scientific}, artifact_dir / "test_metrics_extended.json")

    natural = evaluate_natural_missingness_groups(
        train,
        test,
        predictions,
        discretizer.right_edges,
        config.evaluation.calibration_bins,
    )
    natural.to_csv(artifact_dir / "natural_missingness_metrics.csv", index=False)

    dropouts = evaluate_complete_case_modality_dropouts(
        model=model,
        train=train,
        test=test,
        horizons=discretizer.right_edges,
        calibration_bins=config.evaluation.calibration_bins,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        device=device,
    )
    dropouts.to_csv(artifact_dir / "complete_case_modality_drop_metrics.csv", index=False)

    patch_manifest = {
        "status": "PASS",
        "package_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": str(artifact_dir),
        "device": str(device),
        "training_performed": False,
        "checkpoint_modified": False,
        "split_modified": False,
        "evaluation_patch": "v1.5.7_stratified_discrimination_ipcw_robustness",
        "test_patients": int(len(test)),
        "test_events": int(test.events.sum().item()),
        "outputs": [
            "test_metrics_extended.json",
            "test_predictions_extended.csv",
            "stratified_discrimination.json",
            "stratified_auc_by_horizon.csv",
            "per_cancer_metrics.csv",
            "calibration_ipcw.csv",
            "calibration_ipcw_summary.csv",
            "calibration_ipcw_summary.json",
            "scientific_evaluation_summary.json",
            "natural_missingness_metrics.csv",
            "complete_case_modality_drop_metrics.csv",
        ],
    }
    save_json(patch_manifest, artifact_dir / "evaluation_patch_manifest.json")

    print("PathTokenSurv extended outer-fold evaluation")
    print(f"  package: {__version__}")
    print(f"  device: {device}")
    print(f"  test patients: {len(test)}")
    print(f"  pooled C-index: {metrics.get('c_index', float('nan')):.6f}")
    print(f"  cancer-stratified C-index: {scientific.get('cancer_stratified_c_index', float('nan')):.6f}")
    print(f"  IBS: {metrics.get('ibs', float('nan')):.6f}")
    print(f"  mean tdAUC: {metrics.get('mean_time_dependent_auc', float('nan')):.6f}")
    print(f"  integrated IPCW-ECE: {scientific.get('integrated_ipcw_ece', float('nan')):.6f}")
    print(f"Extended evaluation completed. Artifacts updated in: {artifact_dir}")


if __name__ == "__main__":
    main()
