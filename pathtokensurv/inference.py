from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pathtokensurv.config import ExperimentConfig, ModelConfig
from pathtokensurv.data.dataset import MultiModalSurvivalDataset
from pathtokensurv.data.pathways import PathwaySpec
from pathtokensurv.data.preprocessing import FoldPreprocessor
from pathtokensurv.data.raw import load_raw_cohort
from pathtokensurv.data.time import TimeDiscretizer
from pathtokensurv.metrics import risk_from_survival
from pathtokensurv.models.model import PathTokenSurv
from pathtokensurv.utils.device import move_batch_to_device, resolve_device


def load_inference_model(
    artifact_dir: str | Path,
    device: torch.device,
) -> tuple[PathTokenSurv, ExperimentConfig, FoldPreprocessor, TimeDiscretizer]:
    artifact_dir = Path(artifact_dir)
    config = ExperimentConfig.load(artifact_dir / "config.json")
    preprocessor = FoldPreprocessor.load(artifact_dir / "preprocessor.pkl")
    pathway_spec = PathwaySpec.load(artifact_dir / "pathway_spec.json")
    discretizer = TimeDiscretizer.load(artifact_dir / "time_discretizer.json")
    checkpoint = torch.load(
        artifact_dir / "model_for_inference.pt",
        map_location=device,
        weights_only=False,
    )
    model_config = ModelConfig(**checkpoint["model_config"])
    model = PathTokenSurv(
        model_config=model_config,
        pathway_spec=pathway_spec,
        category_cardinalities=checkpoint["category_cardinalities"],
        num_continuous_clinical=int(checkpoint["num_continuous_clinical"]),
        num_cancers=int(checkpoint["num_cancers"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, config, preprocessor, discretizer


@torch.no_grad()
def run_inference(
    artifact_dir: str | Path,
    data_dir: str | Path,
    output_csv: str | Path,
    device_name: str = "auto",
) -> pd.DataFrame:
    device = resolve_device(device_name)
    model, config, preprocessor, discretizer = load_inference_model(artifact_dir, device)
    config.data.data_dir = str(data_dir)
    raw = load_raw_cohort(config.data, require_outcomes=False)
    processed = preprocessor.transform(raw, np.arange(len(raw)))
    loader = DataLoader(
        MultiModalSurvivalDataset(processed),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )

    patient_ids = []
    survival_parts = []
    uncertainty_parts = []
    for batch in loader:
        patient_ids.extend(list(batch["patient_id"]))
        batch = move_batch_to_device(batch, device)
        output = model(batch)
        survival_parts.append(output["survival"].cpu().numpy())
        uncertainty_parts.append(np.full(output["survival"].shape[0], np.nan, dtype=np.float32))

    survival = np.concatenate(survival_parts)
    frame = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "risk_score": risk_from_survival(survival, discretizer.right_edges),
            "uncertainty": np.concatenate(uncertainty_parts),
        }
    )
    for idx, horizon in enumerate(discretizer.right_edges):
        frame[f"survival_{float(horizon):.6g}"] = survival[:, idx]
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame
