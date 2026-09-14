from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from pathtokensurv.config import DataConfig, ModelConfig
from pathtokensurv.data.dataset import MultiModalSurvivalDataset
from pathtokensurv.data.pathways import build_pathway_spec, load_pathway_mapping
from pathtokensurv.data.preprocessing import FoldPreprocessor
from pathtokensurv.data.raw import load_raw_cohort
from pathtokensurv.data.synthetic import generate_synthetic_cohort
from pathtokensurv.models.model import ModalityMaskSampler, PathTokenSurv


def test_model_forward_and_masking(tmp_path) -> None:
    data_dir = tmp_path / "data"
    generate_synthetic_cohort(
        data_dir,
        n_patients=40,
        n_pathways=3,
        features_per_modality=12,
        n_cancers=3,
        seed=11,
    )
    data_cfg = DataConfig(
        data_dir=str(data_dir),
        variance_thresholds={"mrna": 0.0, "mirna": 0.0, "cnv": 0.0},
        min_features_per_modality=4,
    )
    raw = load_raw_cohort(data_cfg)
    train_indices = np.arange(30)
    preprocessor = FoldPreprocessor(data_cfg).fit(raw, train_indices)
    processed = preprocessor.transform(raw, train_indices)
    mapping = load_pathway_mapping(data_dir / "pathways.json")
    spec = build_pathway_spec(mapping, processed.feature_names)
    model_cfg = ModelConfig(
        d_model=12,
        n_heads=3,
        intramodal_layers=1,
        fusion_layers=1,
        reconstruction_layers=1,
        ffn_multiplier=2,
        dropout=0.0,
        num_time_bins=4,
    )
    model = PathTokenSurv(
        model_cfg,
        spec,
        preprocessor.category_cardinalities,
        num_continuous_clinical=1,
        num_cancers=preprocessor.num_cancers,
    )
    batch = next(iter(DataLoader(MultiModalSurvivalDataset(processed), batch_size=8)))
    output = model(batch)
    assert output["logits"].shape == (8, 4)
    assert output["survival"].shape == (8, 4)
    assert torch.all(output["survival"][:, 1:] <= output["survival"][:, :-1] + 1e-7)

    sampler = ModalityMaskSampler(0.5)
    training_output = model.forward_training(batch, sampler)
    assert training_output["effective_mask"].any(dim=1).all()
    assert torch.all(training_output["effective_mask"] <= training_output["natural_mask"])
