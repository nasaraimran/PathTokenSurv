from __future__ import annotations

import numpy as np

from pathtokensurv.config import DataConfig
from pathtokensurv.data.preprocessing import FoldPreprocessor
from pathtokensurv.data.raw import load_raw_cohort
from pathtokensurv.data.synthetic import generate_synthetic_cohort


def test_preprocessor_is_training_only(tmp_path) -> None:
    data_dir = tmp_path / "data"
    generate_synthetic_cohort(
        data_dir,
        n_patients=48,
        n_pathways=3,
        features_per_modality=12,
        n_cancers=3,
        seed=7,
    )
    cfg = DataConfig(
        data_dir=str(data_dir),
        variance_thresholds={"mrna": 0.0, "mirna": 0.0, "cnv": 0.0},
        min_features_per_modality=4,
    )
    raw = load_raw_cohort(cfg)
    train = np.arange(32)
    test = np.arange(32, 48)
    preprocessor = FoldPreprocessor(cfg).fit(raw, train)
    original_max = preprocessor.molecular_transforms["mrna"].maximums.copy()

    # An extreme value in a held-out patient must not change fitted parameters.
    raw.modalities["mrna"].iloc[test[0], :] = 1e9
    assert np.array_equal(original_max, preprocessor.molecular_transforms["mrna"].maximums)
    transformed = preprocessor.transform(raw, test)
    assert transformed.molecular["mrna"].shape[0] == len(test)
    assert transformed.availability.shape == (len(test), 4)
