from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_ensemble import (
    deterministic_member_seed,
    load_member_predictions,
    load_split_indices,
)


def test_member_seeds_are_deterministic_and_distinct():
    assert deterministic_member_seed(123, 2, 0) == 20123
    assert deterministic_member_seed(123, 2, 1) == 21123


def test_load_split_indices_maps_saved_patient_ids(tmp_path: Path):
    assignments = pd.DataFrame(
        {
            "patient_id": ["P3", "P1", "P4", "P2"],
            "split": ["train", "train", "validation", "test"],
        }
    )
    path = tmp_path / "split_assignments.csv"
    assignments.to_csv(path, index=False)

    train, validation, test = load_split_indices(path, ["P1", "P2", "P3", "P4"])

    np.testing.assert_array_equal(train, [2, 0])
    np.testing.assert_array_equal(validation, [3])
    np.testing.assert_array_equal(test, [1])


def test_load_member_predictions_stacks_aligned_survival_curves(tmp_path: Path):
    member_dirs = [tmp_path / "member_00", tmp_path / "member_01"]
    for member, member_dir in enumerate(member_dirs):
        member_dir.mkdir()
        pd.DataFrame(
            {
                "patient_id": ["P1", "P2"],
                "time": [10.0, 20.0],
                "event": [1.0, 0.0],
                "survival_5": [0.9 - member * 0.1, 0.8],
                "survival_10": [0.7 - member * 0.1, 0.6],
            }
        ).to_csv(member_dir / "test_predictions.csv", index=False)

    reference, survival, horizons = load_member_predictions(member_dirs)

    assert reference["patient_id"].tolist() == ["P1", "P2"]
    assert survival.shape == (2, 2, 2)
    np.testing.assert_array_equal(horizons, [5.0, 10.0])

    misaligned = pd.read_csv(member_dirs[1] / "test_predictions.csv").iloc[::-1]
    misaligned.to_csv(member_dirs[1] / "test_predictions.csv", index=False)
    with pytest.raises(ValueError, match="not aligned"):
        load_member_predictions(member_dirs)
