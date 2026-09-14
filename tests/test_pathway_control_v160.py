from pathlib import Path
import json

import numpy as np
import pandas as pd

from pathtokensurv.data.pathways import PathwaySpec
from pathtokensurv.pathway_control import (
    CONTROL_NAME,
    aggregate_pathway_token_control,
    make_feature_permuted_pathway_spec,
    validate_feature_permuted_spec,
)


def _spec() -> PathwaySpec:
    names = ["P1", "P2", "P3"]
    feature_names = {
        "mrna": [f"g{i}" for i in range(8)],
        "mirna": [f"m{i}" for i in range(7)],
        "cnv": [f"c{i}" for i in range(9)],
    }
    feature_indices = {
        "mrna": [[0, 1, 2], [2, 3], [3, 4, 5]],
        "mirna": [[0, 1], [1, 2, 3], [3, 4]],
        "cnv": [[0, 1, 2], [2, 3, 4], [4, 5, 6]],
    }
    residual = {
        "mrna": [6, 7],
        "mirna": [5, 6],
        "cnv": [7, 8],
    }
    return PathwaySpec(
        pathway_names=names,
        feature_names=feature_names,
        feature_indices=feature_indices,
        residual_indices=residual,
        adjacency=np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32),
        eligible_feature_counts={m: [len(x) for x in feature_indices[m]] for m in feature_indices},
        pathway_selection_scores=[0.3, 0.2, 0.1],
    )


def test_v160_feature_permutation_preserves_structure_and_is_deterministic():
    primary = _spec()
    control1, metadata1, permutations1 = make_feature_permuted_pathway_spec(primary, seed=160001)
    control2, metadata2, permutations2 = make_feature_permuted_pathway_spec(primary, seed=160001)
    assert control1.to_dict() == control2.to_dict()
    assert permutations1 == permutations2
    assert metadata1 == metadata2
    checks = validate_feature_permuted_spec(primary, control1)
    assert checks["status"] == "PASS"
    assert np.array_equal(primary.adjacency, control1.adjacency)
    assert primary.pathway_names == control1.pathway_names
    assert primary.feature_names == control1.feature_names
    for modality, permutation in permutations1.items():
        p = np.asarray(permutation)
        assert sorted(p.tolist()) == list(range(len(p)))
        assert (p != np.arange(len(p))).all()
        assert metadata1["modalities"][modality]["fixed_points"] == 0


def test_v160_different_seed_changes_assignments_without_changing_structure():
    primary = _spec()
    a, _, _ = make_feature_permuted_pathway_spec(primary, seed=160001)
    b, _, _ = make_feature_permuted_pathway_spec(primary, seed=160002)
    assert any(a.feature_indices[m] != b.feature_indices[m] for m in a.feature_indices)
    assert validate_feature_permuted_spec(primary, a)["status"] == "PASS"
    assert validate_feature_permuted_spec(primary, b)["status"] == "PASS"


def _write_metrics(path: Path, shift: float):
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "c_index": 0.78 - shift,
        "cancer_stratified_c_index": 0.65 - shift,
        "ibs": 0.135 + shift,
        "mean_time_dependent_auc": 0.82 - shift,
        "mean_cancer_stratified_time_dependent_auc": 0.68 - shift,
        "integrated_ipcw_ece": 0.036 + shift,
    }
    (path / "test_metrics_extended.json").write_text(json.dumps(payload), encoding="utf-8")


def test_v160_aggregation_pairs_all_five_folds(tmp_path):
    primaries = []
    for fold in range(1, 6):
        p = tmp_path / f"primary_{fold}"
        _write_metrics(p, 0.0)
        (p / "experiment_manifest.json").write_text(json.dumps({"outer_fold": fold}), encoding="utf-8")
        primaries.append(p)

    root = tmp_path / "control"
    for fold in range(1, 6):
        p = root / f"fold_{fold:02d}"
        _write_metrics(p, 0.01)
        (p / "pathway_control_manifest.json").write_text(
            json.dumps({
                "status": "PASS",
                "outer_fold": fold,
                "control": CONTROL_NAME,
                "permutation_seed": 160000 + fold,
            }),
            encoding="utf-8",
        )
        (p / "training_summary.json").write_text(
            json.dumps({"best_epoch": 10, "training_seconds": 100.0}), encoding="utf-8"
        )

    out = tmp_path / "summary"
    manifest = aggregate_pathway_token_control(primaries, root, out)
    assert manifest["completed_control_runs"] == 5
    paired = pd.read_csv(out / "paired_fold_differences.csv")
    assert len(paired) == 5 * 6
    assert (paired["degradation_positive_is_control_worse"] > 0).all()
    publication = pd.read_csv(out / "pathway_token_control_publication_table.csv")
    assert len(publication) == 1
    assert publication.loc[0, "control"] == CONTROL_NAME
    assert (out / "PATHWAY_TOKEN_CONTROL_SUMMARY.md").exists()
