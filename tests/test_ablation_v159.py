from pathlib import Path
import json

import numpy as np
import pandas as pd

from pathtokensurv.ablation import ABLATIONS, CORE_METRICS, apply_ablation, degradation, aggregate_architecture_ablations
from pathtokensurv.config import ExperimentConfig


def test_v159_ablation_definitions_are_hypothesis_driven():
    assert list(ABLATIONS) == [
        "no_structured_masking",
        "no_reconstruction_objective",
        "no_subset_consistency",
        "no_pathway_bias",
        "no_cancer_conditioning",
    ]
    base = ExperimentConfig()
    cfg, _ = apply_ablation(base, "no_structured_masking")
    assert cfg.training.modality_drop_probability == 0.0
    cfg, _ = apply_ablation(base, "no_reconstruction_objective")
    assert cfg.training.lambda_reconstruction == 0.0
    assert cfg.model.use_reconstruction is True
    cfg, _ = apply_ablation(base, "no_subset_consistency")
    assert cfg.model.use_subset_consistency is False
    assert cfg.training.lambda_consistency == 0.0
    cfg, _ = apply_ablation(base, "no_pathway_bias")
    assert cfg.model.use_pathway_bias is False
    cfg, _ = apply_ablation(base, "no_cancer_conditioning")
    assert cfg.model.use_cancer_deviation is False


def test_degradation_convention_positive_means_worse():
    primary = np.array([0.8, 0.8])
    worse_c = np.array([0.7, 0.75])
    assert np.all(degradation(primary, worse_c, "c_index") > 0)
    primary_ibs = np.array([0.13, 0.14])
    worse_ibs = np.array([0.16, 0.15])
    assert np.all(degradation(primary_ibs, worse_ibs, "ibs") > 0)


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


def test_architecture_aggregation_requires_and_pairs_all_folds(tmp_path):
    primaries = []
    for fold in range(1, 6):
        p = tmp_path / f"primary_{fold}"
        _write_metrics(p, 0.0)
        (p / "experiment_manifest.json").write_text(json.dumps({"outer_fold": fold}), encoding="utf-8")
        primaries.append(p)
    root = tmp_path / "ablations"
    for name in ABLATIONS:
        for fold in range(1, 6):
            p = root / name / f"fold_{fold:02d}"
            _write_metrics(p, 0.01)
            (p / "ablation_manifest.json").write_text(
                json.dumps({"status": "PASS", "outer_fold": fold, "ablation": name}), encoding="utf-8"
            )
            (p / "training_summary.json").write_text(
                json.dumps({"best_epoch": 10, "training_seconds": 100.0}), encoding="utf-8"
            )
    out = tmp_path / "summary"
    manifest = aggregate_architecture_ablations(primaries, root, out)
    assert manifest["completed_ablation_runs"] == 25
    paired = pd.read_csv(out / "paired_fold_differences.csv")
    assert len(paired) == 25 * len(CORE_METRICS)
    assert (paired["degradation_positive_is_worse"] > 0).all()
    assert (out / "architecture_ablation_publication_table.csv").exists()
    assert (out / "ARCHITECTURE_ABLATION_SUMMARY.md").exists()
