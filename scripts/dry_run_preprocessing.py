from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pathtokensurv.config import ExperimentConfig
from pathtokensurv.data.dataset import MultiModalSurvivalDataset
from pathtokensurv.data.pathways import build_pathway_spec, load_pathway_mapping, validate_selected_annotation_coverage
from pathtokensurv.data.preprocessing import FoldPreprocessor
from pathtokensurv.data.qc import patient_availability_table, modality_pattern_summary
from pathtokensurv.data.raw import load_raw_cohort
from pathtokensurv.data.splitting import outer_folds
from pathtokensurv.data.time import TimeDiscretizer
from pathtokensurv.models.model import PathTokenSurv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU dry-run of fold-specific preprocessing, pathway construction, and one PathTokenSurv forward pass."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="outputs/tcga_dry_run_v1.5.5")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = load_raw_cohort(cfg.data, require_outcomes=True)
    cancers = raw.outcomes[cfg.data.cancer_col].astype(str).to_numpy()
    events = raw.outcomes[cfg.data.event_col].astype(int).to_numpy()
    folds = list(outer_folds(cancers, events, args.outer_folds, cfg.training.seed))
    if not 1 <= args.fold <= len(folds):
        raise ValueError(f"--fold must be between 1 and {len(folds)}")
    train_idx, test_idx = folds[args.fold - 1]

    preprocessor = FoldPreprocessor(cfg.data).fit(raw, train_idx)
    train = preprocessor.transform(raw, train_idx)
    test = preprocessor.transform(raw, test_idx)
    preprocessor.save(out / "preprocessor.pkl")

    mapping_path = Path(cfg.data.data_dir) / cfg.data.pathway_file
    if not mapping_path.exists():
        raise FileNotFoundError(f"Pathway mapping not found: {mapping_path}")
    mapping = load_pathway_mapping(mapping_path)
    validate_selected_annotation_coverage(
        mapping,
        train.feature_names,
        cfg.data.min_selected_annotation_coverage_pct,
        conservative_mirna_alias_harmonization=cfg.data.mirna_conservative_alias_harmonization,
    )
    spec = build_pathway_spec(
        mapping,
        train.feature_names,
        min_features_per_pathway=cfg.data.min_features_per_pathway,
        max_features_per_pathway=cfg.data.max_features_per_pathway,
        max_features_per_pathway_by_modality=cfg.data.max_features_per_pathway_by_modality,
        conservative_mirna_alias_harmonization=cfg.data.mirna_conservative_alias_harmonization,
        min_modalities_per_pathway=cfg.data.min_modalities_per_pathway,
        max_pathways=cfg.data.max_pathways,
        balance_pathway_sources=cfg.data.balance_pathway_sources,
    )
    spec.save(out / "pathway_spec.json")

    discretizer = TimeDiscretizer().fit(train.times.numpy(), train.events.numpy(), cfg.model.num_time_bins)
    cfg.model.num_time_bins = discretizer.num_bins
    discretizer.save(out / "time_discretizer.json")

    model = PathTokenSurv(
        model_config=cfg.model,
        pathway_spec=spec,
        category_cardinalities=preprocessor.category_cardinalities,
        num_continuous_clinical=len(cfg.data.clinical_continuous),
        num_cancers=preprocessor.num_cancers,
    ).cpu().eval()

    loader = DataLoader(MultiModalSurvivalDataset(test), batch_size=max(1, args.batch_size), shuffle=False)
    batch = next(iter(loader))
    with torch.no_grad():
        output = model(batch)

    availability = patient_availability_table(raw, cfg.data)
    patterns = modality_pattern_summary(availability)
    report = {
        "status": "PASS",
        "version": "1.5.5",
        "device": "cpu",
        "eligible_patients": len(raw),
        "training_patients": int(len(train_idx)),
        "test_patients": int(len(test_idx)),
        "selected_features": {m: len(preprocessor.molecular_transforms[m].selected_features) for m in preprocessor.molecular_transforms},
        "retained_pathways": int(spec.num_pathways),
        "residual_features": {m: len(spec.residual_indices[m]) for m in spec.residual_indices},
        "time_bins": int(discretizer.num_bins),
        "test_batch_size": int(batch["time"].shape[0]),
        "logits_shape": list(output["logits"].shape),
        "survival_shape": list(output["survival"].shape),
        "fused_shape": list(output["fused"].shape),
        "survival_finite": bool(torch.isfinite(output["survival"]).all()),
        "survival_range": [float(output["survival"].min()), float(output["survival"].max())],
        "availability_patterns": patterns[["mask", "pattern", "patients", "percent"]].to_dict(orient="records"),
    }
    if not report["survival_finite"]:
        report["status"] = "FAIL"
    if report["survival_range"][0] < -1e-6 or report["survival_range"][1] > 1 + 1e-6:
        report["status"] = "FAIL"

    (out / "dry_run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError("Dry run failed validation checks.")
    print(f"Dry run passed. Artifacts: {out}")


if __name__ == "__main__":
    main()
