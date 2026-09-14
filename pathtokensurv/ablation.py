from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Dict, Mapping

import numpy as np
import pandas as pd
from scipy import stats

from pathtokensurv.config import ExperimentConfig
from pathtokensurv.utils.io import save_json


@dataclass(frozen=True)
class AblationDefinition:
    name: str
    description: str
    model_overrides: Mapping[str, object]
    training_overrides: Mapping[str, object]


ABLATIONS: Dict[str, AblationDefinition] = {
    "no_structured_masking": AblationDefinition(
        name="no_structured_masking",
        description=(
            "Disable artificial modality dropping during training while preserving the natural "
            "missingness pattern. With no artificially hidden observed modality, reconstruction "
            "supervision is inactive. The stochastic same-mask consistency objective remains active."
        ),
        model_overrides={},
        training_overrides={"modality_drop_probability": 0.0},
    ),
    "no_reconstruction_objective": AblationDefinition(
        name="no_reconstruction_objective",
        description=(
            "Set lambda_reconstruction to zero while leaving the cross-modal reconstructor in the "
            "forward graph. This isolates the supervised reconstruction objective rather than "
            "removing the reconstruction module itself."
        ),
        model_overrides={},
        training_overrides={"lambda_reconstruction": 0.0},
    ),
    "no_subset_consistency": AblationDefinition(
        name="no_subset_consistency",
        description=(
            "Disable the full-view stop-gradient consistency branch and set lambda_consistency to zero."
        ),
        model_overrides={"use_subset_consistency": False},
        training_overrides={"lambda_consistency": 0.0},
    ),
    "no_pathway_bias": AblationDefinition(
        name="no_pathway_bias",
        description=(
            "Disable same-pathway and Reactome graph attention biases while retaining the identical "
            "pathway-token inputs and Transformer architecture."
        ),
        model_overrides={"use_pathway_bias": False},
        training_overrides={},
    ),
    "no_cancer_conditioning": AblationDefinition(
        name="no_cancer_conditioning",
        description=(
            "Disable cancer-specific discrete-time baseline deviations while retaining patient-level "
            "clinical and molecular representations."
        ),
        model_overrides={"use_cancer_deviation": False},
        training_overrides={},
    ),
}


CORE_METRICS = [
    "c_index",
    "cancer_stratified_c_index",
    "ibs",
    "mean_time_dependent_auc",
    "mean_cancer_stratified_time_dependent_auc",
    "integrated_ipcw_ece",
]

# Positive paired degradation always means the ablation is worse than the frozen primary model.
DEGRADATION_DIRECTION = {
    "c_index": "primary_minus_ablation",
    "cancer_stratified_c_index": "primary_minus_ablation",
    "ibs": "ablation_minus_primary",
    "mean_time_dependent_auc": "primary_minus_ablation",
    "mean_cancer_stratified_time_dependent_auc": "primary_minus_ablation",
    "integrated_ipcw_ece": "ablation_minus_primary",
}


def apply_ablation(base: ExperimentConfig, name: str) -> tuple[ExperimentConfig, AblationDefinition]:
    if name not in ABLATIONS:
        raise KeyError(f"Unknown ablation '{name}'. Valid choices: {sorted(ABLATIONS)}")
    definition = ABLATIONS[name]
    cfg = deepcopy(base)
    for key, value in definition.model_overrides.items():
        if not hasattr(cfg.model, key):
            raise AttributeError(f"ModelConfig has no field '{key}'.")
        setattr(cfg.model, key, value)
    for key, value in definition.training_overrides.items():
        if not hasattr(cfg.training, key):
            raise AttributeError(f"TrainingConfig has no field '{key}'.")
        setattr(cfg.training, key, value)
    cfg.validate()
    return cfg, definition


def _mean_ci(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        half = float(stats.t.ppf(0.975, df=n - 1) * sd / np.sqrt(n))
        low, high = mean - half, mean + half
    else:
        low = high = np.nan
    return {"n": n, "mean": mean, "sd": sd, "ci95_low": float(low), "ci95_high": float(high)}


def degradation(primary: np.ndarray, ablation: np.ndarray, metric: str) -> np.ndarray:
    primary = np.asarray(primary, dtype=float)
    ablation = np.asarray(ablation, dtype=float)
    direction = DEGRADATION_DIRECTION[metric]
    if direction == "primary_minus_ablation":
        return primary - ablation
    if direction == "ablation_minus_primary":
        return ablation - primary
    raise ValueError(direction)


def aggregate_architecture_ablations(
    primary_fold_dirs: list[str | Path],
    ablation_root: str | Path,
    output_dir: str | Path,
) -> dict:
    if len(primary_fold_dirs) != 5:
        raise ValueError("Exactly five frozen primary outer-fold directories are required.")
    ablation_root = Path(ablation_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_rows = []
    for fallback_fold, raw_dir in enumerate(primary_fold_dirs, start=1):
        fold_dir = Path(raw_dir)
        metric_path = fold_dir / "test_metrics_extended.json"
        manifest_path = fold_dir / "experiment_manifest.json"
        if not metric_path.exists():
            raise FileNotFoundError(metric_path)
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        fold = int(manifest.get("outer_fold", fallback_fold))
        primary_rows.append({
            "fold": fold,
            "artifact_dir": str(fold_dir),
            **{metric: metrics.get(metric, np.nan) for metric in CORE_METRICS},
        })
    primary = pd.DataFrame(primary_rows).sort_values("fold")
    if primary["fold"].tolist() != [1, 2, 3, 4, 5]:
        raise ValueError(f"Primary folds must resolve to [1,2,3,4,5], found {primary['fold'].tolist()}.")
    primary.to_csv(output_dir / "primary_fold_metrics.csv", index=False)

    ablation_rows = []
    for name in ABLATIONS:
        for fold in range(1, 6):
            fold_dir = ablation_root / name / f"fold_{fold:02d}"
            metric_path = fold_dir / "test_metrics_extended.json"
            manifest_path = fold_dir / "ablation_manifest.json"
            if not metric_path.exists():
                raise FileNotFoundError(f"Missing completed ablation metrics: {metric_path}")
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing ablation manifest: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "PASS":
                raise RuntimeError(f"Ablation run is not PASS: {manifest_path}")
            if int(manifest.get("outer_fold", -1)) != fold:
                raise ValueError(f"Fold mismatch in {manifest_path}")
            if manifest.get("ablation") != name:
                raise ValueError(f"Ablation mismatch in {manifest_path}")
            metrics = json.loads(metric_path.read_text(encoding="utf-8"))
            summary_path = fold_dir / "training_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            ablation_rows.append({
                "ablation": name,
                "fold": fold,
                "artifact_dir": str(fold_dir),
                "best_epoch": summary.get("best_epoch", np.nan),
                "training_seconds": summary.get("training_seconds", np.nan),
                **{metric: metrics.get(metric, np.nan) for metric in CORE_METRICS},
            })
    ablations = pd.DataFrame(ablation_rows).sort_values(["ablation", "fold"])
    ablations.to_csv(output_dir / "ablation_fold_metrics.csv", index=False)

    paired_rows = []
    summary_rows = []
    publication_rows = []
    for name, block in ablations.groupby("ablation", sort=False):
        block = block.sort_values("fold").reset_index(drop=True)
        if block["fold"].tolist() != [1, 2, 3, 4, 5]:
            raise ValueError(f"Ablation {name} does not contain exactly folds 1-5.")
        merged = primary.merge(block, on="fold", suffixes=("_primary", "_ablation"), validate="one_to_one")
        for metric in CORE_METRICS:
            p = merged[f"{metric}_primary"].to_numpy(dtype=float)
            a = merged[f"{metric}_ablation"].to_numpy(dtype=float)
            d = degradation(p, a, metric)
            for fold, pv, av, dv in zip(merged["fold"], p, a, d):
                paired_rows.append({
                    "ablation": name,
                    "fold": int(fold),
                    "metric": metric,
                    "primary": float(pv),
                    "ablation_value": float(av),
                    "degradation_positive_is_worse": float(dv),
                })
            a_summary = _mean_ci(a)
            d_summary = _mean_ci(d)
            summary_rows.append({
                "ablation": name,
                "metric": metric,
                **{f"ablation_{k}": v for k, v in a_summary.items()},
                **{f"degradation_{k}": v for k, v in d_summary.items()},
            })

        def _mean(metric: str) -> float:
            return float(block[metric].astype(float).mean())

        pub = {
            "ablation": name,
            "description": ABLATIONS[name].description,
        }
        for metric in CORE_METRICS:
            p = merged[f"{metric}_primary"].to_numpy(dtype=float)
            a = merged[f"{metric}_ablation"].to_numpy(dtype=float)
            d_summary = _mean_ci(degradation(p, a, metric))
            pub[f"{metric}_mean"] = _mean(metric)
            pub[f"{metric}_degradation_mean"] = d_summary["mean"]
            pub[f"{metric}_degradation_ci95_low"] = d_summary["ci95_low"]
            pub[f"{metric}_degradation_ci95_high"] = d_summary["ci95_high"]
        publication_rows.append(pub)

    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output_dir / "paired_fold_differences.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "architecture_ablation_summary.csv", index=False)
    publication = pd.DataFrame(publication_rows)
    publication.to_csv(output_dir / "architecture_ablation_publication_table.csv", index=False)

    # A compact Markdown summary is convenient for manuscript review and audit.
    lines = [
        "# PathTokenSurv v1.5.9 Architecture Ablation Summary",
        "",
        "Positive degradation means the ablation performed worse than the frozen primary model.",
        "All intervals are two-sided 95% Student-t intervals across the five paired outer-fold differences.",
        "",
        "| Ablation | Δ pooled C | Δ stratified C | Δ IBS | Δ tdAUC | Δ stratified tdAUC | Δ IPCW-ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ABLATIONS:
        row = publication.loc[publication["ablation"] == name].iloc[0]
        lines.append(
            "| {name} | {dc:.4f} | {dcs:.4f} | {dibs:.4f} | {dauc:.4f} | {dsauc:.4f} | {dece:.4f} |".format(
                name=name,
                dc=row["c_index_degradation_mean"],
                dcs=row["cancer_stratified_c_index_degradation_mean"],
                dibs=row["ibs_degradation_mean"],
                dauc=row["mean_time_dependent_auc_degradation_mean"],
                dsauc=row["mean_cancer_stratified_time_dependent_auc_degradation_mean"],
                dece=row["integrated_ipcw_ece_degradation_mean"],
            )
        )
    (output_dir / "ARCHITECTURE_ABLATION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "status": "PASS",
        "primary_fold_count": 5,
        "ablations": list(ABLATIONS),
        "ablation_count": len(ABLATIONS),
        "completed_ablation_runs": int(len(ablations)),
        "paired_ci_method": "two-sided 95% Student-t interval across five paired outer-fold differences",
        "degradation_convention": "positive means worse after ablation",
        "degradation_direction": DEGRADATION_DIRECTION,
        "note": "Frozen primary folds are reused as the full-model reference; they are not retrained in v1.5.9.",
    }
    save_json(manifest, output_dir / "architecture_ablation_aggregation_manifest.json")
    return manifest
