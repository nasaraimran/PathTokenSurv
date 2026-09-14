from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from pathtokensurv.ablation import CORE_METRICS, DEGRADATION_DIRECTION
from pathtokensurv.constants import MOLECULAR_MODALITIES
from pathtokensurv.data.pathways import PathwaySpec
from pathtokensurv.utils.io import save_json


CONTROL_NAME = "feature_permuted_pathway_tokens"
CONTROL_DESCRIPTION = (
    "Deterministically permute selected feature labels separately within mRNA, miRNA, and CNV, "
    "then apply the same bijection to every pathway membership and the residual token. This preserves "
    "the exact pathway-token sizes, feature membership-degree distribution, pairwise token-overlap "
    "topology, residual-token size, pathway adjacency, model capacity, and frozen training protocol "
    "while breaking the biological feature-to-pathway correspondence."
)
DEFAULT_PERMUTATION_SEED_BASE = 160000
_MODALITY_SEED_CODES = {"mrna": 11, "mirna": 23, "cnv": 37}


def _deranged_permutation(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return a deterministic random permutation with no fixed points when n > 1."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return np.arange(n, dtype=int)
    perm = rng.permutation(n).astype(int, copy=False)
    identity = np.arange(n, dtype=int)
    fixed = np.flatnonzero(perm == identity)
    if len(fixed) > 1:
        perm[fixed] = np.roll(perm[fixed], 1)
    elif len(fixed) == 1:
        i = int(fixed[0])
        j = (i + 1) % n
        perm[i], perm[j] = perm[j], perm[i]
    if np.any(perm == identity):
        raise RuntimeError("Failed to construct a derangement.")
    if len(np.unique(perm)) != n:
        raise RuntimeError("Permutation is not bijective.")
    return perm


def _incidence(spec: PathwaySpec, modality: str) -> np.ndarray:
    n_features = len(spec.feature_names[modality])
    matrix = np.zeros((spec.num_pathways, n_features), dtype=np.int8)
    for pathway_idx, indices in enumerate(spec.feature_indices[modality]):
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate feature index inside {modality} pathway {pathway_idx}.")
        if indices:
            matrix[pathway_idx, np.asarray(indices, dtype=int)] = 1
    return matrix


def validate_feature_permuted_spec(primary: PathwaySpec, control: PathwaySpec) -> dict:
    """Verify that the control changes labels only, not pathway-token structure."""
    if primary.pathway_names != control.pathway_names:
        raise ValueError("Pathway names/order changed in pathway-token control.")
    if primary.feature_names != control.feature_names:
        raise ValueError("Selected feature names/order changed in pathway-token control.")
    if not np.array_equal(primary.adjacency, control.adjacency):
        raise ValueError("Pathway adjacency changed in pathway-token control.")
    if primary.eligible_feature_counts != control.eligible_feature_counts:
        raise ValueError("Eligible feature-count metadata changed in pathway-token control.")
    if not np.allclose(primary.pathway_selection_scores, control.pathway_selection_scores, rtol=0, atol=0):
        raise ValueError("Pathway selection scores changed in pathway-token control.")

    modality_checks: Dict[str, dict] = {}
    for modality in MOLECULAR_MODALITIES:
        p_sizes = [len(x) for x in primary.feature_indices[modality]]
        c_sizes = [len(x) for x in control.feature_indices[modality]]
        if p_sizes != c_sizes:
            raise ValueError(f"Pathway token sizes changed for {modality}.")
        if len(primary.residual_indices[modality]) != len(control.residual_indices[modality]):
            raise ValueError(f"Residual-token size changed for {modality}.")

        p_inc = _incidence(primary, modality)
        c_inc = _incidence(control, modality)
        p_overlap = p_inc @ p_inc.T
        c_overlap = c_inc @ c_inc.T
        if not np.array_equal(p_overlap, c_overlap):
            raise ValueError(f"Pairwise pathway-overlap topology changed for {modality}.")
        p_degree = np.sort(p_inc.sum(axis=0))
        c_degree = np.sort(c_inc.sum(axis=0))
        if not np.array_equal(p_degree, c_degree):
            raise ValueError(f"Feature membership-degree distribution changed for {modality}.")

        n_features = len(primary.feature_names[modality])
        p_mapped = set(np.flatnonzero(p_inc.sum(axis=0) > 0).tolist())
        c_mapped = set(np.flatnonzero(c_inc.sum(axis=0) > 0).tolist())
        p_residual = set(map(int, primary.residual_indices[modality]))
        c_residual = set(map(int, control.residual_indices[modality]))
        if p_mapped & p_residual or c_mapped & c_residual:
            raise ValueError(f"Mapped/residual overlap detected for {modality}.")
        if p_mapped | p_residual != set(range(n_features)):
            raise ValueError(f"Primary spec does not cover selected {modality} features.")
        if c_mapped | c_residual != set(range(n_features)):
            raise ValueError(f"Control spec does not cover selected {modality} features.")

        modality_checks[modality] = {
            "num_features": n_features,
            "num_pathways": primary.num_pathways,
            "pathway_token_sizes_preserved": True,
            "pairwise_overlap_matrix_preserved": True,
            "feature_degree_multiset_preserved": True,
            "residual_size_preserved": True,
            "residual_features": len(control.residual_indices[modality]),
        }

    return {
        "status": "PASS",
        "pathway_names_preserved": True,
        "feature_order_preserved": True,
        "adjacency_preserved": True,
        "eligible_counts_preserved": True,
        "selection_scores_preserved": True,
        "modalities": modality_checks,
    }


def make_feature_permuted_pathway_spec(
    primary: PathwaySpec,
    seed: int,
) -> tuple[PathwaySpec, dict, Dict[str, list[int]]]:
    """Create the v1.6.0 feature-label permutation control for one frozen fold.

    A separate bijection is sampled for each molecular modality. Every occurrence
    of feature index ``i`` in pathway memberships and the residual token is replaced
    by ``permutation[i]``. Feature tensors themselves are not permuted. Therefore
    token shapes/topology stay identical while biological feature-to-pathway labels
    are deliberately broken.
    """
    control = deepcopy(primary)
    permutations: Dict[str, list[int]] = {}
    metadata: Dict[str, object] = {
        "control": CONTROL_NAME,
        "description": CONTROL_DESCRIPTION,
        "seed": int(seed),
        "outcome_information_used": False,
        "modalities": {},
    }

    for modality in MOLECULAR_MODALITIES:
        n_features = len(primary.feature_names[modality])
        modality_seed = np.random.SeedSequence([int(seed), _MODALITY_SEED_CODES[modality]])
        rng = np.random.default_rng(modality_seed)
        perm = _deranged_permutation(n_features, rng)
        permutations[modality] = perm.tolist()

        control.feature_indices[modality] = [
            [int(perm[int(idx)]) for idx in indices]
            for indices in primary.feature_indices[modality]
        ]
        control.residual_indices[modality] = [
            int(perm[int(idx)]) for idx in primary.residual_indices[modality]
        ]

        original_occurrences = sum(len(x) for x in primary.feature_indices[modality])
        same_pathway_occurrences = 0
        for before, after in zip(primary.feature_indices[modality], control.feature_indices[modality]):
            same_pathway_occurrences += len(set(before) & set(after))
        retained_fraction = (
            same_pathway_occurrences / original_occurrences if original_occurrences else 0.0
        )
        metadata["modalities"][modality] = {
            "num_features": int(n_features),
            "fixed_points": int(np.sum(perm == np.arange(n_features))),
            "changed_feature_labels": int(np.sum(perm != np.arange(n_features))),
            "pathway_membership_occurrences": int(original_occurrences),
            "same_pathway_occurrence_fraction_after_permutation": float(retained_fraction),
            "residual_features": int(len(primary.residual_indices[modality])),
        }

    invariants = validate_feature_permuted_spec(primary, control)
    metadata["invariant_validation"] = invariants
    return control, metadata, permutations


def _mean_ci(values: Sequence[float]) -> dict:
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


def _degradation(primary: np.ndarray, control: np.ndarray, metric: str) -> np.ndarray:
    direction = DEGRADATION_DIRECTION[metric]
    if direction == "primary_minus_ablation":
        return primary - control
    if direction == "ablation_minus_primary":
        return control - primary
    raise ValueError(direction)


def aggregate_pathway_token_control(
    primary_fold_dirs: Sequence[str | Path],
    control_root: str | Path,
    output_dir: str | Path,
) -> dict:
    """Aggregate the five paired biological-vs-feature-permuted token experiments."""
    if len(primary_fold_dirs) != 5:
        raise ValueError("Exactly five frozen primary outer-fold directories are required.")
    control_root = Path(control_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_rows = []
    for fallback_fold, raw_dir in enumerate(primary_fold_dirs, start=1):
        fold_dir = Path(raw_dir)
        metrics_path = fold_dir / "test_metrics_extended.json"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        manifest_path = fold_dir / "experiment_manifest.json"
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

    control_rows = []
    for fold in range(1, 6):
        fold_dir = control_root / f"fold_{fold:02d}"
        metrics_path = fold_dir / "test_metrics_extended.json"
        manifest_path = fold_dir / "pathway_control_manifest.json"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS":
            raise RuntimeError(f"Pathway-token control run is not PASS: {manifest_path}")
        if int(manifest.get("outer_fold", -1)) != fold:
            raise ValueError(f"Fold mismatch in {manifest_path}")
        if manifest.get("control") != CONTROL_NAME:
            raise ValueError(f"Control mismatch in {manifest_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        summary_path = fold_dir / "training_summary.json"
        training = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        control_rows.append({
            "fold": fold,
            "artifact_dir": str(fold_dir),
            "permutation_seed": manifest.get("permutation_seed"),
            "best_epoch": training.get("best_epoch", np.nan),
            "training_seconds": training.get("training_seconds", np.nan),
            **{metric: metrics.get(metric, np.nan) for metric in CORE_METRICS},
        })
    control = pd.DataFrame(control_rows).sort_values("fold")
    control.to_csv(output_dir / "control_fold_metrics.csv", index=False)

    merged = primary.merge(control, on="fold", suffixes=("_primary", "_control"), validate="one_to_one")
    paired_rows = []
    summary_rows = []
    publication = {
        "control": CONTROL_NAME,
        "description": CONTROL_DESCRIPTION,
    }
    for metric in CORE_METRICS:
        p = merged[f"{metric}_primary"].to_numpy(dtype=float)
        c = merged[f"{metric}_control"].to_numpy(dtype=float)
        degradation = _degradation(p, c, metric)
        for fold, pv, cv, dv in zip(merged["fold"], p, c, degradation):
            paired_rows.append({
                "fold": int(fold),
                "metric": metric,
                "primary": float(pv),
                "feature_permuted_control": float(cv),
                "degradation_positive_is_control_worse": float(dv),
            })
        p_summary = _mean_ci(p)
        c_summary = _mean_ci(c)
        d_summary = _mean_ci(degradation)
        summary_rows.append({
            "metric": metric,
            **{f"primary_{k}": v for k, v in p_summary.items()},
            **{f"control_{k}": v for k, v in c_summary.items()},
            **{f"degradation_{k}": v for k, v in d_summary.items()},
        })
        publication[f"primary_{metric}_mean"] = p_summary["mean"]
        publication[f"control_{metric}_mean"] = c_summary["mean"]
        publication[f"{metric}_degradation_mean"] = d_summary["mean"]
        publication[f"{metric}_degradation_ci95_low"] = d_summary["ci95_low"]
        publication[f"{metric}_degradation_ci95_high"] = d_summary["ci95_high"]

    pd.DataFrame(paired_rows).to_csv(output_dir / "paired_fold_differences.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "pathway_token_control_summary.csv", index=False)
    pd.DataFrame([publication]).to_csv(output_dir / "pathway_token_control_publication_table.csv", index=False)

    lines = [
        "# PathTokenSurv v1.6.0 Pathway-Tokenization Control Summary",
        "",
        "Secondary sensitivity analysis: correct biological pathway assignments are compared with a deterministic feature-label permutation control.",
        "Positive degradation means the feature-permuted control performed worse than the frozen primary model.",
        "All intervals are two-sided 95% Student-t intervals across the five paired outer-fold differences.",
        "",
        "| Metric | Primary mean | Permuted-token mean | Mean degradation | 95% paired CI |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "c_index": "Pooled C-index",
        "cancer_stratified_c_index": "Cancer-stratified C-index",
        "ibs": "IBS",
        "mean_time_dependent_auc": "Mean tdAUC",
        "mean_cancer_stratified_time_dependent_auc": "Cancer-stratified tdAUC",
        "integrated_ipcw_ece": "Integrated IPCW-ECE",
    }
    summary_df = pd.DataFrame(summary_rows).set_index("metric")
    for metric in CORE_METRICS:
        row = summary_df.loc[metric]
        lines.append(
            f"| {labels[metric]} | {row['primary_mean']:.4f} | {row['control_mean']:.4f} | "
            f"{row['degradation_mean']:+.4f} | [{row['degradation_ci95_low']:+.4f}, {row['degradation_ci95_high']:+.4f}] |"
        )
    (output_dir / "PATHWAY_TOKEN_CONTROL_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "status": "PASS",
        "package_feature": "v1.6.0 pathway-tokenization control",
        "control": CONTROL_NAME,
        "primary_fold_count": 5,
        "completed_control_runs": int(len(control)),
        "paired_ci_method": "two-sided 95% Student-t interval across five paired outer-fold differences",
        "degradation_convention": "positive means worse after replacing biological feature-to-pathway assignments with the feature-permuted control",
        "degradation_direction": DEGRADATION_DIRECTION,
        "secondary_sensitivity_analysis": True,
        "note": "The frozen primary folds are reused as reference and are not retrained. The control preserves pathway-token structure and breaks only biological feature-to-pathway correspondence.",
    }
    save_json(manifest, output_dir / "pathway_token_control_aggregation_manifest.json")
    return manifest
