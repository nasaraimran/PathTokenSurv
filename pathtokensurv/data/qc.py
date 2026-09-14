from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence
import json

import numpy as np
import pandas as pd

from pathtokensurv.config import DataConfig
from pathtokensurv.constants import MOLECULAR_MODALITIES
from pathtokensurv.data.pathways import (
    canonical_feature_name, build_pathway_spec, annotation_feature_sets,
    conservative_mirna_alias_map, mapping_conservative_mirna_alias_map,
)
from pathtokensurv.data.preprocessing import FoldPreprocessor, _numeric_frame
from pathtokensurv.data.raw import RawCohort


def patient_availability_table(raw: RawCohort, cfg: DataConfig) -> pd.DataFrame:
    """Return one row per patient with exact natural modality availability."""
    ids = pd.Index(raw.patient_ids.astype(str), name="patient_id")
    out = pd.DataFrame(index=ids)
    out["cancer_type"] = raw.outcomes.loc[ids, cfg.cancer_col].astype(str).values
    if cfg.event_col in raw.outcomes.columns:
        out["event"] = pd.to_numeric(raw.outcomes.loc[ids, cfg.event_col], errors="coerce").values
    if cfg.time_col in raw.outcomes.columns:
        out["time"] = pd.to_numeric(raw.outcomes.loc[ids, cfg.time_col], errors="coerce").values
    out["clinical"] = 1
    for modality in MOLECULAR_MODALITIES:
        frame = raw.modalities[modality].loc[ids]
        out[modality] = (~frame.isna().all(axis=1)).astype(int).values
    cols = ["clinical", *MOLECULAR_MODALITIES]
    out["num_modalities_available"] = out[cols].sum(axis=1)
    out["num_molecular_modalities_available"] = out[list(MOLECULAR_MODALITIES)].sum(axis=1)
    out["mask"] = out[cols].astype(str).agg("".join, axis=1)

    def label(row: pd.Series) -> str:
        names = ["Clinical"]
        for m, display in [("mrna", "mRNA"), ("mirna", "miRNA"), ("cnv", "CNV")]:
            if int(row[m]) == 1:
                names.append(display)
        return " + ".join(names)

    out["pattern"] = out.apply(label, axis=1)
    return out.reset_index()


def modality_pattern_summary(availability: pd.DataFrame) -> pd.DataFrame:
    total = len(availability)
    summary = (
        availability.groupby(["mask", "pattern", "clinical", "mrna", "mirna", "cnv"], dropna=False)
        .size().rename("patients").reset_index()
    )
    summary["percent"] = (summary["patients"] / max(total, 1) * 100.0).round(3)
    return summary.sort_values(["patients", "pattern"], ascending=[False, True]).reset_index(drop=True)


def cancer_specific_missingness(availability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cancer, group in availability.groupby("cancer_type", sort=True):
        row = {"cancer_type": str(cancer), "patients": int(len(group))}
        for modality in ["clinical", *MOLECULAR_MODALITIES]:
            available = int(group[modality].sum())
            missing = int(len(group) - available)
            row[f"{modality}_available"] = available
            row[f"{modality}_missing"] = missing
            row[f"{modality}_availability_pct"] = round(100.0 * available / max(len(group), 1), 3)
            row[f"{modality}_missing_pct"] = round(100.0 * missing / max(len(group), 1), 3)
        row["complete_case"] = int(((group["mrna"] == 1) & (group["mirna"] == 1) & (group["cnv"] == 1)).sum())
        row["complete_case_pct"] = round(100.0 * row["complete_case"] / max(len(group), 1), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def variance_diagnostics(
    raw: RawCohort,
    cfg: DataConfig,
    train_indices: Sequence[int],
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Compute feature variance using training patients only.

    Missing cells within observed rows are median-filled exactly as in
    ``FoldPreprocessor.fit`` before variance is computed.
    """
    train_ids = raw.patient_ids[np.asarray(train_indices, dtype=int)]
    summary_rows = []
    per_feature: Dict[str, pd.DataFrame] = {}

    for modality in MOLECULAR_MODALITIES:
        frame = raw.modalities[modality].loc[train_ids]
        numeric = _numeric_frame(frame)
        row_available = ~numeric.isna().all(axis=1)
        if not bool(row_available.any()):
            raise ValueError(f"No observed training rows for modality '{modality}'.")

        chunk_size = max(1, int(cfg.preprocessing_feature_chunk_size))
        columns = list(numeric.columns)
        variance_parts = []
        for start in range(0, len(columns), chunk_size):
            chunk_cols = columns[start:start + chunk_size]
            block = numeric.loc[row_available, chunk_cols]
            medians = block.median(axis=0).fillna(0.0)
            imputed = block.fillna(medians)
            variance_parts.append(imputed.var(axis=0, ddof=0))
        variances = pd.concat(variance_parts).reindex(columns).fillna(0.0).astype(float)
        threshold = float(cfg.variance_thresholds.get(modality, 0.0))
        passed = variances > threshold
        minimum_count = min(cfg.min_features_per_modality, len(columns))
        selected_by_threshold = int(passed.sum())
        if selected_by_threshold < minimum_count:
            top = set(variances.sort_values(ascending=False).head(minimum_count).index.astype(str))
            selected = pd.Series([str(x) in top for x in variances.index], index=variances.index)
            selection_rule = f"top_{minimum_count}_fallback"
        else:
            selected = passed
            selection_rule = f"variance>{threshold:g}"

        df = pd.DataFrame({
            "feature": variances.index.astype(str),
            "variance": variances.values,
            "threshold": threshold,
            "passes_threshold": passed.values.astype(int),
            "selected": selected.values.astype(int),
        }).sort_values("variance", ascending=False).reset_index(drop=True)
        df["variance_rank"] = np.arange(1, len(df) + 1)
        per_feature[modality] = df

        q = variances.quantile([0, .01, .05, .25, .5, .75, .95, .99, 1.0])
        summary_rows.append({
            "modality": modality,
            "training_patients": int(len(train_ids)),
            "observed_training_patients": int(row_available.sum()),
            "raw_features": int(len(variances)),
            "variance_threshold": threshold,
            "features_passing_threshold": selected_by_threshold,
            "selected_features_final": int(selected.sum()),
            "selection_rule": selection_rule,
            "variance_min": float(q.loc[0.0]),
            "variance_p01": float(q.loc[0.01]),
            "variance_p05": float(q.loc[0.05]),
            "variance_p25": float(q.loc[0.25]),
            "variance_median": float(q.loc[0.5]),
            "variance_p75": float(q.loc[0.75]),
            "variance_p95": float(q.loc[0.95]),
            "variance_p99": float(q.loc[0.99]),
            "variance_max": float(q.loc[1.0]),
        })
    return pd.DataFrame(summary_rows), per_feature


def _annotation_feature_sets(
    mapping: Mapping,
    conservative_mirna_alias_harmonization: bool = True,
) -> Dict[str, set[str]]:
    # Backward-compatible private alias used by older callers/tests.
    return annotation_feature_sets(mapping, conservative_mirna_alias_harmonization)


def annotation_validation(
    raw: RawCohort,
    mapping: Mapping,
    selected_features: Mapping[str, Sequence[str]] | None = None,
    conservative_mirna_alias_harmonization: bool = True,
) -> pd.DataFrame:
    """Validate dataset identifiers against the pathway annotation space."""
    annotation_sets = _annotation_feature_sets(mapping, conservative_mirna_alias_harmonization)
    mirna_raw_annotation = []
    for names in mapping.get("modality_features", {}).get("mirna", {}).values():
        mirna_raw_annotation.extend(map(str, names))
    mirna_aliases = (
        mapping_conservative_mirna_alias_map(mapping)
        if conservative_mirna_alias_harmonization else {}
    )

    rows = []
    for modality in MOLECULAR_MODALITIES:
        raw_names = [str(x) for x in raw.modalities[modality].columns]
        raw_canon = {canonical_feature_name(modality, x) for x in raw_names}
        ann = annotation_sets[modality]
        matched_raw = raw_canon & ann
        selected_names = list(selected_features[modality]) if selected_features is not None else raw_names
        selected_canon = {canonical_feature_name(modality, x) for x in selected_names}
        matched_selected = selected_canon & ann
        row = {
            "modality": modality,
            "dataset_features": len(raw_canon),
            "annotation_features": len(ann),
            "matched_dataset_features": len(matched_raw),
            "dataset_annotation_coverage_pct": round(100.0 * len(matched_raw) / max(len(raw_canon), 1), 3),
            "selected_features": len(selected_canon),
            "matched_selected_features": len(matched_selected),
            "selected_annotation_coverage_pct": round(100.0 * len(matched_selected) / max(len(selected_canon), 1), 3),
        }
        if modality == "mrna":
            numeric = sum(x.isdigit() for x in raw_canon)
            symbol_like = sum((not x.isdigit()) and bool(x) for x in raw_canon)
            ann_numeric = sum(x.isdigit() for x in ann)
            ann_symbol = sum((not x.isdigit()) and bool(x) for x in ann)
            row["dataset_entrez_like_features"] = numeric
            row["dataset_gene_symbol_like_features"] = symbol_like
            row["annotation_entrez_like_features"] = ann_numeric
            row["annotation_gene_symbol_like_features"] = ann_symbol
            if numeric >= 0.8 * max(len(raw_canon), 1):
                row["detected_identifier_mode"] = "entrez"
            elif symbol_like >= 0.8 * max(len(raw_canon), 1):
                row["detected_identifier_mode"] = "gene_symbol"
            else:
                row["detected_identifier_mode"] = "mixed"
        elif modality == "mirna":
            row["dataset_hsa_mirna_like_features"] = sum(x.lower().startswith("hsa-") for x in raw_names)
            row["annotation_hsa_mirna_like_features"] = sum(x.startswith("hsa-") for x in ann)
            row["conservative_aliases_available"] = int(len(mirna_aliases))
            exact_annotation = annotation_feature_sets(mapping, False)["mirna"]
            alias_only_matches = {x for x in selected_canon if x in mirna_aliases and x not in exact_annotation}
            row["selected_features_matched_by_conservative_alias"] = int(len(alias_only_matches))
        rows.append(row)
    return pd.DataFrame(rows)


def unmatched_selected_features(
    mapping: Mapping,
    selected_features: Mapping[str, Sequence[str]],
    conservative_mirna_alias_harmonization: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Return selected features that are absent from the source annotation space."""
    annotation_sets = annotation_feature_sets(mapping, conservative_mirna_alias_harmonization)
    result: Dict[str, pd.DataFrame] = {}
    for modality in MOLECULAR_MODALITIES:
        rows = []
        for feature in selected_features[modality]:
            canonical = canonical_feature_name(modality, feature)
            if canonical not in annotation_sets[modality]:
                rows.append({"feature": str(feature), "canonical_feature": canonical})
        result[modality] = pd.DataFrame(rows, columns=["feature", "canonical_feature"])
    return result

def pathway_coverage_diagnostics(
    mapping: Mapping,
    selected_features: Mapping[str, Sequence[str]],
    cfg: DataConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Summarize fold-specific pathway coverage and size filtering."""
    spec = build_pathway_spec(
        mapping,
        selected_features,
        min_features_per_pathway=cfg.min_features_per_pathway,
        max_features_per_pathway=cfg.max_features_per_pathway,
        max_features_per_pathway_by_modality=cfg.max_features_per_pathway_by_modality,
        conservative_mirna_alias_harmonization=cfg.mirna_conservative_alias_harmonization,
        min_modalities_per_pathway=cfg.min_modalities_per_pathway,
        max_pathways=cfg.max_pathways,
        balance_pathway_sources=cfg.balance_pathway_sources,
    )
    summary_rows = []
    for modality in MOLECULAR_MODALITIES:
        total_selected = len(selected_features[modality])
        mapped = total_selected - len(spec.residual_indices[modality])
        pathways_with_modality = sum(1 for idxs in spec.feature_indices[modality] if idxs)
        summary_rows.append({
            "modality": modality,
            "selected_features": int(total_selected),
            "mapped_selected_features": int(mapped),
            "residual_unmapped_features": int(len(spec.residual_indices[modality])),
            "selected_feature_coverage_pct": round(100.0 * mapped / max(total_selected, 1), 3),
            "retained_pathways_with_modality": int(pathways_with_modality),
            "retained_pathways_total": int(spec.num_pathways),
        })

    pathway_rows = []
    metadata = mapping.get("metadata", {})
    raw_sizes = metadata.get("pathway_gene_sizes", {})
    source_map = metadata.get("pathway_sources", {})
    name_map = metadata.get("pathway_names", {})
    caps = {m: int(cfg.max_features_per_pathway_by_modality.get(m, cfg.max_features_per_pathway)) for m in MOLECULAR_MODALITIES}
    for p_idx, pathway in enumerate(spec.pathway_names):
        counts = {m: len(spec.feature_indices[m][p_idx]) for m in MOLECULAR_MODALITIES}
        eligible = {
            m: (spec.eligible_feature_counts.get(m, [counts[m]] * spec.num_pathways)[p_idx])
            for m in MOLECULAR_MODALITIES
        }
        pathway_rows.append({
            "pathway_id": pathway,
            "pathway_name": name_map.get(pathway, pathway),
            "source": source_map.get(pathway, "Unknown"),
            "source_gene_size": raw_sizes.get(pathway, np.nan),
            "normalized_multimodal_coverage_score": (
                spec.pathway_selection_scores[p_idx] if p_idx < len(spec.pathway_selection_scores) else np.nan
            ),
            "mrna_eligible_before_cap": eligible["mrna"],
            "mirna_eligible_before_cap": eligible["mirna"],
            "cnv_eligible_before_cap": eligible["cnv"],
            "mrna_selected_features": counts["mrna"],
            "mirna_selected_features": counts["mirna"],
            "cnv_selected_features": counts["cnv"],
            "total_selected_features": sum(counts.values()),
            "modalities_with_features": sum(v > 0 for v in counts.values()),
            "graph_degree": int((spec.adjacency[p_idx] > 0).sum()),
            "mrna_feature_cap": caps["mrna"],
            "mirna_feature_cap": caps["mirna"],
            "cnv_feature_cap": caps["cnv"],
            "mrna_at_feature_cap": int(eligible["mrna"] >= caps["mrna"]),
            "mirna_at_feature_cap": int(eligible["mirna"] >= caps["mirna"]),
            "cnv_at_feature_cap": int(eligible["cnv"] >= caps["cnv"]),
            "mrna_truncated_by_cap": int(eligible["mrna"] > counts["mrna"]),
            "mirna_truncated_by_cap": int(eligible["mirna"] > counts["mirna"]),
            "cnv_truncated_by_cap": int(eligible["cnv"] > counts["cnv"]),
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(pathway_rows), spec



def _markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame is None or frame.empty:
        return "_No rows._"
    view = frame.head(max_rows) if max_rows is not None else frame
    headers = [str(c) for c in view.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in view.iterrows():
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append(f"{value:.4g}")
            else:
                vals.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    if max_rows is not None and len(frame) > max_rows:
        lines.append(f"\n_First {max_rows} of {len(frame)} rows shown._")
    return "\n".join(lines)


def supplementary_markdown_report(
    raw: RawCohort,
    cfg: DataConfig,
    pattern_summary: pd.DataFrame,
    cancer_missingness: pd.DataFrame,
    variance_summary: pd.DataFrame | None = None,
    annotation_summary: pd.DataFrame | None = None,
    pathway_summary: pd.DataFrame | None = None,
) -> str:
    meta = raw.metadata
    clinical_qc = meta.get("clinical_qc", {})
    n = len(raw)
    events = int(pd.to_numeric(raw.outcomes[cfg.event_col], errors="coerce").fillna(0).sum()) if cfg.event_col in raw.outcomes else 0
    censored = n - events
    lines = [
        "# Supplementary Data Quality-Control Report",
        "",
        "This report was generated automatically by PathTokenSurv v1.4. Values describe the real cohort loaded by the configured raw-data pipeline. Fold-specific feature statistics are computed from the designated training partition only.",
        "",
        "## Cohort",
        "",
        f"- Clinical records before eligibility filtering: {clinical_qc.get('clinical_rows', 'NA')}",
        f"- Eligible patients retained: {n}",
        f"- Cancer types: {raw.outcomes[cfg.cancer_col].astype(str).nunique()}",
        f"- Observed events: {events}",
        f"- Right-censored observations: {censored}",
        "",
        "## Natural modality availability",
        "",
    ]
    for m in MOLECULAR_MODALITIES:
        available = int((~raw.modalities[m].isna().all(axis=1)).sum())
        missing = n - available
        lines.append(f"- {m}: {available} available ({100*available/max(n,1):.2f}%), {missing} missing ({100*missing/max(n,1):.2f}%).")
    lines += ["", "### Exact availability patterns", "", _markdown_table(pattern_summary), ""]
    lines += ["## Cancer-specific missingness", "", _markdown_table(cancer_missingness), ""]
    if variance_summary is not None:
        lines += ["## Training-fold feature variance diagnostics", "", _markdown_table(variance_summary), ""]
    if annotation_summary is not None:
        lines += ["## Identifier and annotation validation", "", _markdown_table(annotation_summary), ""]
    if pathway_summary is not None:
        lines += ["## Fold-specific pathway coverage", "", _markdown_table(pathway_summary), ""]
    lines += [
        "## Reproducibility note",
        "",
        "Feature selection, imputation statistics, scaling parameters, categorical vocabularies, and fold-specific pathway intersections must be learned from training patients only. The validation and test partitions must not contribute to these quantities.",
        "",
    ]
    return "\n".join(lines)


def save_qc_json(payload: Mapping, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
