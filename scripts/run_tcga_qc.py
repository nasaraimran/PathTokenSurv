from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathtokensurv.config import ExperimentConfig
from pathtokensurv.data.raw import load_raw_cohort
from pathtokensurv.data.splitting import outer_folds
from pathtokensurv.data.preprocessing import FoldPreprocessor
from pathtokensurv.data.pathways import load_pathway_mapping
from pathtokensurv.data.qc import (
    patient_availability_table,
    modality_pattern_summary,
    cancer_specific_missingness,
    variance_diagnostics,
    annotation_validation,
    unmatched_selected_features,
    pathway_coverage_diagnostics,
    supplementary_markdown_report,
)


def _save_plots(output_dir: Path, availability: pd.DataFrame, patterns: pd.DataFrame,
                cancer_missing: pd.DataFrame, variance_tables: dict[str, pd.DataFrame],
                pathway_summary: pd.DataFrame | None) -> None:
    # Overall modality availability.
    labels = ["Clinical", "mRNA", "miRNA", "CNV"]
    cols = ["clinical", "mrna", "mirna", "cnv"]
    values = [float(availability[c].mean() * 100.0) for c in cols]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(labels, values)
    ax.set_ylabel("Availability (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Natural modality availability")
    for i, value in enumerate(values):
        ax.text(i, value + 1.2, f"{value:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "modality_availability.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Exact availability patterns.
    view = patterns.sort_values("patients", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.55 * len(view))))
    ax.barh(view["pattern"], view["patients"])
    ax.set_xlabel("Patients")
    ax.set_title("Natural modality-availability patterns")
    fig.tight_layout()
    fig.savefig(output_dir / "availability_patterns.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Cancer-specific molecular missingness heatmap.
    heat_cols = ["mrna_missing_pct", "mirna_missing_pct", "cnv_missing_pct"]
    heat = cancer_missing.set_index("cancer_type")[heat_cols]
    fig, ax = plt.subplots(figsize=(7, max(7, 0.28 * len(heat))))
    im = ax.imshow(heat.values, aspect="auto")
    ax.set_xticks(range(3), labels=["mRNA", "miRNA", "CNV"])
    ax.set_yticks(range(len(heat)), labels=heat.index)
    ax.set_title("Cancer-specific molecular missingness (%)")
    fig.colorbar(im, ax=ax, label="Missing (%)")
    fig.tight_layout()
    fig.savefig(output_dir / "cancer_specific_missingness.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Training-fold feature variance diagnostics.
    for modality, frame in variance_tables.items():
        vals = frame["variance"].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.hist(finite, bins=60)
        threshold = float(frame["threshold"].iloc[0])
        ax.axvline(threshold, linestyle="--", linewidth=1.5, label=f"threshold = {threshold:g}")
        ax.set_xlabel("Training-fold variance")
        ax.set_ylabel("Features")
        ax.set_title(f"{modality} feature-variance distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"variance_{modality}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    if pathway_summary is not None and not pathway_summary.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.bar(pathway_summary["modality"], pathway_summary["selected_feature_coverage_pct"])
        ax.set_ylabel("Selected features mapped to retained pathways (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Fold-specific pathway coverage")
        fig.tight_layout()
        fig.savefig(output_dir / "pathway_coverage.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate detailed TCGA data, missingness, variance, annotation, and pathway QC outputs."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="outputs/tcga_qc_v1.4")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=1, help="1-based outer fold used for training-only variance/pathway QC.")
    parser.add_argument("--skip-pathways", action="store_true", help="Run raw-data and variance QC even if pathways.json is not ready.")
    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading raw cohort...")
    raw = load_raw_cohort(cfg.data, require_outcomes=True)
    availability = patient_availability_table(raw, cfg.data)
    patterns = modality_pattern_summary(availability)
    cancer_missing = cancer_specific_missingness(availability)

    availability.to_csv(out / "upset_ready_patient_availability.csv", index=False)
    patterns.to_csv(out / "modality_pattern_summary.csv", index=False)
    cancer_missing.to_csv(out / "cancer_specific_missingness.csv", index=False)

    cancer_pattern = (
        availability.groupby(["cancer_type", "mask", "pattern"], dropna=False)
        .size().rename("patients").reset_index()
    )
    cancer_pattern["percent_within_cancer"] = cancer_pattern.groupby("cancer_type")["patients"].transform(
        lambda x: 100.0 * x / x.sum()
    )
    cancer_pattern.to_csv(out / "cancer_specific_pattern_counts.csv", index=False)

    cancers = raw.outcomes[cfg.data.cancer_col].astype(str).to_numpy()
    events = raw.outcomes[cfg.data.event_col].astype(int).to_numpy()
    folds = list(outer_folds(cancers, events, n_splits=args.outer_folds, seed=cfg.training.seed))
    if not 1 <= args.fold <= len(folds):
        raise ValueError(f"--fold must be between 1 and {len(folds)}")
    train_idx, test_idx = folds[args.fold - 1]
    np.savetxt(out / "qc_training_indices.txt", train_idx, fmt="%d")
    np.savetxt(out / "qc_test_indices.txt", test_idx, fmt="%d")

    print(f"Computing training-fold feature variance diagnostics for fold {args.fold}...")
    variance_summary, variance_tables = variance_diagnostics(raw, cfg.data, train_idx)
    variance_summary.to_csv(out / "variance_summary.csv", index=False)
    for modality, frame in variance_tables.items():
        frame.to_csv(out / f"variance_features_{modality}.csv", index=False)

    preprocessor = FoldPreprocessor(cfg.data).fit(raw, train_idx)
    selected_features = {
        modality: preprocessor.molecular_transforms[modality].selected_features
        for modality in preprocessor.molecular_transforms
    }
    pd.DataFrame([
        {"modality": modality, "selected_features": len(names), "threshold": cfg.data.variance_thresholds.get(modality, 0.0)}
        for modality, names in selected_features.items()
    ]).to_csv(out / "selected_feature_counts.csv", index=False)

    annotation_summary = None
    pathway_summary = None
    pathway_sizes = None
    pathway_meta = None
    pathway_path = Path(cfg.data.data_dir) / cfg.data.pathway_file
    if not args.skip_pathways:
        if not pathway_path.exists():
            raise FileNotFoundError(
                f"Pathway mapping not found: {pathway_path}. Build it first or rerun with --skip-pathways."
            )
        print("Validating molecular identifiers and pathway coverage...")
        mapping = load_pathway_mapping(pathway_path)
        annotation_summary = annotation_validation(
            raw, mapping, selected_features,
            conservative_mirna_alias_harmonization=cfg.data.mirna_conservative_alias_harmonization,
        )
        annotation_summary.to_csv(out / "annotation_validation.csv", index=False)
        unmatched = unmatched_selected_features(
            mapping, selected_features,
            conservative_mirna_alias_harmonization=cfg.data.mirna_conservative_alias_harmonization,
        )
        for modality, frame in unmatched.items():
            frame.to_csv(out / f"unmatched_selected_features_{modality}.csv", index=False)
        pathway_summary, pathway_sizes, spec = pathway_coverage_diagnostics(mapping, selected_features, cfg.data)
        pathway_summary.to_csv(out / "pathway_coverage_summary.csv", index=False)
        pathway_sizes.to_csv(out / "retained_pathway_sizes.csv", index=False)
        spec.save(out / "qc_pathway_spec.json")
        pathway_meta = mapping.get("metadata", {})
        (out / "pathway_mapping_metadata.json").write_text(json.dumps(pathway_meta, indent=2), encoding="utf-8")

    report = supplementary_markdown_report(
        raw, cfg.data, patterns, cancer_missing, variance_summary,
        annotation_summary, pathway_summary,
    )
    if pathway_meta:
        report += "\n## Pathway source annotation\n\n"
        for key in [
            "source_unique_pathways", "retained_pathways_after_size_filter", "pathways_removed_by_size_filter",
            "min_genes_per_pathway", "max_genes_per_pathway", "unique_gene_symbols", "unique_entrez_ids",
            "mirna_target_rows", "unique_mirnas", "unique_target_genes",
            "target_genes_overlapping_retained_pathways", "retained_pathway_edges",
            "mrna_identifier_alias_mode", "mirna_pathway_mode", "mirna_min_target_overlap",
            "mirna_enrichment_fdr", "mirna_pathway_candidate_links", "mirna_pathway_retained_links",
            "mirna_pathways_with_links", "mirna_pathway_links_median_per_pathway",
            "mirna_pathway_links_max_per_pathway", "mirna_enrichment_score_definition",
            "mirna_conservative_alias_count",
        ]:
            if key in pathway_meta:
                report += f"- {key}: {pathway_meta[key]}\n"
    (out / "SUPPLEMENTARY_QC_REPORT.md").write_text(report, encoding="utf-8")

    summary_payload = {
        "version": "1.5.5",
        "source_metadata": raw.metadata,
        "eligible_patients": len(raw),
        "outer_folds": args.outer_folds,
        "qc_fold": args.fold,
        "qc_training_patients": int(len(train_idx)),
        "qc_test_patients": int(len(test_idx)),
        "modality_patterns": patterns.to_dict(orient="records"),
        "variance_summary": variance_summary.to_dict(orient="records"),
        "annotation_validation": None if annotation_summary is None else annotation_summary.to_dict(orient="records"),
        "pathway_coverage": None if pathway_summary is None else pathway_summary.to_dict(orient="records"),
    }
    (out / "qc_summary.json").write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")

    _save_plots(out, availability, patterns, cancer_missing, variance_tables, pathway_summary)

    print("\nExact modality patterns:")
    print(patterns[["pattern", "patients", "percent"]].to_string(index=False))
    print("\nTraining-fold variance summary:")
    print(variance_summary[["modality", "raw_features", "variance_threshold", "features_passing_threshold", "selected_features_final"]].to_string(index=False))
    if annotation_summary is not None:
        print("\nAnnotation validation:")
        print(annotation_summary.to_string(index=False))
        failures = []
        for _, row in annotation_summary.iterrows():
            modality = str(row["modality"])
            threshold = float(cfg.data.min_selected_annotation_coverage_pct.get(modality, 0.0))
            coverage = float(row["selected_annotation_coverage_pct"])
            if coverage < threshold:
                failures.append(f"{modality}={coverage:.2f}% < {threshold:.2f}%")
        if failures:
            print("\nQC WARNING: selected-feature annotation coverage below training guard: " + "; ".join(failures))
        else:
            print("\nAnnotation coverage guard: PASS")
    print(f"\nSaved detailed QC package to: {out}")


if __name__ == "__main__":
    main()
