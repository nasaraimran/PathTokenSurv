# PathTokenSurv v1.4

Version 1.4 adds a pre-training quality-control and reproducibility layer for the real TCGA PanCancer experiment.

## Added

- Exact patient-level modality availability matrix and all natural availability-pattern counts.
- Cancer-specific mRNA, miRNA, and CNV missingness summaries.
- UpSet-ready patient-level CSV and cancer-specific pattern-count CSV.
- Training-fold-only feature variance diagnostics, including per-feature variance files and distribution plots.
- Annotation validation for mRNA Entrez IDs, CNV gene symbols, and mature miRNA identifiers.
- Fold-specific pathway coverage diagnostics and residual-feature counts.
- Optional source pathway-size filtering in `build_pathways.py` via `--min-genes-per-pathway` and `--max-genes-per-pathway`.
- `min_modalities_per_pathway` for fold-specific pathway retention.
- miRNA-target and pathway-annotation metadata in generated `pathways.json`.
- `scripts/run_tcga_qc.py` for a complete supplementary QC bundle.
- `scripts/dry_run_preprocessing.py` for CPU-only preprocessing, pathway construction, time discretization, model construction, and one forward-pass validation.
- Automatic `SUPPLEMENTARY_QC_REPORT.md` generation.
- QC plots for modality availability, missingness patterns, cancer-specific missingness, feature variance, and pathway coverage.
- Identifier canonicalization during pathway matching while retaining original feature labels.

## Changed

- Fold preprocessing computes median-imputed feature variances in configurable feature chunks to reduce peak memory usage on the 20,530-feature mRNA and 24,776-feature CNV matrices.
- `inspect_tcga_raw.py` now includes exact natural availability patterns and cancer-specific missingness when all modalities are loaded.
- Package version bumped to 0.1.4.

## Validation

- 10 unit tests pass.
- End-to-end synthetic training and inference pass.
- Detailed QC script passes on synthetic multimodal data.
- CPU dry-run preprocessing and model forward pass pass on synthetic multimodal data.
