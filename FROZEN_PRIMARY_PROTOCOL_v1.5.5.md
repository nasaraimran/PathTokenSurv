# PathTokenSurv v1.5.5 — Frozen Primary Protocol

This file marks the data/pathway construction pipeline as frozen for the primary TCGA PanCancer survival experiment after the v1.5.4 pathway QC and v1.5.4 real-data CPU dry-run passed.

## Primary cohort and split

- Eligible TCGA patients: 10,986.
- Outer validation: 5 folds, jointly stratified by cancer type and survival event where feasible.
- Primary development run: outer fold 1 only.
- The outer test partition is untouched during preprocessing, feature selection, pathway construction, time discretization, optimization, and early stopping.
- Within the outer-training partition, 15% is reserved for early stopping using cancer/event stratification where feasible.
- Split seed is inherited from the primary TCGA configuration (`training.seed = 123`).
- Model seed for outer fold `f` is `123 + 1009*f`.

## Frozen preprocessing and pathway rules

- mRNA variance threshold: > 7.0, training partition only.
- miRNA: all features retained before pathway mapping.
- CNV variance threshold: > 0.2, training partition only.
- Pathway sources: KEGG + Reactome.
- miRNA targets: DIANA-TarBase v9 human experimentally supported interactions.
- miRNA-to-pathway mapping: target-set enrichment, minimum overlap 3, BH FDR <= 0.05.
- mRNA identifiers: gene symbol or Entrez alias.
- miRNA aliases: arm-unspecified names are mapped only when exactly one mature arm exists in the complete annotation.
- Pathway source gene-size filter: 5–500 genes.
- Maximum retained pathway tokens: 256, balanced 128 KEGG + 128 Reactome when sufficient eligible pathways exist.
- Fold-specific pathway ranking: outcome-independent normalized multimodal coverage.
- Per-pathway feature caps: mRNA 128, miRNA 32, CNV 128.
- miRNA cap uses enrichment score `-log10(BH-adjusted q-value)` with target-overlap and miRNA identifier tie breaking.
- Residual tokens preserve selected molecular features not mapped into the retained pathway set.

## Frozen model/training settings

- `d_model = 128`
- attention heads = 8
- intramodal layers = 1
- fusion layers = 2
- reconstruction layers = 1
- dropout = 0.15
- discrete survival bins = 20 (subject to merging of duplicate quantile boundaries)
- batch size = 8
- maximum epochs = 200
- early-stopping patience = 20
- AdamW learning rate = 1e-4
- weight decay = 1e-5
- gradient clipping = 5.0
- cosine learning-rate schedule
- structured modality-drop probability = 0.35
- reconstruction loss weight = 0.25
- subset-consistency loss weight = 0.10
- cancer-baseline regularization weight = 0.01

## Primary metrics

The held-out outer test fold is evaluated with Harrell C-index, IPCW integrated Brier score, cumulative/dynamic time-dependent AUC, IPCW calibration diagnostics, and cancer-specific metrics. Patient-level predictions and exact split assignments are retained.

Any later alteration of these frozen preprocessing/pathway rules must be labeled as an ablation, sensitivity analysis, or new protocol version rather than replacing the primary analysis silently.
