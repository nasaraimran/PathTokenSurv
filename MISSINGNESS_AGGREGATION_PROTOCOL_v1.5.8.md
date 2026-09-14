# PathTokenSurv v1.5.8 — Missingness Robustness Aggregation Protocol

## Purpose

v1.5.8 is an **evaluation-only** amendment. It aggregates the natural-missingness and controlled complete-case modality-drop outputs already produced for each held-out outer fold by v1.5.7. It performs no model fitting, checkpoint selection, preprocessing, pathway construction, or hyperparameter changes.

## Inputs per outer fold

- `natural_missingness_metrics.csv`
- `complete_case_modality_drop_metrics.csv`
- `experiment_manifest.json` (used to recover the outer-fold number when present)

## Natural missingness

The script preserves both exact modality-availability patterns and the aggregate `complete` / `incomplete` groups. Cross-fold mean, SD, and two-sided 95% Student-t intervals are computed for pooled C-index, cancer-stratified C-index, IBS, and mean time-dependent AUC.

Complete and incomplete groups contain different patients. Therefore, complete-versus-incomplete differences are explicitly labeled **observational fold gaps** and must not be interpreted as causal degradation caused by removing assays.

## Controlled complete-case modality dropout

The conditions are:

- `all_observed`
- `drop_mrna`
- `drop_mirna`
- `drop_cnv`
- `clinical_only` (all molecular modalities hidden at inference)

Within each fold, all conditions must use the same naturally complete held-out patients and events. The script also checks that `all_observed` reproduces the natural complete-case metrics.

For publication-facing summaries, degradation is defined so that positive values always mean worse performance after modality removal:

- C-index degradation = all-observed C-index − condition C-index
- stratified C-index degradation = all-observed stratified C-index − condition stratified C-index
- IBS degradation = condition IBS − all-observed IBS
- tdAUC degradation = all-observed tdAUC − condition tdAUC

## Cross-fold intervals

All reported 95% intervals are two-sided Student-t intervals across outer-fold point estimates. They are not patient-level bootstrap intervals and are not multiple-seed ensemble confidence intervals.

## Outputs

- `natural_missingness_fold_metrics.csv`
- `natural_missingness_summary.csv`
- `natural_complete_vs_incomplete_fold_gaps.csv`
- `natural_complete_vs_incomplete_gap_summary.csv`
- `complete_case_modality_drop_fold_metrics.csv`
- `complete_case_modality_drop_summary.csv`
- `missingness_robustness_publication_table.csv`
- `MISSINGNESS_ROBUSTNESS_SUMMARY.md`
- `missingness_aggregation_manifest.json`
