# PathTokenSurv v1.5.7 — Frozen Evaluation Amendment

## Purpose

Version 1.5.7 is an **evaluation-only scientific amendment** applied after completion of outer fold 1 and before launching outer folds 2–5. The primary PathTokenSurv architecture, preprocessing, pathway construction, optimizer, loss weights, model seed logic, validation split logic, checkpoint-selection rule, maximum epochs, and early-stopping patience are unchanged from v1.5.6.

The amendment was introduced because pooled pan-cancer Harrell concordance permits cross-cancer patient pairs. Since cancer type is an explicit survival-conditioning variable, pooled concordance can partly measure between-cancer survival separation rather than only patient-level ranking within cancer type. All completed and future folds are therefore evaluated with both pooled and cancer-restricted discrimination metrics.

## Frozen reporting rules

1. **Checkpoint selection remains unchanged:**
   \(Q_{val}=C_{val}-IBS_{val}\), using the same pooled validation C-index and IBS implementation as v1.5.6. No test-set metric is used for early stopping.
2. **Pooled pan-cancer C-index** is retained for continuity and reported as pan-cancer pooled discrimination.
3. **Cancer-stratified C-index** is mandatory for scientific interpretation. Comparable pairs are formed only within the same cancer type and concordant/comparable counts are pooled across cancers.
4. **Macro per-cancer C-index** is reported as an unweighted descriptive summary across cancer strata with at least one comparable pair.
5. **Cross-cancer concordance and the within-cancer comparable-pair fraction** are saved to quantify how strongly pooled concordance depends on cross-cancer comparisons.
6. **Time-dependent AUC** is reported both pooled and cancer-stratified. Cancer-stratified AUC uses cancer-specific training censoring distributions and permits case-control comparisons only within cancer type.
7. **IPCW calibration** is summarized by a censoring-weighted expected calibration error (IPCW-ECE) at each supported horizon and by a time-integrated IPCW-ECE.
8. **Natural missingness evaluation** reports exact test-set availability patterns plus complete versus incomplete cases.
9. **Controlled modality-drop evaluation** uses the same naturally complete held-out patients for all conditions: all observed, drop mRNA, drop miRNA, drop CNV, and clinical-only.
10. **Cancer-only and clinical-only baselines** use the exact primary train/validation/test assignments, the same fitted clinical preprocessing, the same survival time discretizer, the same optimizer settings, and the same checkpoint-selection rule. The clinical-only baseline includes the frozen clinical covariates and cancer-conditioned survival head; the cancer-only baseline contains no molecular or clinical patient-level predictors beyond cancer type.

## No retraining requirement for fold 1

The existing v1.5.6 fold-1 checkpoint remains valid. `scripts/evaluate_outer_fold_extended.py` computes the amended metrics and missingness analyses from the saved checkpoint and frozen split without updating any model parameter. Fold 1 must **not** be retrained merely to adopt this evaluation amendment.

## Reproducibility outputs

Each completed fold should contain:

- `test_metrics.json`
- `test_metrics_extended.json`
- `test_predictions_extended.csv` (includes the common supported-horizon evaluation risk score)
- `stratified_discrimination.json`
- `stratified_auc_by_horizon.csv`
- `per_cancer_metrics.csv`
- `calibration_ipcw.csv`
- `calibration_ipcw_summary.csv`
- `calibration_ipcw_summary.json`
- `scientific_evaluation_summary.json`
- `natural_missingness_metrics.csv`
- `complete_case_modality_drop_metrics.csv`
- optional `baselines/cancer_only/` and `baselines/clinical_only/` directories

This amendment is frozen before outer folds 2–5. Later changes must be labeled as sensitivity analyses or additional post-hoc analyses rather than silently replacing these rules.
