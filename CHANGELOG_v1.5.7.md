# PathTokenSurv v1.5.7 Changelog

## Scope

v1.5.7 is an **evaluation-only amendment**. It does not change the v1.5.6 model, pathway specification, preprocessing, optimizer, survival loss, reconstruction/consistency objectives, seeds, outer-fold construction, validation split, or early-stopping rule.

## Added

- Cancer-stratified Harrell C-index with within-cancer comparable-pair pooling.
- Macro per-cancer C-index, cross-cancer C-index, and within-cancer pair fraction.
- Cancer-stratified cumulative/dynamic AUC using cancer-specific training censoring references.
- Mean time-dependent AUC in standard survival metric output.
- IPCW-ECE by horizon plus mean, median, maximum, and time-integrated IPCW-ECE summaries.
- `test_metrics_extended.json` and `scientific_evaluation_summary.json`.
- `evaluation_risk_score` on the common supported horizon, saved without removing the legacy full-grid `risk_score`.
- Natural held-out missingness-pattern evaluation.
- Complete-case controlled modality-drop evaluation.
- `scripts/evaluate_outer_fold_extended.py` for retroactive evaluation of completed v1.5.6 folds without retraining.
- Cancer-only and clinical-only frozen-split baselines via `scripts/run_fold_baselines.py`.
- `scripts/run_remaining_outer_folds.ps1` for sequential folds 2–5 on Windows/CUDA.
- `scripts/aggregate_outer_folds.py` for cross-fold summaries and Student-t confidence intervals.
- `EVALUATION_AMENDMENT_v1.5.7.md` documenting the frozen reporting rules.

## Compatibility

The existing Harrell C-index, IBS, time-dependent AUC, and calibration diagnostic values are numerically unchanged for the same inputs. The standard metric dictionary only gains `mean_time_dependent_auc`. The v1.5.6 checkpoint-selection rule remains `validation C-index - validation IBS`.
