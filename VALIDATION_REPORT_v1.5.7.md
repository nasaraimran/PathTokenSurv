# PathTokenSurv v1.5.7 Validation Report

## Automated tests

- `PYTHONPATH=. pytest -q`
- Result during release validation: **28 passed**.

New tests verify:

- cancer-stratified Harrell concordance excludes cross-cancer pairs;
- within-cancer and pooled comparable-pair counts are reported correctly;
- IPCW calibration error is weighted by censoring-adjusted bin mass;
- time-integrated IPCW-ECE behaves correctly on a known synthetic example.

## Backward metric equivalence

The v1.5.6 and v1.5.7 metric implementations were evaluated on the same randomized survival data. Every pre-existing metric key (Harrell C-index, IBS, supported horizon, all time-dependent AUCs, and the existing calibration diagnostic) matched to floating-point precision. v1.5.7 adds only the mean tdAUC summary to that function.

## End-to-end outer-fold smoke test

A two-epoch synthetic outer-fold run completed successfully with:

- frozen train/validation/test splitting;
- model optimization and checkpoint selection;
- held-out prediction;
- pooled and cancer-stratified discrimination;
- cancer-stratified AUC;
- IPCW calibration summaries;
- natural missingness groups;
- complete-case modality-drop evaluation.

## Baseline smoke test

Both frozen-split baselines completed successfully on the same synthetic outer-fold artifacts:

- cancer-only discrete-time survival baseline;
- clinical-only + cancer-conditioned discrete-time survival baseline.

The synthetic numerical metrics are software-validation outputs only and must not be used as study results.

## Training-protocol integrity

The model architecture, tokenizer, fusion, survival head, loss functions, trainer, preprocessing, pathway builder, TCGA loader, and training hyperparameters are unchanged from v1.5.6. `run_outer_fold.py` enables the new robustness evaluations only after primary training and held-out prediction are complete.
