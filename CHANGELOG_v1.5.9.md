# PathTokenSurv v1.5.9 Changelog

## Scope

Frozen architecture ablation suite. No change to the completed v1.5.6/v1.5.7 primary model results, TCGA preprocessing rules, pathway construction, outer-fold definition, or primary checkpoint-selection rule.

## Added

- `pathtokensurv/ablation.py`
  - five prespecified architecture/training ablation definitions;
  - paired degradation convention;
  - five-fold ablation aggregation with Student-t intervals.
- `pathtokensurv/frozen_ablation_experiment.py`
  - trains ablations from the exact primary fold preprocessing, pathway, and time-discretization artifacts.
- `scripts/run_frozen_ablation_fold.py`
  - one-fold frozen ablation runner;
  - reuses exact primary patient assignments and primary model seed;
  - records frozen artifact SHA-256 hashes and provenance.
- `scripts/run_architecture_ablations.ps1`
  - sequential 25-run Windows launcher;
  - skips completed runs;
  - resumes interrupted runs from `last_model.pt` by default;
  - aggregates automatically when all requested runs are complete.
- `scripts/aggregate_architecture_ablations.py`
  - generates fold-level, paired-difference, summary, publication-table, and manifest artifacts.
- `ARCHITECTURE_ABLATION_PROTOCOL_v1.5.9.md`.
- `tests/test_ablation_v159.py`.

## Prespecified ablations

1. no structured modality masking;
2. no reconstruction objective;
3. no subset-consistency objective;
4. no biological pathway-attention bias;
5. no cancer conditioning.

## Scientific controls

- The already completed primary folds are the full-model reference and are not retrained.
- Every ablation fold uses exactly the corresponding primary split, preprocessor, pathway specification, time bins, and model seed.
- Positive paired degradation always means the ablation is worse.
- All five ablations must be reported; results are not used to retroactively modify the primary model.
