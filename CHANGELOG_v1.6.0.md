# PathTokenSurv v1.6.0 Changelog

## Scope

v1.6.0 adds a **secondary pathway-tokenization sensitivity control**. It does not change the frozen primary model, completed primary five-fold results, clinical/cancer baselines, v1.5.8 missingness analysis, or v1.5.9 architecture-ablation results.

## Added

- `pathtokensurv/pathway_control.py`
  - deterministic modality-specific feature-label permutation;
  - no-fixed-point permutation for real molecular feature sets;
  - structural invariant validation;
  - paired five-fold pathway-control aggregation.
- `scripts/run_pathway_token_control_fold.py`
  - one-fold frozen-artifact control training;
  - exact primary split/preprocessor/time discretizer/model seed reuse;
  - SHA-256 provenance and permutation metadata;
  - automatic extended scientific and missingness evaluation.
- `scripts/run_pathway_token_control.ps1`
  - restart-safe folds 1–5 runner;
  - skips completed `PASS` folds;
  - resumes interrupted folds from `last_model.pt`;
  - automatic final aggregation.
- `scripts/aggregate_pathway_token_control.py`
  - paired five-fold summary and publication table.
- `tests/test_pathway_control_v160.py`
  - deterministic permutation tests;
  - preservation of token size/overlap/degree/residual structure;
  - five-fold aggregation checks.
- `PATHWAY_TOKENIZATION_CONTROL_PROTOCOL_v1.6.0.md`.

## Frozen-experiment support

`run_frozen_artifact_ablation()` now accepts an optional `pathway_spec_override`. Existing v1.5.9 architecture ablations leave this unset and behave exactly as before. v1.6.0 uses the override only for the controlled feature-permuted pathway specification.

## Control definition

For each modality and fold, a deterministic bijection relabels selected molecular feature indices everywhere they occur in pathway memberships and the residual token. The feature tensor itself is unchanged. This preserves pathway-token structure while breaking biological feature-to-pathway correspondence.

Default fold permutation seeds are `160001` through `160005`.

## Evaluation status

The control is explicitly labeled a **secondary sensitivity analysis**, because it was motivated after inspecting the prespecified v1.5.9 architecture-ablation results.
