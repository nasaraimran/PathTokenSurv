# PathTokenSurv v1.5.8 Validation Report

## Validation scope

v1.5.8 adds only cross-fold aggregation of missingness-robustness outputs. No training code or scientific primary-model configuration was changed.

## Automated checks

- full existing pytest suite
- new aggregation-output test
- controlled-dropout sign-convention test
- natural complete/incomplete gap test
- all-observed vs natural-complete consistency-failure test
- CLI smoke test using the real v1.5.7 fold-level CSV schema cloned across five fold IDs

## Invariants enforced

1. Every requested fold must contain natural-missingness and complete-case dropout CSVs.
2. Fold IDs must be unique.
3. Natural exact-pattern patient counts must sum to complete + incomplete counts.
4. Controlled dropout conditions must contain identical patient/event counts within a fold.
5. Controlled `all_observed` metrics must reproduce the natural complete-case metrics within numerical tolerance.
6. Positive controlled degradation values always indicate worse performance after modality removal.

## Interpretation guardrail

Natural complete-versus-incomplete differences are observational because the groups contain different patients. Controlled complete-case dropout is the appropriate paired test-time robustness analysis.
