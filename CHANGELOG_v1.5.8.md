# PathTokenSurv v1.5.8 Changelog

## Scope

Evaluation-only missingness-robustness aggregation. The frozen primary training protocol and v1.5.7 model/evaluation definitions are unchanged.

## Added

- `pathtokensurv/missingness_aggregation.py`
- `scripts/aggregate_missingness_robustness.py`
- five-fold aggregation of natural complete/incomplete and exact availability-pattern metrics
- five-fold aggregation of controlled complete-case modality-drop metrics
- foldwise controlled degradation metrics with a sign convention in which positive always means worse after modality removal
- complete-vs-incomplete observational gap summaries with explicit non-causal labeling
- consistency checks for exact-pattern counts, identical controlled complete-case cohorts, and all-observed vs natural-complete metrics
- publication-oriented CSV and Markdown summary outputs
- unit tests for aggregation and consistency checking

## Unchanged

- TCGA cohort definition and outer splits
- feature selection
- KEGG/Reactome pathway construction
- TarBase v9 mapping
- PathTokenSurv architecture
- training losses and optimization
- early-stopping rule
- held-out test predictions and v1.5.7 scientific evaluation definitions
