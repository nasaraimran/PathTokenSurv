# PathTokenSurv v1.6.0 — Pathway-Tokenization Control Protocol

## Purpose

v1.6.0 adds one **secondary pathway-tokenization sensitivity analysis** after completion of the frozen five-fold primary experiment and the prespecified v1.5.9 architecture-ablation suite. It does not alter the primary PathTokenSurv model or retroactively change any reported primary result.

The control asks a narrower and more fundamental question than the v1.5.9 `no_pathway_bias` ablation:

> Does assigning molecular features to their correct biological pathway tokens provide predictive value beyond an equally structured but biologically incorrect tokenization?

The v1.5.9 `no_pathway_bias` model retained the correct pathway-token memberships and removed only the explicit same-pathway/Reactome attention bias. Therefore it could not test the biological feature-to-pathway assignment itself.

## Control construction

For each frozen outer fold, v1.6.0 loads the exact primary `pathway_spec.json`. Separately within mRNA, miRNA, and CNV, it samples one deterministic feature-label permutation. The fold-specific default permutation seed is:

```text
160000 + outer_fold
```

Every occurrence of selected feature index `i` in pathway memberships and in the residual token is replaced by the same permuted index `pi(i)`. Feature tensors themselves are not reordered. The permutation is constructed as a bijection with no fixed points when the modality contains more than one selected feature.

This preserves exactly:

- the 256 pathway tokens and their order;
- pathway names and Reactome/KEGG provenance;
- pathway adjacency/graph structure;
- the number of molecular features assigned to every pathway token;
- the pairwise pathway-token overlap matrix;
- the molecular feature membership-degree distribution;
- the residual-token size and its structural relation to mapped features;
- the selected feature universe and feature tensor order;
- model architecture and parameter count;
- outer train/validation/test patient assignments;
- frozen preprocessing and time discretization;
- all primary training hyperparameters and losses;
- the primary fold's exact model-initialization seed.

It deliberately breaks only the biological correspondence between a molecular feature and the pathway identity to which that feature contributes. The control does not use survival time, event status, cancer outcome, or any other target information to construct the permutation.

## Interpretation

The comparison is:

```text
Frozen primary PathTokenSurv
  correct biological feature -> pathway assignment

versus

Feature-permuted pathway-token control
  identical token structure, incorrect feature -> pathway assignment
```

The experiment is **not** a no-pathway model and should not be described as one. Pathway labels, token topology, pathway adjacency, and model capacity remain present in the control. The analysis tests whether the correct biological assignment of features to those tokens matters.

Because this control was motivated after inspection of the prespecified architecture ablations, it must be labeled a **secondary sensitivity analysis** in the manuscript rather than a prespecified primary ablation.

## Reproducibility and leakage controls

For each fold the runner reuses the completed primary fold's:

```text
split_assignments.csv
preprocessor.pkl
time_discretizer.json
model seed
```

and records SHA-256 hashes for the frozen primary artifacts plus the generated control pathway specification. It also saves:

```text
primary_pathway_spec.json
pathway_spec_permuted.json
pathway_spec.json
pathway_permutation_metadata.json
pathway_feature_permutation_indices.json
pathway_control_manifest.json
```

Before training, invariants are checked programmatically. Training aborts if token sizes, pairwise overlap topology, feature-degree distribution, residual size, pathway order, feature order, pathway adjacency, eligible counts, or pathway-selection scores change.

## Five-fold analysis

The primary models are not retrained. One feature-permuted control is trained on each of the five frozen outer folds with the exact corresponding primary model seed.

The endpoints are paired by outer fold:

- pooled Harrell C-index;
- cancer-stratified Harrell C-index;
- IBS;
- mean time-dependent AUC;
- mean cancer-stratified time-dependent AUC;
- integrated IPCW-ECE.

Positive degradation always means that the feature-permuted control is worse. For metrics where larger is better:

```text
D = primary - control
```

For IBS and IPCW-ECE, where smaller is better:

```text
D = control - primary
```

The aggregator reports the mean paired degradation and a two-sided 95% Student-t interval over the five outer-fold differences. With only five folds, these intervals are descriptive cross-fold uncertainty summaries and should not be presented as independent-sample inferential evidence.

## Default output locations

Per-fold outputs:

```text
outputs/tcga_pathway_token_control_v1.6.0/fold_01
...
outputs/tcga_pathway_token_control_v1.6.0/fold_05
```

Aggregate outputs:

```text
outputs/tcga_pathway_token_control_summary_v1.6.0/
```

The aggregate directory contains:

```text
primary_fold_metrics.csv
control_fold_metrics.csv
paired_fold_differences.csv
pathway_token_control_summary.csv
pathway_token_control_publication_table.csv
PATHWAY_TOKEN_CONTROL_SUMMARY.md
pathway_token_control_aggregation_manifest.json
```

## Run command

```powershell
powershell -ExecutionPolicy Bypass `
    -File .\scripts\run_pathway_token_control.ps1
```

The runner is restart-safe. Completed folds with a `PASS` manifest are skipped, and an interrupted fold resumes from `last_model.pt` unless `-NoResume` is supplied.
