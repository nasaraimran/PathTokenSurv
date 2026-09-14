# PathTokenSurv v1.5.9 — Frozen Architecture Ablation Protocol

## Purpose

v1.5.9 evaluates five hypothesis-driven components of the frozen PathTokenSurv model without changing the already completed five-fold primary experiment. The original primary folds remain the full-model reference and are **not retrained**.

## Frozen reference

Every ablation fold reuses the corresponding primary fold's exact:

- train / validation / held-out test patient assignments;
- fold-fitted `preprocessor.pkl`;
- retained `pathway_spec.json`;
- `time_discretizer.json`;
- model initialization seed;
- optimization schedule, batch size, learning rate, weight decay, maximum epochs, patience, and checkpoint-selection rule;
- extended held-out evaluation protocol.

The only allowed changes are the named ablation overrides below.

## Primary ablations

| Ablation | Frozen change | Question tested |
|---|---|---|
| `no_structured_masking` | `modality_drop_probability = 0` | Does artificial modality masking contribute to robustness/generalization? |
| `no_reconstruction_objective` | `lambda_reconstruction = 0` | Does the explicit cross-modal reconstruction objective add value? |
| `no_subset_consistency` | `use_subset_consistency = false`, `lambda_consistency = 0` | Does subset/full-view consistency regularization add value? |
| `no_pathway_bias` | `use_pathway_bias = false` | Does biological same-pathway / Reactome graph attention bias add value beyond pathway tokens alone? |
| `no_cancer_conditioning` | `use_cancer_deviation = false` | Does cancer-conditioned discrete-time survival modeling add value? |

### Important interpretation of `no_structured_masking`

When artificial modality dropping is disabled, the effective training mask equals the natural availability mask. Therefore no *observed* modality is intentionally hidden to create a reconstruction target, so masked-target reconstruction supervision becomes inactive. The subset-consistency branch remains enabled and compares two stochastic representations generated with the same effective modality mask. This ablation therefore isolates artificial modality dropping and its induced reconstruction supervision; it does not remove the consistency objective.

### Important interpretation of `no_reconstruction_objective`

The cross-modal reconstructor remains in the forward graph. Only the explicit reconstruction target loss is set to zero. This avoids conflating the reconstruction **objective** with removal of the reconstruction **module**.

## Leakage control

The ablation runner loads split membership directly from the already completed primary `split_assignments.csv`. It also loads the primary fold's preprocessing, pathway, and time-discretization artifacts rather than refitting them. Each run writes SHA-256 fingerprints for those frozen artifacts into `ablation_manifest.json`.

Held-out test metrics are generated only after validation-based checkpoint selection has completed. Test results do not influence stopping or hyperparameters.

## Paired five-fold analysis

For each ablation and each metric, the final analysis is paired by outer fold against the corresponding frozen primary result. The convention is:

> **positive degradation = the ablation is worse than the primary model**.

For higher-is-better metrics (C-index and tdAUC):

`degradation = primary - ablation`

For lower-is-better metrics (IBS and IPCW-ECE):

`degradation = ablation - primary`

The report includes the mean paired degradation and a two-sided 95% Student-t interval over the five paired fold differences. These are cross-fold intervals, not multi-seed confidence intervals.

## Execution

Run all five ablations across all five frozen outer folds:

```powershell
powershell -ExecutionPolicy Bypass `
    -File .\scripts\run_architecture_ablations.ps1
```

The runner:

1. skips an ablation/fold already marked `PASS`;
2. automatically resumes an interrupted run from `last_model.pt` when available;
3. aborts if CUDA is unavailable;
4. runs extended evaluation after each completed fold;
5. aggregates all 25 completed ablation runs automatically at the end.

Use `-NoResume` to deliberately restart incomplete runs, or `-SkipAggregate` when running only a subset.

## Output layout

```text
outputs/tcga_architecture_ablations_v1.5.9/
  no_structured_masking/fold_01 ... fold_05
  no_reconstruction_objective/fold_01 ... fold_05
  no_subset_consistency/fold_01 ... fold_05
  no_pathway_bias/fold_01 ... fold_05
  no_cancer_conditioning/fold_01 ... fold_05
```

Each fold contains the standard training/evaluation artifacts plus `ablation_manifest.json`.

Final aggregation:

```text
outputs/tcga_architecture_ablation_summary_v1.5.9/
  primary_fold_metrics.csv
  ablation_fold_metrics.csv
  paired_fold_differences.csv
  architecture_ablation_summary.csv
  architecture_ablation_publication_table.csv
  ARCHITECTURE_ABLATION_SUMMARY.md
  architecture_ablation_aggregation_manifest.json
```

## Reporting rule

All five prespecified ablations are reported regardless of whether they improve, worsen, or leave performance unchanged. The primary model remains fixed; no architecture selection is performed using these held-out ablation results.
