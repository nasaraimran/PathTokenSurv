# PathTokenSurv v1.6.0 Validation Report

## Status

**PASS**

## Automated tests

The complete source test suite reports:

```text
36 passed
```

The v1.6.0 tests verify that the feature-label permutation:

- is deterministic for a fixed seed;
- changes with a different seed;
- is bijective;
- has no fixed points when a modality contains more than one feature;
- leaves pathway names and order unchanged;
- leaves selected feature names/order unchanged;
- leaves pathway adjacency unchanged;
- preserves every pathway token's feature count;
- preserves pairwise pathway-token overlap exactly;
- preserves the feature membership-degree multiset;
- preserves residual-token size and mapped/residual coverage;
- supports a paired five-fold aggregation with the standard positive-degradation convention.

All pre-existing v1.5.9 and earlier tests also pass.

## End-to-end smoke test

A real code-path synthetic smoke test was completed on CPU using the v1.5.5 synthetic smoke configuration:

1. train one leakage-controlled primary outer fold;
2. construct a deterministic feature-permuted pathway specification from that fold's saved `pathway_spec.json`;
3. reuse the exact primary split, preprocessor, time discretizer, and model seed;
4. train the control for the configured two epochs;
5. generate held-out test metrics, extended scientific evaluation, missingness evaluation, predictions, checkpoints, and provenance artifacts.

The pathway-control fold completed successfully with finite held-out C-index and IBS. The synthetic numerical values are software-validation results only and are not study findings.

## Scientific invariants

v1.6.0 does not alter:

- patient inclusion/exclusion;
- outer-fold membership;
- validation membership;
- fold-specific feature selection or scaling;
- pathway count;
- pathway names/order;
- pathway graph;
- token-size distribution;
- model architecture/parameter count;
- losses/hyperparameters;
- checkpoint-selection rule;
- primary model seed.

Only the biological correspondence between selected molecular features and pathway/residual token membership is permuted in the secondary control.
