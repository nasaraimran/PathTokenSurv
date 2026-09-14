# PathTokenSurv v1.5.9 Validation Report

## Status

**PASS**

## Automated tests

Command:

```text
python -m pytest -q
```

Result:

```text
33 passed
```

The v1.5.9 tests verify:

- the exact five prespecified ablation definitions;
- reconstruction-objective ablation retains the reconstruction module;
- subset-consistency ablation disables both its branch and loss weight;
- the positive-is-worse degradation convention for both higher- and lower-is-better metrics;
- complete five-fold pairing across all 25 ablation runs;
- publication-table and Markdown summary creation.

## Synthetic frozen-artifact smoke test

A small synthetic primary outer fold was trained for one epoch, producing a complete primary fold artifact set. A `no_pathway_bias` ablation was then trained through the new frozen-artifact runner using the primary fold's exact:

- train / validation / test assignments;
- preprocessing artifact;
- pathway specification;
- time discretizer;
- model seed.

The ablation completed training, held-out evaluation, scientific evaluation, robustness evaluation, checkpointing, and manifest generation successfully.

The synthetic numerical metrics are software-validation outputs only and are **not** TCGA study results.

## Static validation

`python -m compileall` passed for package modules, scripts, and tests.

## Protocol impact

v1.5.9 does not alter or invalidate any previously completed primary PathTokenSurv fold, clinical/cancer baseline, or v1.5.8 missingness result. It adds a prespecified secondary architecture-ablation experiment only.
