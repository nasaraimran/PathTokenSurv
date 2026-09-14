# PathTokenSurv v1.5.4 Validation Report

Date: 2026-08-09

## Automated tests

Command:

```text
PYTHONPATH=. pytest -q
```

Result:

```text
22 passed
```

The v1.5.4 tests specifically verify:

- enrichment-score-aware top-K miRNA selection;
- independent mRNA/miRNA/CNV pathway caps;
- normalized multimodal pathway ranking rather than raw feature-total ranking;
- conservative unique-arm miRNA aliasing;
- rejection of ambiguous arm-unspecified miRNA aliases;
- v1.5.3 mRNA symbol/Entrez compatibility and enrichment behavior;
- existing model, preprocessing, TCGA loader, annotation-source, and survival-metric tests.

## End-to-end synthetic validation

Command:

```text
PYTHONPATH=. python scripts/run_synthetic.py --clean
```

Result:

```text
End-to-end validation passed.
```

The synthetic run completed data generation, fold preprocessing, pathway specification, model training, checkpointing, held-out evaluation, model reload, and inference.

Synthetic performance values are software-validation outputs only and must not be reported as TCGA study results.

## Real-data status

v1.5.4 has not been run against the user's complete TCGA molecular matrices in this environment. The next required real-data step is to rebuild `pathways.json` from the existing processed annotation tables and rerun fold-1 TCGA QC locally.
