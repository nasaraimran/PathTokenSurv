# PathTokenSurv v1.5.3 validation report

Validation performed after the mRNA alias and miRNA-enrichment changes.

## Automated tests

`python -m pytest -q`

Result: **18 passed**.

The v1.5.3-specific tests verify:
- a pathway can match an mRNA feature supplied as either a gene symbol or an Entrez ID;
- selected-feature annotation coverage recognizes gene-symbol mRNA features;
- the annotation-coverage training guard operates before model training;
- target-set enrichment retains a pathway-specific miRNA while rejecting a deliberately promiscuous background miRNA;
- existing TCGA loading, preprocessing, pathway, model, survival, and annotation-source tests continue to pass.

## End-to-end synthetic test

`python scripts/run_synthetic.py --config configs/synthetic.json --clean`

Result: **End-to-end validation passed.**

This software test exercises data generation, preprocessing, pathway specification, model training, checkpointing, held-out evaluation, reload, and inference. Synthetic numerical performance is not a scientific result.

## Real-data status

The real TCGA v1.5.2 QC correctly exposed two issues before training:
1. mRNA identifiers were predominantly gene symbols but were being compared with Entrez-only pathway identifiers;
2. the permissive TarBase any-target rule saturated the 128-miRNA per-pathway cap.

v1.5.3 addresses both issues. The real pathway JSON and QC must now be regenerated to quantify final coverage.
