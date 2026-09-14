# PathTokenSurv v1.3 validation report

## Automated tests

The complete test suite was executed after the v1.3 changes:

```text
7 passed
```

The tests cover the original survival-time, preprocessing, and forward-pass checks plus:

- TCGA patient-barcode normalization;
- TCGA sample-type parsing;
- pathologic/clinical stage harmonization;
- native features-by-samples matrix loading;
- duplicate primary-tumor aliquot aggregation;
- rejection of non-selected normal samples;
- zero-survival-time exclusion;
- Entrez/gene-symbol/miRNA pathway JSON construction; and
- pathway adjacency construction.

## End-to-end synthetic regression test

`python scripts/run_synthetic.py --config configs/synthetic.json --clean` completed successfully after the raw-TCGA changes. The synthetic values are software checks only and are not manuscript results.

## Validation against the uploaded TCGA-CDR workbook

The v1.3 clinical reader was executed directly on the supplied `PanCancer_Clinical.xlsx`, sheet `TCGA-CDR`.

```text
Clinical rows:                 11,160
Unique patient barcodes:      11,160
Cancer types:                 33
Missing OS.time:                  66
OS.time = 0:                    108
Invalid/missing OS event:         9
Eligible with positive time:  10,986
Observed deaths:               3,573
Censored:                      7,413
```

The reader also generated the expected canonical fields: `age`, `sex`, `race`, `stage`, and `histology`, while keeping cancer type in the outcome table for cancer conditioning.

### Important manuscript implication

The TCGA-CDR sheet has 11,094 non-missing `OS.time` values, which explains the previous manuscript count. However, 108 of those observations have `OS.time = 0`. The discrete-time survival likelihood requires a positive at-risk interval, so v1.3 excludes zero-time observations by default. The final cohort count must be frozen after the full molecular matching/QC run and then propagated through the Methods, Results, tables, and abstract.

## Molecular-file validation

The native TCGA molecular loader was validated with synthetic files matching the screenshots supplied by the user:

- first column contains feature identifiers;
- sample columns contain TCGA sample barcodes;
- mRNA feature IDs may be Entrez identifiers;
- miRNA feature IDs may be mature miRNA names;
- CNV feature IDs may be gene symbols;
- matrices are transposed internally to patient x feature orientation;
- duplicate same-type tumor aliquots are averaged by default;
- sample types are selected in configured priority order; and
- patients absent from a molecular modality remain in the cohort with that modality marked unavailable.
