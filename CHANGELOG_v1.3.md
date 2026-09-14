# PathTokenSurv v1.3

- Added direct loading of native TCGA PanCancer features-by-samples matrices.
- Added direct TCGA-CDR Excel parsing for OS/OS.time and cancer type.
- Added 12-character patient-barcode normalization and configurable tumor sample-type priority (01, 03, 09, then 06 by default).
- Added deterministic handling of duplicate primary-tumor aliquots and duplicate molecular features.
- Added stage harmonization with pathologic-stage preference and clinical-stage fallback.
- Added natural modality-availability preservation after patient alignment.
- Added raw-data QC metadata and `scripts/inspect_tcga_raw.py`.
- Added local annotation-to-`pathways.json` builder for Entrez mRNA IDs, CNV gene symbols, and miRNA target mappings.
- Added `configs/tcga_pancancer_raw.json` and annotation templates.
- Added `openpyxl` dependency for the TCGA-CDR workbook.
- Added native TCGA loader and pathway-builder unit tests.
