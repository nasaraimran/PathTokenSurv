# PathTokenSurv v1.5.2 validation report

Validation performed on the packaged source before release.

## Automated tests

- Full unit/integration suite: **16 passed**.
- TarBase v9 parser human-species filtering: passed.
- TarBase v9 gene-symbol -> Entrez harmonization using NCBI gene_info: passed.
- TarBase v9 Entrez-only -> gene-symbol harmonization: passed.
- TarBase v9 chunked CSV/TSV parsing: passed.
- Default `prepare_unified_annotations(..., mirna_target_source="tarbase")` integration: passed.
- Existing miRTarBase parser tests: passed.
- Existing KEGG/Reactome parsers: passed.
- Pathway-builder and source-balanced pathway-cap tests: passed.
- TCGA raw-data/preprocessing/QC tests: passed.
- Model-forward and survival-metric tests: passed.

## Synthetic workflow

The complete synthetic PathTokenSurv training/inference smoke workflow was rerun after the v1.5.2 changes to ensure that the provider refactor did not alter downstream model execution.

## Deliberate design choice

No guessed TarBase bulk-download URL is embedded in the package. TarBase v9 is the primary provider, but the exact human export is supplied as a local file (or via an exact user-provided export URL). This preserves source provenance and avoids silently downloading a stale or unrelated mirror.
