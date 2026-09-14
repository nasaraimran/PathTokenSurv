# PathTokenSurv v1.5.2

Pathway-construction update that makes **DIANA-TarBase v9.0** the primary miRNA-target provider while retaining miRTarBase as an optional alternative.

## Main changes

- `--mirna-target-source tarbase` is now the default in `construct_pathways.py` and `prepare_pathway_annotations.py`.
- Added a memory-aware TarBase v9 parser supporting CSV, TSV/TXT, CSV.GZ, and XLSX exports.
- TarBase parsing auto-detects common miRNA, gene-symbol, Entrez-ID, species, experimental-method, and experimental-type columns.
- Human rows are restricted to `Homo sapiens`/human/9606 when a species field exists, and only `hsa-` miRNAs are retained for TCGA.
- Official NCBI gene mappings fill missing Entrez IDs from gene symbols and missing gene symbols from Entrez IDs.
- Large delimited TarBase exports are processed in chunks (`--mirna-target-chunksize`, default 250,000 rows) before duplicate miRNA-gene pairs are collapsed.
- Added `scripts/validate_tarbase.py` for preflight validation and SHA-256 provenance reporting.
- Added `--tarbase-url` as an optional exact export URL. No unverified TarBase bulk URL is hard-coded.
- `source_manifest.json` now records that the TarBase v9 portal requires a local export when no exact download URL is supplied.
- `annotation_source_qc.json` records provider, source file, SHA-256 checksum, unique miRNAs, unique target genes, and unique miRNA-target pairs.
- miRTarBase remains available with `--mirna-target-source mirtarbase --mirtarbase-file ...` and its legacy download attempt is opt-in via `--download-mirtarbase`.
- `--allow-empty-mirna` remains restricted to deliberate ablation experiments.
- Added three v1.5.2 tests for TarBase schema handling, human filtering, Entrez mapping, and default-provider integration.

## Methodological behavior unchanged

KEGG and Reactome still define the shared gene-pathway space, Reactome parent-child relations define the explicit pathway graph, pathway-size filtering remains 5-500 genes by default, and all fold-specific TCGA feature/pathway selection remains training-only.
