# PathTokenSurv v1.5

## Added

- Reproducible pathway-source download and cache pipeline.
- SHA-256 provenance manifest for external annotations.
- Reactome human Entrez-pathway parser.
- Reactome parent-child pathway graph parser.
- KEGG Homo sapiens pathway/gene REST parser.
- NCBI Entrez-to-gene-symbol harmonization.
- miRTarBase human target parser with robust CSV/TSV column detection.
- Unified `gene_pathway_membership.tsv`, `mirna_targets.tsv`, and `pathway_edges.tsv` generation.
- One-command `scripts/construct_pathways.py` workflow.
- Source-prefixed pathway IDs (`REACTOME:` and `KEGG:`).
- Source metadata retained in `pathways.json`.
- Configurable fold-specific `max_pathways` cap.
- Optional source-balanced pathway selection, enabled by default for the TCGA configuration.
- Pathway name/source fields in fold-specific QC output.

## Default TCGA pathway-token policy

- Source pathway size: 5–500 genes.
- Minimum matched selected features per pathway: 2.
- Maximum features per modality/pathway: 128.
- Maximum fold-specific pathway tokens: 256.
- Balance KEGG and Reactome under the pathway cap: enabled.

The pathway cap uses training-fold molecular feature coverage only. It never uses survival labels.
