# PathTokenSurv v1.5.3

## Why this release exists
Real TCGA QC showed that the mRNA matrix being used by PathTokenSurv is gene-symbol based (for example KRT5, KRT6A, CEACAM5), while v1.5.2 built the mRNA pathway map with Entrez IDs only. This produced 0% selected mRNA annotation coverage. The same QC also showed that the permissive TarBase any-target rule saturated the per-pathway miRNA cap.

## Changes
- mRNA pathway membership now stores both NCBI/Entrez IDs and gene-symbol aliases.
- mRNA matching canonicalizes numeric Entrez IDs and upper-case gene symbols automatically.
- QC reports the detected mRNA identifier mode: `entrez`, `gene_symbol`, or `mixed`.
- QC exports `unmatched_selected_features_{mrna,mirna,cnv}.csv`.
- Training and CPU dry-run now fail early if selected annotation coverage is below configurable modality-specific minimums.
- Default minimum selected annotation coverage: mRNA 20%, miRNA 10%, CNV 20%.
- TarBase miRNA-to-pathway construction now defaults to target-set enrichment instead of the permissive any-target rule.
- Default miRNA enrichment rule: at least 3 overlapping target genes and per-miRNA Benjamini-Hochberg FDR <= 0.05.
- Legacy `any_target` mapping remains available for sensitivity/ablation experiments.
- Enriched miRNA-pathway links are ranked by FDR then target overlap before the per-pathway feature cap is applied.
- Pathway metadata now records enrichment universe size, candidate links, retained links, and pathway-link distributions.
- `retained_pathway_sizes.csv` now flags modalities that hit `max_features_per_pathway`.
- Added explicit SciPy dependency for the hypergeometric enrichment calculation.

## Important migration requirement
Existing `pathways.json` files created by v1.5.2 must be rebuilt. Reusing the old JSON will preserve the mRNA mismatch and permissive miRNA mapping.
