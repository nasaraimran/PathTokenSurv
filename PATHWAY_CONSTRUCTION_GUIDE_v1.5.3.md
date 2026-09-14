# PathTokenSurv v1.5.3 pathway construction guide

## Real TCGA identifier finding
The real PathTokenSurv QC showed that the loaded mRNA feature names are predominantly gene symbols (examples include `KRT5`, `KRT6A`, `CEACAM5`, `KRT14`) rather than Entrez-only identifiers. v1.5.3 therefore stores both gene-symbol and Entrez aliases in the mRNA pathway map.

## Recommended miRNA mapping
TarBase v9 remains the primary experimentally supported miRNA-target source. v1.5.3 maps a miRNA to a KEGG/Reactome pathway only when its TarBase target set is enriched in that pathway.

Default rule:
- at least 3 unique target genes overlap the pathway;
- one-sided hypergeometric enrichment test;
- Benjamini-Hochberg correction performed separately across candidate pathways for each miRNA;
- FDR <= 0.05.

This replaces the v1.5.2 rule in which one target gene was sufficient to link a miRNA to a pathway.

## Fast rebuild using already processed v1.5.2 annotations
If these files already exist:

- `annotations/processed/gene_pathway_membership.tsv`
- `annotations/processed/mirna_targets.tsv`
- `annotations/processed/pathway_edges.tsv`

there is no need to download TarBase, KEGG, Reactome, or NCBI again. Run:

```powershell
python .\scripts\build_pathways.py `
    --gene-membership .\annotations\processed\gene_pathway_membership.tsv `
    --mirna-targets .\annotations\processed\mirna_targets.tsv `
    --pathway-edges .\annotations\processed\pathway_edges.tsv `
    --output .\data\tcga_pancancer_raw\pathways.json `
    --min-genes-per-pathway 5 `
    --max-genes-per-pathway 500 `
    --mirna-pathway-mode enrichment `
    --mirna-min-target-overlap 3 `
    --mirna-enrichment-fdr 0.05
```

## Full rebuild from the raw TarBase v9 file
Use this if you also want to regenerate the processed annotation tables:

```powershell
python .\scripts\construct_pathways.py `
    --raw-dir .\annotations\raw `
    --processed-dir .\annotations\processed `
    --output .\data\tcga_pancancer_raw\pathways.json `
    --mirna-target-source tarbase `
    --tarbase-file .\annotations\raw\Homo_sapiens_TarBase-v9.tsv.gz `
    --min-genes-per-pathway 5 `
    --max-genes-per-pathway 500 `
    --mirna-pathway-mode enrichment `
    --mirna-min-target-overlap 3 `
    --mirna-enrichment-fdr 0.05
```

Add `--download` only if the NCBI/Reactome/KEGG source files also need to be downloaded again.

## Legacy sensitivity mapping
The old permissive mapping remains available only for sensitivity/ablation work:

```text
--mirna-pathway-mode any_target --mirna-min-target-overlap 1
```

Do not use that mode for the primary experiment.

## Rerun QC
After rebuilding `pathways.json`:

```powershell
python .\scripts\run_tcga_qc.py `
    --config .\configs\tcga_pancancer_raw.json `
    --output-dir .\outputs\tcga_qc_v1.5.3 `
    --outer-folds 5 `
    --fold 1
```

Inspect:
- `annotation_validation.csv`
- `pathway_coverage_summary.csv`
- `retained_pathway_sizes.csv`
- `pathway_mapping_metadata.json`
- `unmatched_selected_features_mrna.csv`
- `unmatched_selected_features_mirna.csv`
- `unmatched_selected_features_cnv.csv`

Do not begin the CPU dry run until the annotation coverage guard passes.
