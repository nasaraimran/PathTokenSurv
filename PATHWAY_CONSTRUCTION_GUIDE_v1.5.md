# PathTokenSurv v1.5 — Pathway Construction Guide

## Goal

Build one reproducible `pathways.json` that aligns the three TCGA molecular identifier spaces to a common biological pathway vocabulary:

- mRNA: Entrez Gene ID -> KEGG/Reactome pathway
- CNV: gene symbol -> KEGG/Reactome pathway
- miRNA: mature human miRNA -> experimentally validated target gene -> KEGG/Reactome pathway

The pipeline caches every downloaded source and writes SHA-256 checksums and retrieval timestamps. Raw source files are not bundled with PathTokenSurv.

## Recommended database policy

- Reactome: current human NCBI-to-Reactome all-level mappings plus the official human parent-child relation file.
- KEGG: current Homo sapiens pathway list and human gene-to-pathway links from the KEGG REST API.
- Gene symbol harmonization: NCBI Homo sapiens `gene_info`.
- miRNA targets: human miRTarBase release 10.0 `hsa_MTI.csv` by default. The release is configurable.

Reactome provides explicit pathway parent-child edges. KEGG contributes pathway membership. v1.5 does not invent KEGG pathway-pathway edges. This keeps the graph prior curated and auditable.

## One-command construction

From the project root:

```powershell
python .\scripts\construct_pathways.py `
    --download `
    --raw-dir .\annotations\raw `
    --processed-dir .\annotations\processed `
    --output .\data\tcga_pancancer_raw\pathways.json `
    --mirtarbase-release 10.0 `
    --min-genes-per-pathway 5 `
    --max-genes-per-pathway 500
```

If the miRTarBase server does not download successfully, download the human `hsa_MTI.csv` manually and rerun:

```powershell
python .\scripts\construct_pathways.py `
    --raw-dir .\annotations\raw `
    --processed-dir .\annotations\processed `
    --output .\data\tcga_pancancer_raw\pathways.json `
    --mirtarbase-file "D:\path\to\hsa_MTI.csv" `
    --min-genes-per-pathway 5 `
    --max-genes-per-pathway 500
```

## Outputs

### annotations/raw/
Downloaded source snapshots and `source_manifest.json` with URLs, timestamps, file sizes and SHA-256 hashes.

### annotations/processed/gene_pathway_membership.tsv
Columns:

- `pathway_id`
- `pathway_name`
- `gene_symbol`
- `entrez_id`
- `source`

Pathway IDs are source-prefixed, e.g. `REACTOME:R-HSA-...` and `KEGG:hsa...`, so IDs never collide.

### annotations/processed/mirna_targets.tsv
Human miRNA-target pairs parsed from miRTarBase. Extra evidence columns are retained in this table even though `build_pathways.py` only requires `mirna` and `gene_symbol`.

### annotations/processed/pathway_edges.tsv
Reactome human parent-child relations, with source-prefixed IDs.

### data/tcga_pancancer_raw/pathways.json
The final PathTokenSurv mapping consumed by training and QC.

## Two-stage pathway filtering

### Stage A: source annotation filter

`construct_pathways.py` removes source pathways outside the configured source size range. The initial recommendation is 5–500 unique genes.

### Stage B: training-fold filter

During each outer training fold, the model intersects the source pathways with the selected molecular features. The default v1.5 settings are:

- minimum selected features per pathway: 2
- maximum selected features per modality/pathway: 128
- minimum molecular modalities represented: 1
- maximum retained pathway tokens: 256
- source-balanced cap: enabled

The 256-pathway cap is unsupervised. It uses only training-fold feature coverage and never survival outcomes. If more than 256 pathways remain, v1.5 ranks pathways by (1) number of represented molecular modalities and (2) number of matched selected features. When both KEGG and Reactome are present, the cap first allocates equal source quotas and then fills unused positions from the overall ranking.

`max_pathways` remains configurable and should be included in sensitivity analysis before the final paper is submitted.

## Next QC command

After construction succeeds:

```powershell
python .\scripts\run_tcga_qc.py `
    --config .\configs\tcga_pancancer_raw.json `
    --output-dir .\outputs\tcga_qc_v1.5 `
    --outer-folds 5 `
    --fold 1
```

Review these files before training:

- `annotation_validation.csv`
- `pathway_coverage_summary.csv`
- `retained_pathway_sizes.csv`
- `pathway_mapping_metadata.json`
- `qc_pathway_spec.json`

## Pass criteria before real training

Do not start the full nested-CV run until:

1. mRNA selected-feature annotation coverage is high enough to justify pathway tokenization.
2. CNV selected-feature annotation coverage is high enough to justify pathway tokenization.
3. A meaningful fraction of the 743 miRNAs match miRTarBase and produce pathway memberships.
4. Retained pathways include both KEGG and Reactome when source balancing is enabled.
5. The residual token is not carrying nearly all features for any modality.
6. `dry_run_preprocessing.py` reports `status: PASS`.
