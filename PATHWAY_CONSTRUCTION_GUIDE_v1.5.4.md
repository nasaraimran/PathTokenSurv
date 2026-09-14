# PathTokenSurv v1.5.4 Pathway Construction Guide

## Recommended TCGA settings

The primary TCGA configuration uses:

```json
"max_features_per_pathway_by_modality": {
  "mrna": 128,
  "mirna": 32,
  "cnv": 128
},
"mirna_conservative_alias_harmonization": true,
"max_pathways": 256,
"balance_pathway_sources": true
```

The miRNA pathway resource is built with target-set enrichment:

```text
minimum target overlap = 3 genes
BH FDR = 0.05
score = -log10(BH-adjusted q-value)
```

The score is used only to choose the strongest miRNA features when a pathway exceeds the miRNA top-K cap. It is not a survival-derived weight.

## Fast upgrade from v1.5.3

If these files already exist:

```text
annotations/processed/gene_pathway_membership.tsv
annotations/processed/mirna_targets.tsv
annotations/processed/pathway_edges.tsv
```

you do not need to re-download or re-parse TarBase, KEGG, Reactome, or NCBI. Rebuild only `pathways.json`:

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

## What is new in pathways.json

v1.5.4 stores aligned miRNA link metadata under:

```text
modality_feature_metadata
└── mirna
    └── <pathway_id>
        ├── enrichment_score
        ├── q_value
        └── target_overlap
```

It also stores provider-wide conservative miRNA aliases in the pathway metadata. A base miRNA name is aliased to a mature arm only when exactly one arm is present in the complete provider input.

## Fold-specific ranking

For retained fold-selected features, PathTokenSurv first applies the modality-specific caps. For pathway p and modality m:

```text
C(m,p) = number of retained pathway features / number of selected features in modality m
```

The pathway score is the equal-weight mean across mRNA, miRNA, and CNV:

```text
C(p) = mean(C(mRNA,p), C(miRNA,p), C(CNV,p))
```

The score is outcome-independent. When more than `max_pathways` qualify, pathways are ranked by represented modality count and then normalized multimodal coverage. Optional KEGG/Reactome source balancing is then applied.

## Run QC

```powershell
python .\scripts\run_tcga_qc.py `
    --config .\configs\tcga_pancancer_raw.json `
    --output-dir .\outputs\tcga_qc_v1.5.4 `
    --outer-folds 5 `
    --fold 1
```

Review:

```text
annotation_validation.csv
pathway_coverage_summary.csv
retained_pathway_sizes.csv
pathway_mapping_metadata.json
unmatched_selected_features_mrna.csv
unmatched_selected_features_mirna.csv
unmatched_selected_features_cnv.csv
```

In `retained_pathway_sizes.csv`, distinguish `eligible_before_cap` from `selected_features`. A pathway may legitimately have more than 32 significant miRNA links; v1.5.4 deliberately keeps the 32 strongest links by enrichment score.

## Next step

If annotation coverage remains plausible, all three molecular modalities are represented, and the QC confirms score-aware miRNA truncation, proceed to:

```powershell
python .\scripts\dry_run_preprocessing.py `
    --config .\configs\tcga_pancancer_raw.json `
    --output-dir .\outputs\tcga_dry_run_v1.5.4 `
    --outer-folds 5 `
    --fold 1 `
    --batch-size 4
```
