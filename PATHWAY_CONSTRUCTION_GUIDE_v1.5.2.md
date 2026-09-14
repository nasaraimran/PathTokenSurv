# PathTokenSurv v1.5.2 pathway construction guide

## 1. Biological mapping

The main analysis uses:

- mRNA Entrez IDs -> KEGG/Reactome pathways;
- CNV gene symbols -> NCBI-harmonized KEGG/Reactome pathways;
- TCGA human miRNAs -> **DIANA-TarBase v9.0 experimentally supported target genes** -> KEGG/Reactome pathways.

miRTarBase remains supported as an optional alternative/sensitivity provider.

## 2. Why TarBase is a local-file input

TarBase v9 documents unrestricted local retrieval through its web interface, but a stable direct bulk-file URL is not documented in the publication. PathTokenSurv therefore does not hard-code an unverified endpoint. Export/download the Homo sapiens interaction table from the official TarBase v9 portal and preserve the original file under `annotations/raw/`.

Official portal:

`https://dianalab.e-ce.uth.gr/tarbasev9`

The parser accepts CSV, TSV/TXT, CSV.GZ, or XLSX.

## 3. Recommended directory

```text
PathTokenSurv/
├── annotations/
│   ├── raw/
│   │   ├── Homo_sapiens.gene_info.gz
│   │   ├── NCBI2Reactome_All_Levels.txt
│   │   ├── ReactomePathways.txt
│   │   ├── ReactomePathwaysRelation.txt
│   │   ├── kegg_info.txt
│   │   ├── kegg_hsa_pathways.txt
│   │   ├── kegg_hsa_gene_pathway_links.txt
│   │   └── TarBase_v9_human.csv
│   └── processed/
└── data/tcga_pancancer_raw/
```

## 4. Download NCBI, Reactome and KEGG

```powershell
python .\scripts\construct_pathways.py `
    --download `
    --raw-dir .\annotations\raw `
    --processed-dir .\annotations\processed `
    --output .\data\tcga_pancancer_raw\pathways.json `
    --mirna-target-source tarbase `
    --tarbase-file .\annotations\raw\TarBase_v9_human.csv `
    --min-genes-per-pathway 5 `
    --max-genes-per-pathway 500
```

If the TarBase file is not present yet, first run only the NCBI/Reactome/KEGG download through `prepare_pathway_annotations.py --download --mirna-target-source none`, then rerun after obtaining the TarBase export.

## 5. Validate TarBase before pathway construction

```powershell
python .\scripts\validate_tarbase.py `
    --file .\annotations\raw\TarBase_v9_human.csv
```

The validator reports file size, SHA-256 checksum, unique human miRNA-target pairs, unique human miRNAs, unique target genes, and the number of pairs carrying an Entrez ID.

## 6. Build using an already cached NCBI/Reactome/KEGG bundle

```powershell
python .\scripts\construct_pathways.py `
    --raw-dir .\annotations\raw `
    --processed-dir .\annotations\processed `
    --output .\data\tcga_pancancer_raw\pathways.json `
    --mirna-target-source tarbase `
    --tarbase-file .\annotations\raw\TarBase_v9_human.csv `
    --min-genes-per-pathway 5 `
    --max-genes-per-pathway 500
```

For very large exports, adjust:

```powershell
--mirna-target-chunksize 250000
```

Lower this value if memory is limited.

## 7. Optional miRTarBase analysis

```powershell
python .\scripts\construct_pathways.py `
    --raw-dir .\annotations\raw `
    --processed-dir .\annotations\processed_mirtarbase `
    --output .\data\tcga_pancancer_raw\pathways_mirtarbase.json `
    --mirna-target-source mirtarbase `
    --mirtarbase-file .\annotations\raw\hsa_MTI.csv `
    --min-genes-per-pathway 5 `
    --max-genes-per-pathway 500
```

This is optional and should not replace the declared TarBase main analysis without updating the manuscript.

## 8. Output provenance

`annotations/processed/annotation_source_qc.json` records:

- miRNA-target provider;
- exact source file;
- SHA-256 checksum;
- number of unique miRNA-target pairs;
- number of unique human miRNAs;
- number of unique target genes;
- KEGG/Reactome pathway counts;
- Reactome edge policy.

`annotations/processed/pathway_construction_report.json` provides a compact summary for the final experiment log.

## 9. Run pathway QC after construction

```powershell
python .\scripts\run_tcga_qc.py `
    --config .\configs\tcga_pancancer_raw.json `
    --output-dir .\outputs\tcga_qc_v1.5.2 `
    --outer-folds 5 `
    --fold 1
```

Inspect:

```text
annotation_validation.csv
pathway_coverage_summary.csv
retained_pathway_sizes.csv
pathway_mapping_metadata.json
```

Do not start full nested-CV training until the identifier and pathway-coverage diagnostics have been reviewed.
