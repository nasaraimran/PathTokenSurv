from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

from pathtokensurv.data.annotation_sources import (
    load_ncbi_gene_map,
    parse_kegg_membership,
    parse_mirtarbase_targets,
    parse_reactome_membership,
    parse_reactome_relations,
)
from pathtokensurv.data.pathways import build_pathway_spec


def test_annotation_source_parsers(tmp_path: Path):
    gene_info = tmp_path / "Homo_sapiens.gene_info.gz"
    with gzip.open(gene_info, "wt", encoding="utf-8") as f:
        f.write("#tax_id\tGeneID\tSymbol\n9606\t7157\tTP53\n9606\t1956\tEGFR\n9606\t672\tBRCA1\n")
    gm = load_ncbi_gene_map(gene_info)
    assert dict(zip(gm.entrez_id, gm.gene_symbol))["7157"] == "TP53"

    react = tmp_path / "NCBI2Reactome_All_Levels.txt"
    react.write_text(
        "7157\tR-HSA-1\thttps://x\tDNA repair\tTAS\tHomo sapiens\n"
        "1956\tR-HSA-1\thttps://x\tDNA repair\tTAS\tHomo sapiens\n"
        "672\tR-MMU-2\thttps://x\tMouse path\tTAS\tMus musculus\n",
        encoding="utf-8",
    )
    rm = parse_reactome_membership(react, gm)
    assert set(rm.pathway_id) == {"REACTOME:R-HSA-1"}
    assert set(rm.gene_symbol) == {"TP53", "EGFR"}

    rel = tmp_path / "ReactomePathwaysRelation.txt"
    rel.write_text("R-HSA-1\tR-HSA-3\nR-MMU-1\tR-MMU-2\n", encoding="utf-8")
    rr = parse_reactome_relations(rel)
    assert rr.iloc[0].source_pathway == "REACTOME:R-HSA-1"

    klist = tmp_path / "kegg_hsa_pathways.txt"
    klist.write_text("hsa05200\tPathways in cancer - Homo sapiens (human)\n", encoding="utf-8")
    klink = tmp_path / "kegg_hsa_gene_pathway_links.txt"
    klink.write_text("hsa:7157\tpath:hsa05200\nhsa:1956\tpath:hsa05200\n", encoding="utf-8")
    km = parse_kegg_membership(klist, klink, gm)
    assert set(km.entrez_id) == {"7157", "1956"}
    assert km.pathway_id.iloc[0] == "KEGG:hsa05200"

    mt = tmp_path / "hsa_MTI.csv"
    pd.DataFrame({
        "miRNA": ["hsa-miR-125b-5p", "mmu-miR-1"],
        "Target Gene": ["TP53", "EGFR"],
        "Target Gene (Entrez Gene ID)": ["7157", "1956"],
        "Support Type": ["Functional MTI", "Functional MTI"],
        "Experiments": ["Reporter assay", "Western blot"],
    }).to_csv(mt, index=False)
    targets = parse_mirtarbase_targets(mt, gm)
    assert len(targets) == 1
    assert targets.iloc[0].mirna == "hsa-miR-125b-5p"


def test_source_balanced_pathway_cap():
    pathways = [f"R{i}" for i in range(6)] + [f"K{i}" for i in range(6)]
    mapping = {
        "pathways": pathways,
        "modality_features": {m: {} for m in ["mrna", "mirna", "cnv"]},
        "adjacency": [[0.0] * len(pathways) for _ in pathways],
        "metadata": {"pathway_sources": {}},
    }
    for i, p in enumerate(pathways):
        source = "Reactome" if p.startswith("R") else "KEGG"
        mapping["metadata"]["pathway_sources"][p] = source
        mapping["modality_features"]["mrna"][p] = [f"g{i}"]
        mapping["modality_features"]["mirna"][p] = [f"mi{i}"]
        mapping["modality_features"]["cnv"][p] = [f"c{i}"]
    features = {
        "mrna": [f"g{i}" for i in range(12)],
        "mirna": [f"mi{i}" for i in range(12)],
        "cnv": [f"c{i}" for i in range(12)],
    }
    spec = build_pathway_spec(mapping, features, max_pathways=4, balance_pathway_sources=True)
    assert spec.num_pathways == 4
    sources = [mapping["metadata"]["pathway_sources"][p] for p in spec.pathway_names]
    assert sources.count("Reactome") == 2
    assert sources.count("KEGG") == 2
