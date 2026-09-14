from __future__ import annotations

from pathlib import Path

from pathtokensurv.data.pathway_builder import build_pathway_json_from_tables
from pathtokensurv.data.pathways import (
    build_pathway_spec,
    selected_annotation_coverage,
    validate_selected_annotation_coverage,
)


def _write_background(tmp_path: Path):
    genes = tmp_path / "genes.tsv"
    targets = tmp_path / "targets.tsv"
    rows = ["pathway_id\tpathway_name\tgene_symbol\tentrez_id\tsource"]
    target_rows = ["mirna\tgene_symbol"]
    # Ten non-overlapping pathways, ten genes each => a 100-gene enrichment universe.
    for p in range(10):
        for g in range(10):
            idx = p * 10 + g
            rows.append(f"P{p}\tPathway {p}\tG{idx}\t{1000+idx}\tKEGG")
            target_rows.append(f"hsa-miR-background\tG{idx}")
    # miR-specific strongly targets five genes from P0 only.
    for g in range(5):
        target_rows.append(f"hsa-miR-specific\tG{g}")
    genes.write_text("\n".join(rows) + "\n", encoding="utf-8")
    targets.write_text("\n".join(target_rows) + "\n", encoding="utf-8")
    return genes, targets


def test_mrna_gene_symbol_and_entrez_aliases(tmp_path):
    genes, targets = _write_background(tmp_path)
    out = tmp_path / "pathways.json"
    payload = build_pathway_json_from_tables(
        genes, out, mirna_targets_file=targets,
        min_genes_per_pathway=5, max_genes_per_pathway=20,
        mirna_pathway_mode="enrichment", mirna_min_target_overlap=3,
        mirna_enrichment_fdr=0.05,
    )
    assert "G0" in payload["modality_features"]["mrna"]["P0"]
    assert "1000" in payload["modality_features"]["mrna"]["P0"]

    selected = {
        "mrna": ["G0", "G1", "G2"],
        "mirna": ["hsa-miR-specific"],
        "cnv": ["G0", "G1"],
    }
    coverage = selected_annotation_coverage(payload, selected)
    assert coverage["mrna"]["matched_selected_features"] == 3
    validate_selected_annotation_coverage(payload, selected, {"mrna": 50, "mirna": 1, "cnv": 50})

    spec = build_pathway_spec(payload, selected, max_features_per_pathway=128)
    assert len(spec.feature_indices["mrna"][spec.pathway_names.index("P0")]) == 3


def test_mirna_enrichment_removes_promiscuous_any_target_links(tmp_path):
    genes, targets = _write_background(tmp_path)
    out = tmp_path / "pathways.json"
    payload = build_pathway_json_from_tables(
        genes, out, mirna_targets_file=targets,
        min_genes_per_pathway=5, max_genes_per_pathway=20,
        mirna_pathway_mode="enrichment", mirna_min_target_overlap=3,
        mirna_enrichment_fdr=0.05,
    )
    # The pathway-specific miRNA is enriched in P0, while the background miRNA
    # targeting every gene should not be called enriched in every pathway.
    assert "hsa-miR-specific" in payload["modality_features"]["mirna"]["P0"]
    assert all(
        "hsa-miR-background" not in payload["modality_features"]["mirna"][p]
        for p in payload["pathways"]
    )
    meta = payload["metadata"]
    assert meta["mirna_pathway_mode"] == "enrichment"
    assert meta["mirna_pathway_retained_links"] >= 1
