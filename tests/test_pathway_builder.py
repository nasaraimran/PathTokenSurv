from pathtokensurv.data.pathway_builder import build_pathway_json_from_tables


def test_build_pathway_json(tmp_path):
    genes = tmp_path / "genes.tsv"
    targets = tmp_path / "targets.tsv"
    edges = tmp_path / "edges.tsv"
    genes.write_text("pathway_id\tgene_symbol\tentrez_id\nP1\tTP53\t7157\nP2\tEGFR\t1956\n")
    targets.write_text("mirna\tgene_symbol\nhsa-miR-X\tTP53\n")
    edges.write_text("source_pathway\ttarget_pathway\nP1\tP2\n")
    out = tmp_path / "pathways.json"
    payload = build_pathway_json_from_tables(genes, out, targets, edges, mirna_pathway_mode="any_target", mirna_min_target_overlap=1)
    assert set(payload["modality_features"]["mrna"]["P1"]) == {"7157", "TP53"}
    assert payload["modality_features"]["cnv"]["P2"] == ["EGFR"]
    assert payload["modality_features"]["mirna"]["P1"] == ["hsa-miR-X"]
    assert payload["adjacency"][0][1] == 1.0


def test_pathway_size_filter_and_metadata(tmp_path):
    genes = tmp_path / "genes2.tsv"
    genes.write_text(
        "pathway_id\tgene_symbol\tentrez_id\n"
        "Psmall\tTP53\t7157\n"
        "Pkeep\tEGFR\t1956\n"
        "Pkeep\tKRAS\t3845\n"
        "Pkeep\tBRAF\t673\n"
    )
    out = tmp_path / "pathways2.json"
    payload = build_pathway_json_from_tables(
        genes, out, min_genes_per_pathway=2, max_genes_per_pathway=4
    )
    assert payload["pathways"] == ["Pkeep"]
    assert payload["metadata"]["source_unique_pathways"] == 2
    assert payload["metadata"]["pathways_removed_by_size_filter"] == 1
