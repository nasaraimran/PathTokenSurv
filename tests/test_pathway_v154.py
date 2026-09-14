from __future__ import annotations

from pathtokensurv.data.pathways import (
    annotation_feature_sets,
    build_pathway_spec,
    mapping_conservative_mirna_alias_map,
)


def _base_mapping():
    return {
        "pathways": ["P1"],
        "modality_features": {
            "mrna": {"P1": ["G1", "G2", "G3"]},
            "mirna": {"P1": ["hsa-miR-a-3p", "hsa-miR-b-3p", "hsa-miR-c-3p"]},
            "cnv": {"P1": ["G1", "G2", "G3"]},
        },
        "modality_feature_metadata": {
            "mirna": {
                "P1": {
                    "enrichment_score": [2.0, 9.0, 5.0],
                    "q_value": [1e-2, 1e-9, 1e-5],
                    "target_overlap": [4, 8, 6],
                }
            }
        },
        "metadata": {
            "pathway_sources": {"P1": "KEGG"},
            "mirna_conservative_aliases": {},
        },
        "adjacency": [[0.0]],
    }


def test_score_aware_top_k_mirna_selection():
    mapping = _base_mapping()
    selected = {
        "mrna": ["G1", "G2", "G3"],
        "mirna": ["hsa-miR-a-3p", "hsa-miR-b-3p", "hsa-miR-c-3p"],
        "cnv": ["G1", "G2", "G3"],
    }
    spec = build_pathway_spec(
        mapping,
        selected,
        max_features_per_pathway_by_modality={"mrna": 3, "mirna": 2, "cnv": 3},
    )
    idx = spec.feature_indices["mirna"][0]
    assert [selected["mirna"][i] for i in idx] == ["hsa-miR-b-3p", "hsa-miR-c-3p"]
    assert spec.eligible_feature_counts["mirna"][0] == 3


def test_modality_specific_caps_are_independent():
    mapping = _base_mapping()
    selected = {
        "mrna": ["G1", "G2", "G3"],
        "mirna": ["hsa-miR-a-3p", "hsa-miR-b-3p", "hsa-miR-c-3p"],
        "cnv": ["G1", "G2", "G3"],
    }
    spec = build_pathway_spec(
        mapping,
        selected,
        max_features_per_pathway_by_modality={"mrna": 1, "mirna": 2, "cnv": 3},
    )
    assert len(spec.feature_indices["mrna"][0]) == 1
    assert len(spec.feature_indices["mirna"][0]) == 2
    assert len(spec.feature_indices["cnv"][0]) == 3


def test_normalized_multimodal_ranking_not_raw_total():
    mrna = [f"M{i}" for i in range(1000)]
    mirna = [f"hsa-miR-{i}-3p" for i in range(10)]
    cnv = [f"C{i}" for i in range(1000)]
    mapping = {
        "pathways": ["RAW_BIG", "NORM_BIG"],
        "modality_features": {
            "mrna": {"RAW_BIG": mrna[:100], "NORM_BIG": mrna[:1]},
            "mirna": {"RAW_BIG": mirna[:1], "NORM_BIG": mirna[:10]},
            "cnv": {"RAW_BIG": cnv[:100], "NORM_BIG": cnv[:1]},
        },
        "metadata": {
            "pathway_sources": {"RAW_BIG": "KEGG", "NORM_BIG": "KEGG"},
            "mirna_conservative_aliases": {},
        },
        "adjacency": [[0.0, 0.0], [0.0, 0.0]],
    }
    selected = {"mrna": mrna, "mirna": mirna, "cnv": cnv}
    spec = build_pathway_spec(
        mapping,
        selected,
        max_pathways=1,
        max_features_per_pathway_by_modality={"mrna": 128, "mirna": 32, "cnv": 128},
    )
    # RAW_BIG has 201 raw matches vs NORM_BIG's 12, but NORM_BIG covers an
    # entire small miRNA modality and therefore has the larger normalized mean.
    assert spec.pathway_names == ["NORM_BIG"]


def test_conservative_mirna_alias_requires_unique_arm():
    unique = {
        "pathways": ["P"],
        "modality_features": {
            "mrna": {"P": ["G"]},
            "mirna": {"P": ["hsa-miR-375-3p"]},
            "cnv": {"P": ["G"]},
        },
        "metadata": {
            "pathway_sources": {"P": "KEGG"},
            "mirna_conservative_aliases": {"hsa-mir-375": "hsa-mir-375-3p"},
        },
        "adjacency": [[0.0]],
    }
    assert mapping_conservative_mirna_alias_map(unique)["hsa-mir-375"] == "hsa-mir-375-3p"
    assert "hsa-mir-375" in annotation_feature_sets(unique, True)["mirna"]
    selected = {"mrna": ["G"], "mirna": ["hsa-miR-375"], "cnv": ["G"]}
    spec = build_pathway_spec(unique, selected, conservative_mirna_alias_harmonization=True)
    assert len(spec.feature_indices["mirna"][0]) == 1

    ambiguous = {
        **unique,
        "modality_features": {
            "mrna": {"P": ["G"]},
            "mirna": {"P": ["hsa-miR-1-3p", "hsa-miR-1-5p"]},
            "cnv": {"P": ["G"]},
        },
        "metadata": {"pathway_sources": {"P": "KEGG"}, "mirna_conservative_aliases": {}},
    }
    selected2 = {"mrna": ["G"], "mirna": ["hsa-miR-1"], "cnv": ["G"]}
    spec2 = build_pathway_spec(ambiguous, selected2, conservative_mirna_alias_harmonization=True)
    assert len(spec2.feature_indices["mirna"][0]) == 0
