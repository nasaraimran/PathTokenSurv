from pathlib import Path
import gzip
import pandas as pd

from pathtokensurv.data.annotation_sources import (
    load_ncbi_gene_map,
    parse_tarbase_targets,
    prepare_unified_annotations,
)


def _gene_map_file(tmp_path: Path) -> Path:
    p = tmp_path / "Homo_sapiens.gene_info.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write("#tax_id\tGeneID\tSymbol\n")
        f.write("9606\t7157\tTP53\n")
        f.write("9606\t5728\tPTEN\n")
        f.write("9606\t3845\tKRAS\n")
    return p


def test_tarbase_parser_filters_human_and_maps_entrez(tmp_path: Path):
    gene_info = _gene_map_file(tmp_path)
    gm = load_ncbi_gene_map(gene_info)
    p = tmp_path / "TarBase_v9_human.csv"
    pd.DataFrame({
        "miRNA name": ["hsa-miR-21-5p", "hsa-let-7a-5p", "mmu-miR-1-3p", "hsv1-miR-H1"],
        "Gene": ["PTEN", "KRAS", "TP53", "TP53"],
        "Species": ["Homo sapiens", "human", "Mus musculus", "Homo sapiens"],
        "Experimental method": ["Reporter assay", "CLIP-Seq", "Western blot", "CLIP-Seq"],
        "Experimental type": ["Direct", "Direct", "Indirect", "Direct"],
    }).to_csv(p, index=False)
    out = parse_tarbase_targets(p, gm, chunksize=2)
    assert set(out["mirna"]) == {"hsa-miR-21-5p", "hsa-let-7a-5p"}
    assert dict(zip(out["gene_symbol"], out["entrez_id"]))["PTEN"] == "5728"
    assert dict(zip(out["gene_symbol"], out["entrez_id"]))["KRAS"] == "3845"
    assert set(out["source"]) == {"TarBase-v9.0"}


def test_tarbase_parser_handles_entrez_without_symbol(tmp_path: Path):
    gm = load_ncbi_gene_map(_gene_map_file(tmp_path))
    p = tmp_path / "tarbase.tsv"
    pd.DataFrame({
        "miRNA": ["hsa-miR-125b-5p"],
        "NCBI Gene ID": ["7157"],
        "Organism": ["9606"],
        "Method": ["Luciferase reporter assay"],
    }).to_csv(p, sep="\t", index=False)
    out = parse_tarbase_targets(p, gm)
    assert len(out) == 1
    assert out.iloc[0]["gene_symbol"] == "TP53"
    assert out.iloc[0]["entrez_id"] == "7157"


def test_prepare_annotations_defaults_to_tarbase(tmp_path: Path):
    raw = tmp_path / "raw"
    outdir = tmp_path / "processed"
    raw.mkdir()
    _gene_map_file(raw)
    # Minimal Reactome source; disable KEGG in this unit test.
    (raw / "NCBI2Reactome_All_Levels.txt").write_text(
        "7157\tR-HSA-1\thttps://x\tDNA repair\tTAS\tHomo sapiens\n"
        "5728\tR-HSA-1\thttps://x\tDNA repair\tTAS\tHomo sapiens\n",
        encoding="utf-8",
    )
    (raw / "ReactomePathwaysRelation.txt").write_text("", encoding="utf-8")
    tb = raw / "TarBase_v9_human.csv"
    pd.DataFrame({
        "miRNA": ["hsa-miR-21-5p"],
        "gene_symbol": ["PTEN"],
        "species": ["Homo sapiens"],
        "method": ["Reporter assay"],
    }).to_csv(tb, index=False)
    result = prepare_unified_annotations(raw, outdir, include_kegg=False)
    assert result["qc"]["mirna_target_provider"] == "tarbase"
    assert result["qc"]["mirna_target_pairs"] == 1
    assert result["qc"]["mirna_target_provider_label"] == "DIANA-TarBase v9.0"
