from pathlib import Path
import pandas as pd
from pathtokensurv.data.annotation_sources import parse_mirtarbase_targets


def test_mirtarbase_common_entrez_header(tmp_path: Path):
    p = tmp_path / "hsa_MTI.csv"
    pd.DataFrame({
        "miRNA": ["hsa-miR-21-5p"],
        "Target Gene": ["PTEN"],
        "Target Gene (Entrez ID)": ["5728"],
        "Experiments": ["Reporter assay"],
        "Support Type": ["Functional MTI"],
    }).to_csv(p, index=False)
    out = parse_mirtarbase_targets(p)
    assert len(out) == 1
    assert out.iloc[0]["entrez_id"] == "5728"
