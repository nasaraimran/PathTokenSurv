from __future__ import annotations

import argparse
import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pathtokensurv.data.annotation_sources import load_ncbi_gene_map, parse_mirtarbase_targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a local human miRTarBase hsa_MTI.csv before pathway construction.")
    parser.add_argument("--file", required=True)
    parser.add_argument("--gene-info", default="annotations/raw/Homo_sapiens.gene_info.gz")
    args = parser.parse_args()

    path = _BootstrapPath(args.file)
    if not path.exists():
        raise FileNotFoundError(f"miRTarBase file not found: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"miRTarBase file is empty: {path}")

    gene_map = None
    gene_info = _BootstrapPath(args.gene_info)
    if gene_info.exists():
        gene_map = load_ncbi_gene_map(gene_info)

    targets = parse_mirtarbase_targets(path, gene_map)
    if targets.empty:
        raise RuntimeError("The file was readable, but zero human miRNA-target pairs were parsed.")
    print(f"PASS: {path}")
    print(f"Bytes: {path.stat().st_size:,}")
    print(f"Human miRNA-target pairs: {len(targets):,}")
    print(f"Unique human miRNAs: {targets['mirna'].str.lower().nunique():,}")
    print(f"Unique target genes: {targets['gene_symbol'].str.upper().nunique():,}")
    print("Columns parsed successfully. You can now use this file with construct_pathways.py --mirtarbase-file.")


if __name__ == "__main__":
    main()
