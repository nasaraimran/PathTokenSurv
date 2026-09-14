from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from pathlib import Path

from pathtokensurv.data.annotation_sources import load_ncbi_gene_map, parse_tarbase_targets, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize a DIANA-TarBase v9 export for PathTokenSurv.")
    parser.add_argument("--file", required=True)
    parser.add_argument("--gene-info", default="annotations/raw/Homo_sapiens.gene_info.gz")
    parser.add_argument("--chunksize", type=int, default=250000)
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(path)
    gene_map = None
    gene_info = Path(args.gene_info)
    if gene_info.exists():
        gene_map = load_ncbi_gene_map(gene_info)

    targets = parse_tarbase_targets(path, gene_map, chunksize=args.chunksize)
    if len(targets) == 0:
        raise RuntimeError("The TarBase file was readable, but zero human hsa-* miRNA-target pairs were parsed.")

    print(f"PASS: {path}")
    print(f"Bytes: {path.stat().st_size:,}")
    print(f"SHA256: {sha256_file(path)}")
    print(f"Unique human miRNA-target pairs: {len(targets):,}")
    print(f"Unique human miRNAs: {targets['mirna'].str.lower().nunique():,}")
    print(f"Unique target genes: {targets['gene_symbol'].str.upper().nunique():,}")
    print(f"Pairs with Entrez ID: {(targets['entrez_id'] != '').sum():,}")
    print("Provider: DIANA-TarBase v9.0")
    print("Columns parsed successfully. Use this file with construct_pathways.py --mirna-target-source tarbase --tarbase-file <FILE>.")


if __name__ == "__main__":
    main()
