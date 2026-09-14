from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from pathlib import Path

from pathtokensurv.data.annotation_sources import download_source_bundle, prepare_unified_annotations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare NCBI/Reactome/KEGG annotations plus TarBase-v9 (default) or miRTarBase miRNA-target mappings."
    )
    parser.add_argument("--raw-dir", default="annotations/raw")
    parser.add_argument("--output-dir", default="annotations/processed")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--reactome-direct-only", action="store_true")
    parser.add_argument("--no-kegg", action="store_true")
    parser.add_argument("--no-reactome", action="store_true")
    parser.add_argument("--mirna-target-source", choices=["tarbase", "mirtarbase", "none"], default="tarbase")
    parser.add_argument("--mirna-target-file", default=None)
    parser.add_argument("--tarbase-file", default=None)
    parser.add_argument("--tarbase-url", default=None)
    parser.add_argument("--mirna-target-chunksize", type=int, default=250000)
    parser.add_argument("--mirtarbase-release", default="10.0")
    parser.add_argument("--mirtarbase-file", default=None)
    parser.add_argument("--download-mirtarbase", action="store_true")
    parser.add_argument("--no-mirtarbase-download", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    if args.download:
        print("Downloading and checksumming NCBI/Reactome/KEGG annotation sources...")
        download_source_bundle(
            raw_dir,
            reactome_all_levels=not args.reactome_direct_only,
            include_kegg=not args.no_kegg,
            include_mirtarbase=(args.download_mirtarbase and not args.no_mirtarbase_download),
            mirtarbase_release=args.mirtarbase_release,
            tarbase_url=args.tarbase_url if args.mirna_target_source == "tarbase" else None,
            tarbase_file=(args.mirna_target_file or args.tarbase_file) if args.mirna_target_source == "tarbase" else None,
        )

    print("Building unified annotation tables...")
    result = prepare_unified_annotations(
        raw_dir,
        output_dir,
        mirna_target_source=args.mirna_target_source,
        mirna_target_file=args.mirna_target_file,
        tarbase_file=args.tarbase_file,
        mirtarbase_file=args.mirtarbase_file,
        mirna_target_chunksize=args.mirna_target_chunksize,
        include_kegg=not args.no_kegg,
        include_reactome=not args.no_reactome,
    )
    print(f"Gene/pathway table : {result['membership']}")
    print(f"miRNA target table : {result['mirna_targets']}")
    print(f"Pathway edge table : {result['edges']}")
    for key, value in result["qc"].items():
        if key != "source_counts":
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
