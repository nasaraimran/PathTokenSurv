from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from pathlib import Path

from pathtokensurv.data.annotation_sources import (
    MIRTARBASE_RELEASE_PAGE,
    TARBASE_PORTAL,
    download_source_bundle,
    prepare_unified_annotations,
)
from pathtokensurv.data.pathway_builder import build_pathway_json_from_tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end construction of PathTokenSurv pathways.json from NCBI, Reactome, KEGG and experimentally supported miRNA-target annotations."
    )
    parser.add_argument("--raw-dir", default="annotations/raw")
    parser.add_argument("--processed-dir", default="annotations/processed")
    parser.add_argument("--output", default="data/tcga_pancancer_raw/pathways.json")
    parser.add_argument("--download", action="store_true", help="Download NCBI/Reactome/KEGG before processing.")
    parser.add_argument("--reactome-direct-only", action="store_true")
    parser.add_argument("--no-kegg", action="store_true")
    parser.add_argument("--no-reactome", action="store_true")

    # v1.5.2: TarBase v9 is primary; miRTarBase remains optional.
    parser.add_argument("--mirna-target-source", choices=["tarbase", "mirtarbase", "none"], default="tarbase")
    parser.add_argument("--mirna-target-file", default=None, help="Generic local miRNA-target export path; overrides provider-specific file options.")
    parser.add_argument("--tarbase-file", default=None, help="Local TarBase v9 CSV/TSV/TXT/XLSX export. Recommended for the main experiment.")
    parser.add_argument("--tarbase-url", default=None, help="Optional exact TarBase export URL copied from the official portal; used only with --download.")
    parser.add_argument("--mirna-target-chunksize", type=int, default=250000, help="Chunk size for large TarBase delimited exports.")
    parser.add_argument("--mirtarbase-release", default="10.0")
    parser.add_argument("--mirtarbase-file", default=None, help="Optional local miRTarBase hsa_MTI.csv for sensitivity/alternative analyses.")
    parser.add_argument("--download-mirtarbase", action="store_true", help="Attempt the legacy miRTarBase direct download (optional; server may be unavailable).")
    # Backward-compatible v1.5/v1.5.1 flag. It now simply suppresses any legacy attempt.
    parser.add_argument("--no-mirtarbase-download", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument("--min-genes-per-pathway", type=int, default=5)
    parser.add_argument("--max-genes-per-pathway", type=int, default=500)
    parser.add_argument("--mirna-pathway-mode", choices=["enrichment", "any_target"], default="enrichment",
                        help="Use pathway-enriched TarBase targets (recommended) or the legacy any-target rule.")
    parser.add_argument("--mirna-min-target-overlap", type=int, default=3,
                        help="Minimum unique miRNA target genes overlapping a pathway before enrichment testing.")
    parser.add_argument("--mirna-enrichment-fdr", type=float, default=0.05,
                        help="Per-miRNA Benjamini-Hochberg FDR threshold for miRNA-pathway enrichment.")
    parser.add_argument("--allow-empty-mirna", action="store_true", help="Only for a deliberate miRNA-ablation run.")
    args = parser.parse_args()

    raw_dir, processed_dir = Path(args.raw_dir), Path(args.processed_dir)
    if args.download:
        manifest = download_source_bundle(
            raw_dir,
            reactome_all_levels=not args.reactome_direct_only,
            include_kegg=not args.no_kegg,
            include_mirtarbase=(args.download_mirtarbase and not args.no_mirtarbase_download),
            mirtarbase_release=args.mirtarbase_release,
            tarbase_url=args.tarbase_url if args.mirna_target_source == "tarbase" else None,
            tarbase_file=(args.mirna_target_file or args.tarbase_file) if args.mirna_target_source == "tarbase" else None,
        )
        if args.download_mirtarbase and not args.no_mirtarbase_download:
            mt = manifest.get("mirtarbase")
            if (
                args.mirna_target_source == "mirtarbase"
                and args.mirna_target_file is None
                and args.mirtarbase_file is None
                and isinstance(mt, dict)
                and mt.get("status") == "download_failed"
            ):
                release_page = MIRTARBASE_RELEASE_PAGE.format(release=args.mirtarbase_release)
                raise RuntimeError(
                    "miRTarBase could not be downloaded automatically.\n"
                    f"Automatic download error: {mt.get('message', 'unknown error')}\n"
                    f"Official release page: {release_page}\n"
                    "For the main v1.5.2 analysis, use TarBase v9 instead, or provide a local miRTarBase file."
                )

    tables = prepare_unified_annotations(
        raw_dir,
        processed_dir,
        mirna_target_source=args.mirna_target_source,
        mirna_target_file=args.mirna_target_file,
        tarbase_file=args.tarbase_file,
        mirtarbase_file=args.mirtarbase_file,
        mirna_target_chunksize=args.mirna_target_chunksize,
        include_kegg=not args.no_kegg,
        include_reactome=not args.no_reactome,
    )

    if tables["qc"]["mirna_target_pairs"] == 0 and not args.allow_empty_mirna:
        if args.mirna_target_source == "tarbase":
            raise RuntimeError(
                "No human TarBase v9 miRNA-target pairs were loaded.\n"
                f"Official TarBase v9 portal: {TARBASE_PORTAL}\n"
                "Export/download the Homo sapiens interaction table, save it locally, then rerun with:\n"
                "  --mirna-target-source tarbase --tarbase-file .\\annotations\\raw\\TarBase_v9_human.csv\n"
                "The parser accepts CSV, TSV/TXT, CSV.GZ and XLSX exports.\n"
                "Do not use --allow-empty-mirna for the main PathTokenSurv experiment."
            )
        if args.mirna_target_source == "mirtarbase":
            release_page = MIRTARBASE_RELEASE_PAGE.format(release=args.mirtarbase_release)
            raise RuntimeError(
                "No human miRTarBase miRNA-target pairs were loaded.\n"
                f"Release page: {release_page}\n"
                "Provide a local hsa_MTI.csv with --mirtarbase-file, or use TarBase v9 as the primary provider."
            )
        raise RuntimeError("No miRNA-target provider is enabled. Use --allow-empty-mirna only for a deliberate ablation.")

    payload = build_pathway_json_from_tables(
        tables["membership"],
        args.output,
        mirna_targets_file=tables["mirna_targets"],
        pathway_edges_file=tables["edges"],
        min_genes_per_pathway=args.min_genes_per_pathway,
        max_genes_per_pathway=args.max_genes_per_pathway,
        mirna_pathway_mode=args.mirna_pathway_mode,
        mirna_min_target_overlap=args.mirna_min_target_overlap,
        mirna_enrichment_fdr=args.mirna_enrichment_fdr,
    )
    # Attach provider provenance to the final JSON metadata through the already-written
    # construction report. Downstream QC also reads annotation_source_qc.json.
    report = {
        "mirna_target_provider": tables["qc"].get("mirna_target_provider_label"),
        "mirna_target_source_file": tables["qc"].get("mirna_target_source_file"),
        "mirna_target_source_sha256": tables["qc"].get("mirna_target_source_sha256"),
        "mirna_target_pairs": tables["qc"].get("mirna_target_pairs"),
        "unique_mirnas": tables["qc"].get("unique_mirnas"),
        "unique_mirna_target_genes": tables["qc"].get("unique_mirna_target_genes"),
        "retained_pathways_after_size_filter": payload.get("metadata", {}).get("retained_pathways_after_size_filter"),
        "mirna_pathway_mode": payload.get("metadata", {}).get("mirna_pathway_mode"),
        "mirna_min_target_overlap": payload.get("metadata", {}).get("mirna_min_target_overlap"),
        "mirna_enrichment_fdr": payload.get("metadata", {}).get("mirna_enrichment_fdr"),
        "mirna_pathway_candidate_links": payload.get("metadata", {}).get("mirna_pathway_candidate_links"),
        "mirna_pathway_retained_links": payload.get("metadata", {}).get("mirna_pathway_retained_links"),
        "mirna_enrichment_score_definition": payload.get("metadata", {}).get("mirna_enrichment_score_definition"),
        "mirna_conservative_alias_count": payload.get("metadata", {}).get("mirna_conservative_alias_count"),
    }
    processed_dir.mkdir(parents=True, exist_ok=True)
    import json
    (processed_dir / "pathway_construction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved final pathway mapping to {args.output}")
    print(f"miRNA-target provider: {tables['qc'].get('mirna_target_provider_label')}")
    print(f"miRNA-target pairs loaded: {tables['qc']['mirna_target_pairs']:,}")
    print(f"Unique human miRNAs: {tables['qc']['unique_mirnas']:,}")
    print(f"Unique target genes: {tables['qc']['unique_mirna_target_genes']:,}")
    print(f"Retained source pathways: {payload['metadata']['retained_pathways_after_size_filter']:,}")
    print(f"miRNA pathway mode: {payload['metadata'].get('mirna_pathway_mode')}")
    print(f"miRNA pathway links retained: {payload['metadata'].get('mirna_pathway_retained_links', 0):,}")
    print(f"Construction report: {processed_dir / 'pathway_construction_report.json'}")


if __name__ == "__main__":
    main()
