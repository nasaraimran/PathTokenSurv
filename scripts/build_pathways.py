from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse

from pathtokensurv.data.pathway_builder import build_pathway_json_from_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PathTokenSurv pathways.json from local pathway annotation tables.")
    parser.add_argument("--gene-membership", required=True)
    parser.add_argument("--mirna-targets")
    parser.add_argument("--pathway-edges")
    parser.add_argument("--output", required=True)
    parser.add_argument("--delimiter", default="\t")
    parser.add_argument("--min-genes-per-pathway", type=int, default=1,
                        help="Remove source pathways with fewer unique annotated genes than this value.")
    parser.add_argument("--max-genes-per-pathway", type=int, default=None,
                        help="Optionally remove very large source pathways above this number of unique genes.")
    parser.add_argument("--mirna-pathway-mode", choices=["enrichment", "any_target"], default="enrichment")
    parser.add_argument("--mirna-min-target-overlap", type=int, default=3)
    parser.add_argument("--mirna-enrichment-fdr", type=float, default=0.05)
    args = parser.parse_args()
    payload = build_pathway_json_from_tables(
        gene_membership_file=args.gene_membership,
        mirna_targets_file=args.mirna_targets,
        pathway_edges_file=args.pathway_edges,
        output_file=args.output,
        delimiter=args.delimiter,
        min_genes_per_pathway=args.min_genes_per_pathway,
        max_genes_per_pathway=args.max_genes_per_pathway,
        mirna_pathway_mode=args.mirna_pathway_mode,
        mirna_min_target_overlap=args.mirna_min_target_overlap,
        mirna_enrichment_fdr=args.mirna_enrichment_fdr,
    )
    meta = payload.get("metadata", {})
    print(f"Saved {len(payload['pathways'])} pathways to {args.output}")
    if meta:
        print(
            "Pathway-size filter: "
            f"source={meta.get('source_unique_pathways')} retained={meta.get('retained_pathways_after_size_filter')} "
            f"removed={meta.get('pathways_removed_by_size_filter')}"
        )


if __name__ == "__main__":
    main()
