from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse

from pathtokensurv.pathway_control import aggregate_pathway_token_control


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the five-fold v1.6.0 pathway-tokenization control.")
    parser.add_argument("--primary-fold-dirs", nargs=5, required=True)
    parser.add_argument("--control-root", required=True)
    parser.add_argument("--output-dir", default="outputs/tcga_pathway_token_control_summary_v1.6.0")
    args = parser.parse_args()
    manifest = aggregate_pathway_token_control(
        primary_fold_dirs=args.primary_fold_dirs,
        control_root=args.control_root,
        output_dir=args.output_dir,
    )
    print("PathTokenSurv pathway-tokenization control aggregation: PASS")
    print(f"  completed control runs: {manifest['completed_control_runs']}")
    print(f"  output: {args.output_dir}")


if __name__ == "__main__":
    main()
