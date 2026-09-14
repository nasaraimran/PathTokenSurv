from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse

from pathtokensurv.ablation import aggregate_architecture_ablations


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate five-fold frozen PathTokenSurv architecture ablations.")
    parser.add_argument("--primary-fold-dirs", nargs=5, required=True)
    parser.add_argument("--ablation-root", required=True)
    parser.add_argument("--output-dir", default="outputs/tcga_architecture_ablation_summary_v1.5.9")
    args = parser.parse_args()
    manifest = aggregate_architecture_ablations(
        primary_fold_dirs=args.primary_fold_dirs,
        ablation_root=args.ablation_root,
        output_dir=args.output_dir,
    )
    print("PathTokenSurv architecture ablation aggregation: PASS")
    print(f"  completed ablation runs: {manifest['completed_ablation_runs']}")
    print(f"  output: {args.output_dir}")


if __name__ == "__main__":
    main()
