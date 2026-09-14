from __future__ import annotations

# Allow direct execution from a source checkout.
import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from pathlib import Path

from pathtokensurv import __version__
from pathtokensurv.missingness_aggregation import aggregate_missingness_robustness


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate PathTokenSurv v1.5.7+ natural-missingness and controlled "
            "complete-case modality-drop results across completed outer folds. "
            "This is evaluation-only and performs no model training."
        )
    )
    parser.add_argument("--fold-dirs", nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        default="outputs/tcga_missingness_robustness_v1.5.8",
    )
    args = parser.parse_args()

    result = aggregate_missingness_robustness(args.fold_dirs, args.output_dir)
    natural = result["natural_summary"]
    dropout = result["dropout_summary"]

    print("PathTokenSurv missingness robustness aggregation")
    print(f"  package: {__version__}")
    print(f"  folds: {len(set(result['natural_fold']['fold']))}")
    print("\nNatural missingness aggregate groups")
    cols = [
        "group",
        "total_patients",
        "total_events",
        "c_index_mean",
        "cancer_stratified_c_index_mean",
        "ibs_mean",
        "mean_time_dependent_auc_mean",
    ]
    print(natural[natural["group_type"] == "aggregate"][cols].to_string(index=False))
    print("\nControlled complete-case modality dropout")
    cols = [
        "condition",
        "c_index_mean",
        "cancer_stratified_c_index_mean",
        "ibs_mean",
        "mean_time_dependent_auc_mean",
        "degradation_c_index_mean",
        "degradation_ibs_mean",
    ]
    print(dropout[cols].to_string(index=False))
    print(f"\nMissingness robustness artifacts: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
