from __future__ import annotations

# Allow direct execution of scripts from a source checkout without requiring
# an editable installation first (e.g., ``python scripts/run_synthetic.py``).
import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from pathlib import Path

from pathtokensurv.robustness import evaluate_missingness, evaluate_modality_subsets


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate modality subsets and controlled missingness.")
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/robustness")
    parser.add_argument("--rates", nargs="+", type=float, default=[0.0, 0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = evaluate_missingness(
        args.artifacts,
        args.data_dir,
        rates=args.rates,
        repeats=args.repeats,
        device_name=args.device,
    )
    missing.to_csv(output_dir / "missingness_metrics.csv", index=False)
    subsets = evaluate_modality_subsets(
        args.artifacts,
        args.data_dir,
        device_name=args.device,
    )
    subsets.to_csv(output_dir / "modality_subset_metrics.csv", index=False)
    print(missing.groupby("missing_rate")[["c_index", "ibs"]].mean().to_string())
    print(subsets[["modalities", "n_patients", "c_index", "ibs"]].to_string(index=False))


if __name__ == "__main__":
    main()
