from __future__ import annotations

# Allow direct execution of scripts from a source checkout without requiring
# an editable installation first (e.g., ``python scripts/run_synthetic.py``).
import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse

from pathtokensurv.evaluation import evaluate_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved model on a labeled cohort.")
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/external_evaluation")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    metrics = evaluate_artifact(args.artifacts, args.data_dir, args.output_dir, args.device)
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
