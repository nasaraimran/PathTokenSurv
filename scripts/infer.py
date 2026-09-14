from __future__ import annotations

# Allow direct execution of scripts from a source checkout without requiring
# an editable installation first (e.g., ``python scripts/run_synthetic.py``).
import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse

from pathtokensurv.inference import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PathTokenSurv inference.")
    parser.add_argument("--artifacts", required=True, help="Training artifact directory.")
    parser.add_argument("--data-dir", required=True, help="Directory with inference TSV files.")
    parser.add_argument("--output", default="outputs/inference_predictions.csv")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    frame = run_inference(args.artifacts, args.data_dir, args.output, args.device)
    print(frame.head().to_string(index=False))
    print(f"Saved {len(frame)} predictions to {args.output}")


if __name__ == "__main__":
    main()
