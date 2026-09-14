from __future__ import annotations

# Allow direct execution of scripts from a source checkout without requiring
# an editable installation first (e.g., ``python scripts/run_synthetic.py``).
import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse

from pathtokensurv.data.synthetic import generate_synthetic_cohort


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic PathTokenSurv cohort.")
    parser.add_argument("--output", default="data/synthetic")
    parser.add_argument("--patients", type=int, default=320)
    parser.add_argument("--pathways", type=int, default=8)
    parser.add_argument("--features", type=int, default=40)
    parser.add_argument("--cancers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    path = generate_synthetic_cohort(
        args.output,
        n_patients=args.patients,
        n_pathways=args.pathways,
        features_per_modality=args.features,
        n_cancers=args.cancers,
        seed=args.seed,
    )
    print(f"Synthetic cohort written to {path}")


if __name__ == "__main__":
    main()
