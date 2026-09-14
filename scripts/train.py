from __future__ import annotations

# Allow direct execution of scripts from a source checkout without requiring
# an editable installation first (e.g., ``python scripts/run_synthetic.py``).
import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse

from pathtokensurv.config import ExperimentConfig
from pathtokensurv.data.raw import load_raw_cohort
from pathtokensurv.data.splitting import train_val_test_split
from pathtokensurv.experiment import run_single_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate PathTokenSurv.")
    parser.add_argument("--config", required=True, help="Path to a JSON experiment config.")
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    raw = load_raw_cohort(config.data, require_outcomes=True)
    cancer = raw.outcomes[config.data.cancer_col].astype(str).to_numpy()
    events = raw.outcomes[config.data.event_col].astype(int).to_numpy()
    train_idx, val_idx, test_idx = train_val_test_split(
        cancer,
        events,
        seed=config.training.seed,
    )
    result = run_single_split(raw, config, train_idx, val_idx, test_idx)
    print("Test metrics:")
    for name, value in result.metrics.items():
        print(f"  {name}: {value:.6f}")
    print(f"Artifacts: {result.output_dir}")


if __name__ == "__main__":
    main()
