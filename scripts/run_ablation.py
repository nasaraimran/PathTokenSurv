from __future__ import annotations

# Allow direct execution of scripts from a source checkout without requiring
# an editable installation first (e.g., ``python scripts/run_synthetic.py``).
import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from pathtokensurv.config import ExperimentConfig
from pathtokensurv.data.raw import load_raw_cohort
from pathtokensurv.data.splitting import train_val_test_split
from pathtokensurv.experiment import run_single_split


VARIANTS = {
    "full": {},
    "no_pathway_bias": {"use_pathway_bias": False},
    "no_reconstruction": {"use_reconstruction": False},
    "no_subset_consistency": {"use_subset_consistency": False},
    "shared_cancer_baseline": {"use_cancer_deviation": False},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run core PathTokenSurv ablations.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    base = ExperimentConfig.load(args.config)
    raw = load_raw_cohort(base.data)
    cancers = raw.outcomes[base.data.cancer_col].astype(str).to_numpy()
    events = raw.outcomes[base.data.event_col].astype(int).to_numpy()
    train_idx, val_idx, test_idx = train_val_test_split(cancers, events, base.training.seed)
    root = Path(base.output_dir) / "ablations"
    root.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant_index, (name, overrides) in enumerate(VARIANTS.items()):
        config = deepcopy(base)
        config.training.seed = base.training.seed + variant_index * 1009
        for key, value in overrides.items():
            setattr(config.model, key, value)
        result = run_single_split(raw, config, train_idx, val_idx, test_idx, root / name)
        rows.append({"variant": name, **result.metrics})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "ablation_metrics.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
