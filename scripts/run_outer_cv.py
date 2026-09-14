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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pathtokensurv.config import ExperimentConfig
from pathtokensurv.data.raw import load_raw_cohort
from pathtokensurv.data.splitting import joint_strata, outer_folds
from pathtokensurv.experiment import run_single_split
from pathtokensurv.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-controlled outer cross-validation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    base = ExperimentConfig.load(args.config)
    raw = load_raw_cohort(base.data)
    cancers = raw.outcomes[base.data.cancer_col].astype(str).to_numpy()
    events = raw.outcomes[base.data.event_col].astype(int).to_numpy()
    root = Path(base.output_dir) / "outer_cv"
    root.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    for fold, (outer_train, outer_test) in enumerate(
        outer_folds(cancers, events, args.folds, base.training.seed), start=1
    ):
        labels = joint_strata(cancers[outer_train], events[outer_train])
        unique, counts = np.unique(labels, return_counts=True)
        stratify = labels if counts.min() >= 2 else None
        inner_train_local, val_local = train_test_split(
            np.arange(len(outer_train)),
            test_size=0.15,
            random_state=base.training.seed + fold,
            stratify=stratify,
        )
        train_idx = outer_train[inner_train_local]
        val_idx = outer_train[val_local]
        config = deepcopy(base)
        config.training.seed = base.training.seed + fold * 1009
        fold_dir = root / f"fold_{fold:02d}"
        result = run_single_split(
            raw,
            config,
            train_idx,
            val_idx,
            outer_test,
            fold_dir,
            run_extended_robustness=True,
        )
        fold_rows.append({"fold": fold, **result.metrics})

    frame = pd.DataFrame(fold_rows)
    frame.to_csv(root / "fold_metrics.csv", index=False)
    summary = {}
    for column in frame.columns:
        if column == "fold":
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        summary[column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    save_json(summary, root / "summary.json")
    print(frame.to_string(index=False))
    print(f"Cross-validation outputs written to {root}")


if __name__ == "__main__":
    main()
