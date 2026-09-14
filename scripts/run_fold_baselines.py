from __future__ import annotations

# Allow direct execution from a source checkout.
import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pathtokensurv import __version__
from pathtokensurv.baselines import train_baseline
from pathtokensurv.config import ExperimentConfig
from pathtokensurv.data.preprocessing import FoldPreprocessor
from pathtokensurv.data.raw import load_raw_cohort
from pathtokensurv.data.time import TimeDiscretizer
from pathtokensurv.utils.device import resolve_device


def _indices(raw, assignments: pd.DataFrame, split: str) -> np.ndarray:
    ids = assignments.loc[assignments["split"] == split, "patient_id"].astype(str).tolist()
    lookup = {str(pid): idx for idx, pid in enumerate(raw.patient_ids)}
    missing = [pid for pid in ids if pid not in lookup]
    if missing:
        raise ValueError(f"Missing {len(missing)} patient IDs for split '{split}'.")
    return np.asarray([lookup[pid] for pid in ids], dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train cancer-only and clinical-only discrete-time baselines on the exact "
            "train/validation/test split and time bins of a completed PathTokenSurv outer fold."
        )
    )
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["cancer_only", "clinical_only"],
        choices=["cancer_only", "clinical_only"],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    artifact_dir = Path(args.artifacts)
    config = ExperimentConfig.load(artifact_dir / "config.json")
    if args.data_dir is not None:
        config.data.data_dir = str(args.data_dir)
    device = resolve_device(args.device)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA was required for baseline training but is unavailable.")

    raw = load_raw_cohort(config.data, require_outcomes=True)
    assignments = pd.read_csv(artifact_dir / "split_assignments.csv")
    train_idx = _indices(raw, assignments, "train")
    val_idx = _indices(raw, assignments, "validation")
    test_idx = _indices(raw, assignments, "test")

    preprocessor = FoldPreprocessor.load(artifact_dir / "preprocessor.pkl")
    discretizer = TimeDiscretizer.load(artifact_dir / "time_discretizer.json")
    config.model.num_time_bins = discretizer.num_bins
    train = preprocessor.transform(raw, train_idx)
    val = preprocessor.transform(raw, val_idx)
    test = preprocessor.transform(raw, test_idx)

    print("PathTokenSurv frozen-split baseline evaluation")
    print(f"  package: {__version__}")
    print(f"  device: {device}")
    print(f"  train / validation / test: {len(train)} / {len(val)} / {len(test)}")
    print(f"  time bins: {discretizer.num_bins}")

    for baseline in args.baselines:
        out = artifact_dir / "baselines" / baseline
        print(f"\nTraining baseline: {baseline}")
        result = train_baseline(
            baseline=baseline,
            config=config,
            train=train,
            val=val,
            test=test,
            discretizer=discretizer,
            category_cardinalities=preprocessor.category_cardinalities,
            num_cancers=preprocessor.num_cancers,
            output_dir=out,
            device=device,
        )
        print(f"  best epoch: {result.best_epoch}")
        print(f"  best validation score: {result.best_score:.6f}")
        print(f"  test C-index: {result.metrics.get('c_index', float('nan')):.6f}")
        print(f"  test IBS: {result.metrics.get('ibs', float('nan')):.6f}")
        print(f"  test mean tdAUC: {result.metrics.get('mean_time_dependent_auc', float('nan')):.6f}")
        print(f"  output: {result.output_dir}")


if __name__ == "__main__":
    main()
