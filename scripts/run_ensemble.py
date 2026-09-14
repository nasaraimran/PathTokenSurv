
from __future__ import annotations

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from pathtokensurv.config import ExperimentConfig
from pathtokensurv.data.raw import load_raw_cohort
from pathtokensurv.data.splitting import train_val_test_split
from pathtokensurv.experiment import run_single_split
from pathtokensurv.metrics import evaluate_survival_predictions, risk_from_survival
from pathtokensurv.utils.io import save_json


def deterministic_member_seed(base_seed: int, fold: int, member: int) -> int:
    return int(base_seed + fold * 10000 + member * 1000)


def load_split_indices(
    assignments_path: str | Path, patient_ids
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assignments = pd.read_csv(assignments_path)
    required = {"patient_id", "split"}
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(f"Split assignments are missing columns: {sorted(missing)}")
    if assignments["patient_id"].duplicated().any():
        raise ValueError("Split assignments contain duplicate patient IDs.")

    raw_index = {str(patient_id): idx for idx, patient_id in enumerate(patient_ids)}
    unknown = sorted(set(assignments["patient_id"].astype(str)).difference(raw_index))
    if unknown:
        raise ValueError(f"Split assignments contain {len(unknown)} unknown patient IDs.")

    indices = {}
    for name in ("train", "validation", "test"):
        selected = assignments.loc[assignments["split"] == name, "patient_id"].astype(str)
        if selected.empty:
            raise ValueError(f"Split assignments contain no {name} patients.")
        indices[name] = np.asarray([raw_index[patient_id] for patient_id in selected], dtype=int)
    return indices["train"], indices["validation"], indices["test"]


def load_member_predictions(
    member_dirs: list[Path],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    reference = None
    reference_columns = None
    survival_parts = []
    horizons = None

    for member_dir in member_dirs:
        path = member_dir / "test_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing completed member predictions: {path}")
        frame = pd.read_csv(path)
        survival_columns = [column for column in frame.columns if column.startswith("survival_")]
        if not survival_columns:
            raise ValueError(f"Member predictions contain no survival columns: {path}")
        identity_columns = ["patient_id", "time", "event"]
        if any(column not in frame for column in identity_columns):
            raise ValueError(f"Member predictions lack patient identity or outcome columns: {path}")

        if reference is None:
            reference = frame
            reference_columns = survival_columns
            horizons = np.asarray(
                [float(column.removeprefix("survival_")) for column in survival_columns],
                dtype=float,
            )
        elif survival_columns != reference_columns or not frame[identity_columns].equals(
            reference[identity_columns]
        ):
            raise ValueError(f"Ensemble member predictions are not aligned: {path}")
        survival_parts.append(frame[survival_columns].to_numpy(dtype=float))

    return reference, np.stack(survival_parts, axis=0), horizons


def main():
    parser = argparse.ArgumentParser(
        description="PathTokenSurv v1.7.1 deep ensemble runner"
    )

    parser.add_argument("--config", required=True)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--split-assignments",
        default=None,
        help="Saved primary-fold split_assignments.csv. If omitted, create a deterministic train/validation/test split.",
    )

    args = parser.parse_args()

    base = ExperimentConfig.load(args.config)

    members = args.members or base.training.ensemble_size
    if members < 2:
        raise ValueError("Ensemble requires at least two members.")

    raw = load_raw_cohort(base.data)

    cancers = raw.outcomes[base.data.cancer_col].astype(str).to_numpy()
    events = raw.outcomes[base.data.event_col].astype(int).to_numpy()

    if args.split_assignments:
        train_idx, val_idx, test_idx = load_split_indices(
            args.split_assignments, raw.patient_ids
        )
    else:
        train_idx, val_idx, test_idx = train_val_test_split(
            cancers,
            events,
            base.training.seed,
        )

    ensemble_dir = Path(
        args.output_dir
        if args.output_dir
        else Path(base.output_dir) / f"ensemble_fold_{args.fold:02d}"
    )

    ensemble_dir.mkdir(parents=True, exist_ok=True)

    member_dirs = []

    for member in range(members):

        member_dir = ensemble_dir / f"member_{member:02d}"
        member_dirs.append(member_dir)

        predictions_file = member_dir / "test_predictions.csv"

        if predictions_file.exists():
            print(
                f"Skipping completed ensemble member {member:02d}"
            )
            continue

        config = deepcopy(base)

        config.training.seed = deterministic_member_seed(
            base.training.seed,
            args.fold,
            member,
        )

        print(
            f"\nTraining ensemble member {member+1}/{members}"
        )
        print(
            f"Seed: {config.training.seed}"
        )

        run_single_split(
            raw,
            config,
            train_idx,
            val_idx,
            test_idx,
            member_dir,
        )

    reference, survival_stack, horizons = load_member_predictions(member_dirs)

    mean_survival = survival_stack.mean(axis=0)
    uncertainty = survival_stack.var(
        axis=0,
        ddof=1,
    )

    metrics = evaluate_survival_predictions(
        train_times=raw.outcomes.iloc[train_idx][base.data.time_col].to_numpy(),
        train_events=raw.outcomes.iloc[train_idx][base.data.event_col].to_numpy(),
        test_times=reference["time"].to_numpy(),
        test_events=reference["event"].to_numpy(),
        survival=mean_survival,
        horizons=horizons,
        calibration_bins=base.evaluation.calibration_bins,
    )

    save_json(
        metrics,
        ensemble_dir / "ensemble_metrics.json",
    )

    frame = pd.DataFrame(
        {
            "patient_id": reference["patient_id"],
            "time": reference["time"],
            "event": reference["event"],
            "risk_score": risk_from_survival(
                mean_survival,
                horizons,
            ),
            "mean_uncertainty": uncertainty.mean(axis=1),
        }
    )
    for idx, horizon in enumerate(horizons):
        frame[f"survival_{float(horizon):.6g}"] = mean_survival[:, idx]

    frame.to_csv(
        ensemble_dir / "ensemble_predictions.csv",
        index=False,
    )

    save_json(
        {
            "members": members,
            "fold": args.fold,
            "seed_base": base.training.seed,
            "member_seeds": [
                deterministic_member_seed(base.training.seed, args.fold, member)
                for member in range(members)
            ],
            "split_assignments": args.split_assignments,
            "status": "completed",
        },
        ensemble_dir / "ensemble_manifest.json",
    )

    print(
        f"\nEnsemble complete: {ensemble_dir}"
    )


if __name__ == "__main__":
    main()
