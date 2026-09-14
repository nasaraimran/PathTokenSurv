from __future__ import annotations

# Allow direct execution from a source checkout.
import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import platform
from pathlib import Path

import torch

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pathtokensurv import __version__
from pathtokensurv.config import ExperimentConfig
from pathtokensurv.data.raw import load_raw_cohort
from pathtokensurv.data.splitting import joint_strata, outer_folds
from pathtokensurv.experiment import run_single_split
from pathtokensurv.utils.io import save_json
from pathtokensurv.utils.device import resolve_device


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_inner_validation_split(
    outer_train: np.ndarray,
    cancers: np.ndarray,
    events: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = joint_strata(cancers[outer_train], events[outer_train])
    _, counts = np.unique(labels, return_counts=True)
    stratify = labels if len(counts) > 1 and counts.min() >= 2 else None
    fit_local, val_local = train_test_split(
        np.arange(len(outer_train)),
        test_size=validation_fraction,
        random_state=seed,
        stratify=stratify,
    )
    return np.sort(outer_train[fit_local]), np.sort(outer_train[val_local])


def save_split_artifacts(
    raw,
    config: ExperimentConfig,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    output_dir: Path,
) -> None:
    assignments = []
    for split_name, indices in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        block = raw.outcomes.iloc[indices]
        for row_idx, patient_id in zip(indices, raw.patient_ids[indices]):
            assignments.append(
                {
                    "patient_id": str(patient_id),
                    "raw_index": int(row_idx),
                    "split": split_name,
                    "cancer_type": str(raw.outcomes.iloc[row_idx][config.data.cancer_col]),
                    "event": int(raw.outcomes.iloc[row_idx][config.data.event_col]),
                    "time": float(raw.outcomes.iloc[row_idx][config.data.time_col]),
                }
            )
    frame = pd.DataFrame(assignments)
    frame.to_csv(output_dir / "split_assignments.csv", index=False)

    summary = (
        frame.groupby(["split", "cancer_type", "event"], dropna=False)
        .size()
        .reset_index(name="patients")
        .sort_values(["split", "cancer_type", "event"])
    )
    summary.to_csv(output_dir / "split_summary_by_cancer_event.csv", index=False)
    save_json(
        {
            "train_patients": int(len(train_idx)),
            "validation_patients": int(len(val_idx)),
            "test_patients": int(len(test_idx)),
            "train_events": int(frame.loc[frame.split == "train", "event"].sum()),
            "validation_events": int(frame.loc[frame.split == "validation", "event"].sum()),
            "test_events": int(frame.loc[frame.split == "test", "event"].sum()),
            "unique_cancers_train": int(frame.loc[frame.split == "train", "cancer_type"].nunique()),
            "unique_cancers_validation": int(frame.loc[frame.split == "validation", "cancer_type"].nunique()),
            "unique_cancers_test": int(frame.loc[frame.split == "test", "cancer_type"].nunique()),
        },
        output_dir / "split_summary.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one leakage-controlled outer fold using the same outer split definition as QC/dry-run."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=1, help="1-based outer fold to train.")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Optional last_model.pt/best_model.pt checkpoint from the same fold. v1.5.7 is training-compatible with v1.5.6; do not resume across different outer folds.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Abort before preprocessing/training unless PyTorch can use an NVIDIA CUDA device.",
    )
    args = parser.parse_args()

    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must lie between 0 and 0.5.")

    config_path = Path(args.config)
    base = ExperimentConfig.load(config_path)
    device = resolve_device(base.training.device)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError(
            "CUDA was required but this Python environment cannot use CUDA. "
            "Install a CUDA-enabled PyTorch build and verify torch.cuda.is_available() before rerunning."
        )
    raw = load_raw_cohort(base.data, require_outcomes=True)
    cancers = raw.outcomes[base.data.cancer_col].astype(str).to_numpy()
    events = raw.outcomes[base.data.event_col].astype(int).to_numpy()

    folds = list(outer_folds(cancers, events, args.outer_folds, base.training.seed))
    if not 1 <= args.fold <= len(folds):
        raise ValueError(f"--fold must be between 1 and {len(folds)}.")
    outer_train, outer_test = folds[args.fold - 1]

    validation_seed = base.training.seed + args.fold
    train_idx, val_idx = make_inner_validation_split(
        outer_train,
        cancers,
        events,
        validation_fraction=args.validation_fraction,
        seed=validation_seed,
    )

    config = deepcopy(base)
    model_seed = base.training.seed + args.fold * 1009
    config.training.seed = model_seed
    output_dir = Path(
        args.output_dir
        or (Path(base.output_dir) / "single_outer_fold" / f"fold_{args.fold:02d}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    save_split_artifacts(raw, base, train_idx, val_idx, outer_test, output_dir)

    pathway_path = Path(base.data.data_dir) / base.data.pathway_file
    environment = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "resolved_device": str(device),
    }
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        props = torch.cuda.get_device_properties(device)
        environment.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_compute_capability": f"{major}.{minor}",
                "gpu_total_memory_bytes": int(props.total_memory),
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )

    manifest = {
        "status": "STARTED",
        "package_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "outer_folds": int(args.outer_folds),
        "outer_fold": int(args.fold),
        "outer_split_seed": int(base.training.seed),
        "validation_fraction": float(args.validation_fraction),
        "validation_split_seed": int(validation_seed),
        "model_seed": int(model_seed),
        "train_patients": int(len(train_idx)),
        "validation_patients": int(len(val_idx)),
        "test_patients": int(len(outer_test)),
        "config_source": str(config_path),
        "config_sha256": sha256_file(config_path),
        "pathway_mapping": str(pathway_path),
        "pathway_mapping_sha256": sha256_file(pathway_path) if pathway_path.exists() else None,
        "primary_protocol_frozen": True,
        "resume_from": str(args.resume_from) if args.resume_from else None,
        "performance_engine": "v1.5.6_chunked_tokenizer_one_pass_validation",
        "evaluation_engine": "v1.5.7_stratified_discrimination_ipcw_robustness",
        "training_protocol_changed_from_v1.5.6": False,
        "environment": environment,
    }
    save_json(manifest, output_dir / "experiment_manifest.json")

    print("PathTokenSurv single-outer-fold training")
    print(f"  package: {__version__}")
    print(f"  outer fold: {args.fold}/{args.outer_folds}")
    print(f"  train / validation / test: {len(train_idx)} / {len(val_idx)} / {len(outer_test)}")
    print(f"  model seed: {model_seed}")
    print(f"  device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        major, minor = torch.cuda.get_device_capability(device)
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        print(f"  compute capability: {major}.{minor}")
        print(f"  GPU memory: {props.total_memory / (1024**3):.2f} GiB")
        print(f"  PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"  output: {output_dir}")

    result = run_single_split(
        raw,
        config,
        train_idx,
        val_idx,
        outer_test,
        output_dir,
        resume_from=args.resume_from,
        run_extended_robustness=True,
    )
    manifest.update(
        {
            "status": "PASS",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "test_metrics": result.metrics,
        }
    )
    save_json(manifest, output_dir / "experiment_manifest.json")

    print("\nHeld-out outer-test metrics")
    for name, value in result.metrics.items():
        if isinstance(value, (float, int)):
            print(f"  {name}: {value:.6f}")
    print(f"\nSingle outer fold completed. Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
