from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil

import numpy as np
import pandas as pd
import torch

from pathtokensurv import __version__
from pathtokensurv.config import ExperimentConfig
from pathtokensurv.data.pathways import PathwaySpec
from pathtokensurv.data.raw import load_raw_cohort
from pathtokensurv.frozen_ablation_experiment import run_frozen_artifact_ablation
from pathtokensurv.pathway_control import (
    CONTROL_DESCRIPTION,
    CONTROL_NAME,
    make_feature_permuted_pathway_spec,
)
from pathtokensurv.utils.device import resolve_device
from pathtokensurv.utils.io import save_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def indices_from_primary_split(raw, primary_artifacts: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = primary_artifacts / "split_assignments.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    split = pd.read_csv(path, dtype={"patient_id": str, "split": str})
    if split["patient_id"].duplicated().any():
        raise ValueError("Primary split_assignments.csv contains duplicate patient IDs.")
    index_by_patient = {str(pid): idx for idx, pid in enumerate(raw.patient_ids)}
    output = []
    for name in ["train", "validation", "test"]:
        ids = split.loc[split["split"] == name, "patient_id"].astype(str).tolist()
        missing = [pid for pid in ids if pid not in index_by_patient]
        if missing:
            raise KeyError(f"{len(missing)} patients from frozen {name} split are absent from the loaded cohort.")
        output.append(np.asarray([index_by_patient[pid] for pid in ids], dtype=int))
    if sum(len(x) for x in output) != len(split):
        unknown = sorted(set(split["split"]) - {"train", "validation", "test"})
        raise ValueError(f"Unknown split labels in primary assignment file: {unknown}")
    return tuple(output)  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one frozen-fold feature-permuted pathway-token control.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--primary-artifacts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--permutation-seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    primary_artifacts = Path(args.primary_artifacts)
    primary_manifest_path = primary_artifacts / "experiment_manifest.json"
    if not primary_manifest_path.exists():
        raise FileNotFoundError(primary_manifest_path)
    primary_manifest = json.loads(primary_manifest_path.read_text(encoding="utf-8"))
    if primary_manifest.get("status") != "PASS":
        raise RuntimeError("Primary fold manifest must have status PASS before pathway-token control training.")
    if not bool(primary_manifest.get("primary_protocol_frozen", False)):
        raise RuntimeError("Primary fold manifest does not mark the protocol as frozen.")
    outer_fold = int(primary_manifest["outer_fold"])

    config.training.seed = int(primary_manifest.get("model_seed", config.training.seed + outer_fold * 1009))
    config.training.device = args.device
    device = resolve_device(config.training.device)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA was required but the resolved device is not CUDA.")

    raw = load_raw_cohort(config.data, require_outcomes=True)
    train_idx, val_idx, test_idx = indices_from_primary_split(raw, primary_artifacts)
    expected_sizes = (
        int(primary_manifest.get("train_patients", len(train_idx))),
        int(primary_manifest.get("validation_patients", len(val_idx))),
        int(primary_manifest.get("test_patients", len(test_idx))),
    )
    if (len(train_idx), len(val_idx), len(test_idx)) != expected_sizes:
        raise ValueError(
            "Frozen split sizes differ from the primary manifest: "
            f"found {(len(train_idx), len(val_idx), len(test_idx))}, expected {expected_sizes}."
        )

    primary_spec_path = primary_artifacts / "pathway_spec.json"
    primary_spec = PathwaySpec.load(primary_spec_path)
    control_spec, permutation_metadata, permutations = make_feature_permuted_pathway_spec(
        primary_spec, seed=args.permutation_seed
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(primary_artifacts / "split_assignments.csv", output_dir / "split_assignments.csv")
    if (primary_artifacts / "split_summary.json").exists():
        shutil.copy2(primary_artifacts / "split_summary.json", output_dir / "split_summary.json")
    if (primary_artifacts / "split_summary_by_cancer_event.csv").exists():
        shutil.copy2(primary_artifacts / "split_summary_by_cancer_event.csv", output_dir / "split_summary_by_cancer_event.csv")
    shutil.copy2(primary_spec_path, output_dir / "primary_pathway_spec.json")
    control_spec.save(output_dir / "pathway_spec_permuted.json")
    save_json(permutation_metadata, output_dir / "pathway_permutation_metadata.json")
    save_json(permutations, output_dir / "pathway_feature_permutation_indices.json")

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
        environment.update({
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_compute_capability": f"{major}.{minor}",
            "gpu_total_memory_bytes": int(props.total_memory),
            "cudnn_version": torch.backends.cudnn.version(),
        })

    frozen_hashes = {}
    for name in ["preprocessor.pkl", "pathway_spec.json", "time_discretizer.json", "split_assignments.csv"]:
        frozen_hashes[name] = sha256_file(primary_artifacts / name)
    control_spec_hash = sha256_file(output_dir / "pathway_spec_permuted.json")

    manifest = {
        "status": "STARTED",
        "package_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "control": CONTROL_NAME,
        "control_description": CONTROL_DESCRIPTION,
        "secondary_sensitivity_analysis": True,
        "outer_fold": outer_fold,
        "primary_artifacts": str(primary_artifacts),
        "primary_model_seed": int(config.training.seed),
        "same_model_seed_as_primary": True,
        "permutation_seed": int(args.permutation_seed),
        "train_patients": int(len(train_idx)),
        "validation_patients": int(len(val_idx)),
        "test_patients": int(len(test_idx)),
        "frozen_primary_artifact_sha256": frozen_hashes,
        "control_pathway_spec_sha256": control_spec_hash,
        "primary_config_sha256": primary_manifest.get("config_sha256"),
        "primary_pathway_mapping_sha256": primary_manifest.get("pathway_mapping_sha256"),
        "primary_protocol_frozen": True,
        "reuses_primary_split": True,
        "reuses_primary_preprocessor": True,
        "reuses_primary_time_discretizer": True,
        "reuses_primary_pathway_names": True,
        "reuses_primary_pathway_adjacency": True,
        "reuses_primary_pathway_token_sizes": True,
        "biological_feature_to_pathway_correspondence_permuted": True,
        "permutation_invariant_validation": permutation_metadata["invariant_validation"],
        "resume_from": str(args.resume_from) if args.resume_from else None,
        "environment": environment,
    }
    save_json(manifest, output_dir / "pathway_control_manifest.json")

    print("PathTokenSurv frozen pathway-tokenization control")
    print(f"  package: {__version__}")
    print(f"  control: {CONTROL_NAME}")
    print(f"  outer fold: {outer_fold}/5")
    print(f"  train / validation / test: {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")
    print(f"  model seed: {config.training.seed} (same as primary fold)")
    print(f"  permutation seed: {args.permutation_seed}")
    print(f"  device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
    print(f"  output: {output_dir}")

    result = run_frozen_artifact_ablation(
        raw=raw,
        config=config,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        primary_artifacts=primary_artifacts,
        output_dir=output_dir,
        resume_from=args.resume_from,
        run_extended_robustness=True,
        pathway_spec_override=control_spec,
    )
    manifest.update({
        "status": "PASS",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "test_metrics": result.metrics,
    })
    save_json(manifest, output_dir / "pathway_control_manifest.json")

    print("\nHeld-out pathway-token control metrics")
    for key in ["c_index", "ibs"]:
        if key in result.metrics:
            print(f"  {key}: {result.metrics[key]:.6f}")
    print(f"Pathway-token control fold completed: {output_dir}")


if __name__ == "__main__":
    main()
