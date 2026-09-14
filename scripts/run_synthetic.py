from __future__ import annotations

# Allow direct execution of scripts from a source checkout without requiring
# an editable installation first (e.g., ``python scripts/run_synthetic.py``).
import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from pathlib import Path
import shutil

from pathtokensurv.config import ExperimentConfig
from pathtokensurv.data.raw import load_raw_cohort
from pathtokensurv.data.splitting import train_val_test_split
from pathtokensurv.data.synthetic import generate_synthetic_cohort
from pathtokensurv.experiment import run_single_split
from pathtokensurv.inference import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate data, train, test, and run inference.")
    parser.add_argument("--config", default="configs/synthetic.json")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    data_dir = Path(config.data.data_dir)
    output_dir = Path(config.output_dir)
    if args.clean:
        shutil.rmtree(data_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
    generate_synthetic_cohort(
        data_dir,
        n_patients=240,
        n_pathways=6,
        features_per_modality=24,
        n_cancers=4,
        seed=config.training.seed,
    )
    raw = load_raw_cohort(config.data)
    cancer = raw.outcomes[config.data.cancer_col].astype(str).to_numpy()
    events = raw.outcomes[config.data.event_col].astype(int).to_numpy()
    train_idx, val_idx, test_idx = train_val_test_split(cancer, events, config.training.seed)
    result = run_single_split(raw, config, train_idx, val_idx, test_idx, output_dir)
    print("Synthetic test metrics:")
    for name, value in result.metrics.items():
        print(f"  {name}: {value:.6f}")

    inference_output = output_dir / "inference_predictions.csv"
    inference = run_inference(output_dir, data_dir, inference_output, config.training.device)
    assert len(inference) == len(raw), "Inference did not return one row per patient."
    print(f"End-to-end validation passed. Artifacts are in {output_dir}")


if __name__ == "__main__":
    main()
