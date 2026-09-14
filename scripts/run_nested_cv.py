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
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pathtokensurv.config import ExperimentConfig
from pathtokensurv.data.raw import load_raw_cohort
from pathtokensurv.data.splitting import joint_strata, outer_folds
from pathtokensurv.experiment import run_single_split
from pathtokensurv.utils.io import save_json


def apply_overrides(config: ExperimentConfig, overrides: Dict[str, Any]) -> ExperimentConfig:
    updated = deepcopy(config)
    for section_name, section_values in overrides.items():
        if section_name not in {"data", "model", "training", "evaluation"}:
            raise ValueError(f"Unsupported override section: {section_name}")
        section = getattr(updated, section_name)
        for key, value in section_values.items():
            if not hasattr(section, key):
                raise ValueError(f"Unknown setting {section_name}.{key}")
            setattr(section, key, value)
    updated.validate()
    return updated


def early_stop_split(indices: np.ndarray, cancers: np.ndarray, events: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    labels = joint_strata(cancers[indices], events[indices])
    _, counts = np.unique(labels, return_counts=True)
    stratify = labels if counts.min() >= 2 else None
    fit_local, val_local = train_test_split(
        np.arange(len(indices)),
        test_size=0.15,
        random_state=seed,
        stratify=stratify,
    )
    return indices[fit_local], indices[val_local]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nested cross-validation with a compact candidate grid.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--search", required=True, help="JSON file containing a 'candidates' list.")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    args = parser.parse_args()

    base = ExperimentConfig.load(args.config)
    search = json.loads(Path(args.search).read_text(encoding="utf-8"))
    candidates = search.get("candidates", [])
    if not candidates:
        raise ValueError("Search file must contain at least one candidate.")

    raw = load_raw_cohort(base.data)
    cancers = raw.outcomes[base.data.cancer_col].astype(str).to_numpy()
    events = raw.outcomes[base.data.event_col].astype(int).to_numpy()
    root = Path(base.output_dir) / "nested_cv"
    root.mkdir(parents=True, exist_ok=True)

    outer_rows = []
    for outer_number, (outer_train, outer_test) in enumerate(
        outer_folds(cancers, events, args.outer_folds, base.training.seed), start=1
    ):
        outer_dir = root / f"outer_{outer_number:02d}"
        outer_dir.mkdir(parents=True, exist_ok=True)
        candidate_rows = []

        for candidate_number, candidate in enumerate(candidates):
            name = candidate.get("name", f"candidate_{candidate_number:02d}")
            overrides = candidate.get("overrides", {})
            candidate_scores = []
            for inner_number, (inner_train_local, inner_test_local) in enumerate(
                outer_folds(
                    cancers[outer_train],
                    events[outer_train],
                    args.inner_folds,
                    base.training.seed + outer_number * 100 + candidate_number,
                ),
                start=1,
            ):
                inner_train_full = outer_train[inner_train_local]
                inner_test = outer_train[inner_test_local]
                fit_idx, early_val_idx = early_stop_split(
                    inner_train_full,
                    cancers,
                    events,
                    seed=base.training.seed + outer_number * 1000 + inner_number,
                )
                config = apply_overrides(base, overrides)
                config.training.seed = base.training.seed + outer_number * 10000 + candidate_number * 100 + inner_number
                run_dir = outer_dir / "inner_search" / name / f"fold_{inner_number:02d}"
                result = run_single_split(raw, config, fit_idx, early_val_idx, inner_test, run_dir)
                score = float(result.metrics["c_index"] - result.metrics["ibs"])
                candidate_scores.append(score)
                candidate_rows.append(
                    {
                        "candidate": name,
                        "inner_fold": inner_number,
                        "score": score,
                        "c_index": result.metrics["c_index"],
                        "ibs": result.metrics["ibs"],
                    }
                )
            candidate["mean_score"] = float(np.mean(candidate_scores))

        candidate_frame = pd.DataFrame(candidate_rows)
        candidate_frame.to_csv(outer_dir / "inner_search_metrics.csv", index=False)
        selected = max(candidates, key=lambda item: item["mean_score"])
        selected_name = selected.get("name", "selected")
        save_json(selected, outer_dir / "selected_candidate.json")

        fit_idx, early_val_idx = early_stop_split(
            outer_train,
            cancers,
            events,
            seed=base.training.seed + outer_number * 1009,
        )
        final_config = apply_overrides(base, selected.get("overrides", {}))
        final_config.training.seed = base.training.seed + outer_number * 1009
        final_result = run_single_split(
            raw,
            final_config,
            fit_idx,
            early_val_idx,
            outer_test,
            outer_dir / "final_model",
        )
        outer_rows.append(
            {
                "outer_fold": outer_number,
                "selected_candidate": selected_name,
                **final_result.metrics,
            }
        )

    outer_frame = pd.DataFrame(outer_rows)
    outer_frame.to_csv(root / "outer_fold_metrics.csv", index=False)
    summary = {
        "mean_c_index": float(outer_frame["c_index"].mean()),
        "std_c_index": float(outer_frame["c_index"].std(ddof=1)),
        "mean_ibs": float(outer_frame["ibs"].mean()),
        "std_ibs": float(outer_frame["ibs"].std(ddof=1)),
    }
    save_json(summary, root / "summary.json")
    print(outer_frame.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
