from __future__ import annotations

from pathlib import Path
import json
from typing import Dict

import numpy as np
import pandas as pd


MOLECULAR = ("mrna", "mirna", "cnv")


def generate_synthetic_cohort(
    output_dir: str | Path,
    n_patients: int = 320,
    n_pathways: int = 8,
    features_per_modality: int = 40,
    n_cancers: int = 5,
    seed: int = 123,
    missingness: Dict[str, float] | None = None,
) -> Path:
    """Create a small multimodal survival cohort with known pathway signal."""

    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    missingness = missingness or {"mrna": 0.08, "mirna": 0.15, "cnv": 0.12}

    patient_ids = np.asarray([f"SYN-{i:05d}" for i in range(n_patients)])
    cancers = rng.integers(0, n_cancers, size=n_patients)
    latent = rng.normal(size=(n_patients, n_pathways)).astype(np.float32)

    age = np.clip(rng.normal(60, 11, size=n_patients), 20, 90)
    sex = rng.choice(["Female", "Male"], size=n_patients)
    race = rng.choice(["Group_A", "Group_B", "Group_C"], size=n_patients, p=[0.55, 0.30, 0.15])
    histology = np.asarray([f"Histology_{x}" for x in rng.integers(0, 6, size=n_patients)])

    cancer_effects = np.linspace(-0.45, 0.55, n_cancers)
    linear_risk = (
        0.80 * latent[:, 0]
        - 0.55 * latent[:, 2 % n_pathways]
        + 0.35 * latent[:, 4 % n_pathways]
        + 0.018 * (age - 60)
        + cancer_effects[cancers]
    )
    baseline_scale = 8.0
    event_times = rng.exponential(scale=baseline_scale / np.exp(np.clip(linear_risk, -2.0, 2.0))) + 0.1
    censor_times = rng.exponential(scale=10.0, size=n_patients) + 0.2
    observed_times = np.minimum(event_times, censor_times)
    events = (event_times <= censor_times).astype(int)

    outcomes = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "time": observed_times.round(5),
            "event": events,
            "cancer_type": [f"Cancer_{x}" for x in cancers],
        }
    )
    clinical = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "age": age.round(3),
            "sex": sex,
            "race": race,
            "histology": histology,
        }
    )

    pathways = [f"Pathway_{p}" for p in range(n_pathways)]
    modality_features: Dict[str, Dict[str, list]] = {m: {p: [] for p in pathways} for m in MOLECULAR}

    for modality_index, modality in enumerate(MOLECULAR):
        data = np.zeros((n_patients, features_per_modality), dtype=np.float32)
        feature_names = []
        for feature_idx in range(features_per_modality):
            pathway_idx = feature_idx % n_pathways
            name = f"{modality}_feature_{feature_idx:03d}"
            feature_names.append(name)
            modality_features[modality][pathways[pathway_idx]].append(name)
            loading = 0.75 + 0.1 * modality_index + rng.normal(0, 0.05)
            data[:, feature_idx] = loading * latent[:, pathway_idx] + rng.normal(0, 0.45, n_patients)
        # Put values on modality-specific scales to exercise fold-specific preprocessing.
        if modality == "mrna":
            data = 8.0 + 2.5 * data
        elif modality == "mirna":
            data = 4.0 + 1.2 * data
        else:
            data = np.tanh(data)
        frame = pd.DataFrame(data, columns=feature_names)
        frame.insert(0, "patient_id", patient_ids)
        missing_rows = rng.random(n_patients) < float(missingness.get(modality, 0.0))
        frame.loc[missing_rows, feature_names] = np.nan
        frame.to_csv(output_dir / f"{modality}.tsv", sep="\t", index=False)

    adjacency = np.zeros((n_pathways, n_pathways), dtype=float)
    for p in range(n_pathways):
        adjacency[p, (p + 1) % n_pathways] = 1.0
        adjacency[(p + 1) % n_pathways, p] = 1.0
    for p in range(0, n_pathways - 2, 3):
        adjacency[p, p + 2] = adjacency[p + 2, p] = 1.0

    pathway_payload = {
        "pathways": pathways,
        "modality_features": modality_features,
        "adjacency": adjacency.tolist(),
    }
    outcomes.to_csv(output_dir / "outcomes.tsv", sep="\t", index=False)
    clinical.to_csv(output_dir / "clinical.tsv", sep="\t", index=False)
    (output_dir / "pathways.json").write_text(json.dumps(pathway_payload, indent=2), encoding="utf-8")
    return output_dir
