from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from pathtokensurv.config import ExperimentConfig
from pathtokensurv.data.dataset import MultiModalSurvivalDataset
from pathtokensurv.data.preprocessing import ProcessedCohort
from pathtokensurv.data.time import TimeDiscretizer
from pathtokensurv.losses import discrete_survival_loss
from pathtokensurv.metrics import evaluate_survival_predictions
from pathtokensurv.models.encoders import IntraModalEncoder
from pathtokensurv.models.survival import CancerStratifiedDiscreteSurvivalHead
from pathtokensurv.models.tokenizers import ClinicalTokenizer
from pathtokensurv.scientific_evaluation import save_scientific_evaluation
from pathtokensurv.trainer import PredictionBundle
from pathtokensurv.utils.device import move_batch_to_device
from pathtokensurv.utils.io import save_json
from pathtokensurv.utils.seed import seed_everything


class CancerOnlySurvivalBaseline(nn.Module):
    """Discrete-time baseline whose only patient-specific input is cancer type."""

    def __init__(self, num_cancers: int, num_time_bins: int) -> None:
        super().__init__()
        self.global_baseline = nn.Parameter(torch.zeros(num_time_bins))
        self.cancer_deviation = nn.Embedding(num_cancers, num_time_bins)
        nn.init.zeros_(self.cancer_deviation.weight)

    def forward(self, batch: dict) -> dict:
        cancers = batch["cancer"].long()
        logits = self.global_baseline.unsqueeze(0) + self.cancer_deviation(cancers)
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1.0 - hazards.clamp(max=1.0 - 1e-7), dim=1)
        fused = logits.new_zeros((logits.shape[0], 1))
        return {"logits": logits, "survival": survival, "fused": fused}

    def baseline_regularization(self, smoothness: float) -> torch.Tensor:
        deviation = self.cancer_deviation.weight
        magnitude = deviation.pow(2).mean()
        adjacent = (
            (deviation[:, 1:] - deviation[:, :-1]).pow(2).mean()
            if deviation.shape[1] > 1
            else deviation.new_zeros(())
        )
        return magnitude + float(smoothness) * adjacent


class ClinicalOnlySurvivalBaseline(nn.Module):
    """Clinical covariates plus cancer-conditioned discrete-time survival head."""

    def __init__(
        self,
        config: ExperimentConfig,
        category_cardinalities: Dict[str, int],
        num_continuous: int,
        num_cancers: int,
    ) -> None:
        super().__init__()
        d_model = config.model.d_model
        self.tokenizer = ClinicalTokenizer(
            num_continuous=num_continuous,
            categorical_cardinalities=category_cardinalities,
            d_model=d_model,
            modality_id=0,
            dropout=config.model.dropout,
        )
        self.encoder = IntraModalEncoder(
            d_model=d_model,
            n_heads=config.model.n_heads,
            num_layers=config.model.intramodal_layers,
            ffn_multiplier=config.model.ffn_multiplier,
            dropout=config.model.dropout,
        )
        self.survival_head = CancerStratifiedDiscreteSurvivalHead(
            d_model=d_model,
            num_time_bins=config.model.num_time_bins,
            num_cancers=num_cancers,
            dropout=config.model.dropout,
            use_cancer_deviation=True,
        )

    def forward(self, batch: dict) -> dict:
        tokens = self.tokenizer(
            batch["clinical_continuous"], batch["clinical_categorical"]
        )
        encoded = self.encoder(tokens)
        fused = encoded[:, 0, :]
        logits = self.survival_head(fused, batch["cancer"])
        survival = self.survival_head.survival_from_logits(logits)
        return {"logits": logits, "survival": survival, "fused": fused}

    def baseline_regularization(self, smoothness: float) -> torch.Tensor:
        return self.survival_head.baseline_regularization(smoothness)


@dataclass
class BaselineRunResult:
    baseline: str
    best_epoch: int
    best_score: float
    metrics: Dict[str, float]
    output_dir: Path


def _loader(cohort: ProcessedCohort, config: ExperimentConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        MultiModalSurvivalDataset(cohort),
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.training.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> PredictionBundle:
    model.eval()
    patient_ids: List[str] = []
    times: List[np.ndarray] = []
    events: List[np.ndarray] = []
    cancers: List[np.ndarray] = []
    survival: List[np.ndarray] = []
    fused: List[np.ndarray] = []
    for batch in loader:
        patient_ids.extend(list(batch["patient_id"]))
        batch = move_batch_to_device(batch, device)
        output = model(batch)
        times.append(batch["time"].detach().cpu().numpy())
        events.append(batch["event"].detach().cpu().numpy())
        cancers.append(batch["cancer"].detach().cpu().numpy())
        survival.append(output["survival"].detach().cpu().numpy())
        fused.append(output["fused"].detach().cpu().numpy())
    return PredictionBundle(
        patient_ids=patient_ids,
        times=np.concatenate(times),
        events=np.concatenate(events),
        cancers=np.concatenate(cancers),
        survival=np.concatenate(survival),
        fused=np.concatenate(fused),
    )


def _save_predictions(
    bundle: PredictionBundle,
    horizons: np.ndarray,
    path: Path,
    labels: Sequence[str],
    evaluation_horizon_max: float | None = None,
) -> None:
    from pathtokensurv.metrics import risk_from_survival

    payload = {
        "patient_id": bundle.patient_ids,
        "time": bundle.times,
        "event": bundle.events,
        "cancer_id": bundle.cancers,
        "cancer_type": [
            labels[int(value)] if 0 <= int(value) < len(labels) else "UNKNOWN"
            for value in bundle.cancers
        ],
        "risk_score": risk_from_survival(bundle.survival, horizons),
    }
    if evaluation_horizon_max is not None:
        edges = np.asarray(horizons, dtype=float)
        supported = np.flatnonzero(edges <= float(evaluation_horizon_max) + 1e-6)
        if supported.size:
            payload["evaluation_risk_score"] = risk_from_survival(
                bundle.survival[:, supported], edges[supported]
            )
    frame = pd.DataFrame(payload)
    for idx, horizon in enumerate(horizons):
        frame[f"survival_{float(horizon):.6g}"] = bundle.survival[:, idx]
    frame.to_csv(path, index=False)


def train_baseline(
    baseline: str,
    config: ExperimentConfig,
    train: ProcessedCohort,
    val: ProcessedCohort,
    test: ProcessedCohort,
    discretizer: TimeDiscretizer,
    category_cardinalities: Dict[str, int],
    num_cancers: int,
    output_dir: str | Path,
    device: torch.device,
) -> BaselineRunResult:
    """Train a frozen-protocol comparator on the exact primary split/time bins."""
    baseline = str(baseline).lower()
    if baseline not in {"cancer_only", "clinical_only"}:
        raise ValueError("baseline must be 'cancer_only' or 'clinical_only'.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.training.seed)

    if baseline == "cancer_only":
        model: nn.Module = CancerOnlySurvivalBaseline(
            num_cancers=num_cancers,
            num_time_bins=discretizer.num_bins,
        )
    else:
        model = ClinicalOnlySurvivalBaseline(
            config=config,
            category_cardinalities=category_cardinalities,
            num_continuous=len(config.data.clinical_continuous),
            num_cancers=num_cancers,
        )
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.training.scheduler_t_max),
        eta_min=config.training.min_learning_rate,
    )
    train_loader = _loader(train, config, True)
    val_loader = _loader(val, config, False)
    test_loader = _loader(test, config, False)
    train_times = train.times.detach().cpu().numpy()
    train_events = train.events.detach().cpu().numpy()

    best_epoch = 0
    best_score = -np.inf
    wait = 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, config.training.max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_total = 0.0
        train_survival = 0.0
        count = 0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            survival_loss = discrete_survival_loss(
                output["logits"], batch["time"], batch["event"], discretizer
            )
            baseline_reg = model.baseline_regularization(config.training.baseline_smoothness)
            loss = survival_loss + config.training.lambda_baseline * baseline_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            optimizer.step()
            n = int(batch["time"].shape[0])
            train_total += float(loss.detach().cpu()) * n
            train_survival += float(survival_loss.detach().cpu()) * n
            count += n
        scheduler.step()

        val_start = time.perf_counter()
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        patient_ids: List[str] = []
        times_parts: List[np.ndarray] = []
        events_parts: List[np.ndarray] = []
        cancers_parts: List[np.ndarray] = []
        survival_parts: List[np.ndarray] = []
        fused_parts: List[np.ndarray] = []
        with torch.no_grad():
            for batch in val_loader:
                patient_ids.extend(list(batch["patient_id"]))
                batch = move_batch_to_device(batch, device)
                output = model(batch)
                loss = discrete_survival_loss(
                    output["logits"], batch["time"], batch["event"], discretizer
                )
                n = int(batch["time"].shape[0])
                val_loss_sum += float(loss.detach().cpu()) * n
                val_count += n
                times_parts.append(batch["time"].cpu().numpy())
                events_parts.append(batch["event"].cpu().numpy())
                cancers_parts.append(batch["cancer"].cpu().numpy())
                survival_parts.append(output["survival"].cpu().numpy())
                fused_parts.append(output["fused"].cpu().numpy())
        val_predictions = PredictionBundle(
            patient_ids=patient_ids,
            times=np.concatenate(times_parts),
            events=np.concatenate(events_parts),
            cancers=np.concatenate(cancers_parts),
            survival=np.concatenate(survival_parts),
            fused=np.concatenate(fused_parts),
        )
        val_metrics = evaluate_survival_predictions(
            train_times,
            train_events,
            val_predictions.times,
            val_predictions.events,
            val_predictions.survival,
            discretizer.right_edges,
            calibration_bins=config.evaluation.calibration_bins,
        )
        score = float(val_metrics["c_index"] - val_metrics["ibs"])
        val_seconds = time.perf_counter() - val_start
        epoch_seconds = time.perf_counter() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "train_total": train_total / max(count, 1),
                "train_survival": train_survival / max(count, 1),
                "val_survival_loss": val_loss_sum / max(val_count, 1),
                "val_c_index": val_metrics["c_index"],
                "val_ibs": val_metrics["ibs"],
                "selection_score": score,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "validation_seconds": val_seconds,
                "epoch_seconds": epoch_seconds,
            }
        )
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            wait = 0
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "score": score},
                output_dir / "best_model.pt",
            )
        else:
            wait += 1
        if wait >= config.training.patience:
            break

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    predictions = _predict(model, test_loader, device)
    metrics = evaluate_survival_predictions(
        train_times,
        train_events,
        predictions.times,
        predictions.events,
        predictions.survival,
        discretizer.right_edges,
        calibration_bins=config.evaluation.calibration_bins,
    )
    save_json(metrics, output_dir / "test_metrics.json")
    _save_predictions(
        predictions,
        discretizer.right_edges,
        output_dir / "test_predictions.csv",
        test.cancer_labels,
        evaluation_horizon_max=metrics.get("evaluation_horizon_max"),
    )
    scientific = save_scientific_evaluation(
        train,
        predictions,
        discretizer.right_edges,
        output_dir,
        config.evaluation.calibration_bins,
    )
    save_json({**metrics, **scientific}, output_dir / "test_metrics_extended.json")
    save_json(
        {
            "status": "PASS",
            "baseline": baseline,
            "best_epoch": int(best_epoch),
            "best_score": float(best_score),
            "training_seconds": float(time.perf_counter() - start),
            "training_seed": int(config.training.seed),
            "checkpoint_selection_rule": "validation_c_index_minus_validation_ibs",
            "same_primary_split": True,
            "same_time_discretizer": True,
            "same_optimizer_hyperparameters": True,
        },
        output_dir / "baseline_manifest.json",
    )
    return BaselineRunResult(
        baseline=baseline,
        best_epoch=best_epoch,
        best_score=best_score,
        metrics=metrics,
        output_dir=output_dir,
    )
