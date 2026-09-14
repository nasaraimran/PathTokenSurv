from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from pathtokensurv.constants import MODALITIES, MODALITY_TO_INDEX
from pathtokensurv.data.time import TimeDiscretizer


def discrete_survival_loss(
    logits: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    discretizer: TimeDiscretizer,
) -> torch.Tensor:
    labels, observed = discretizer.make_targets(times, events)
    if logits.shape != labels.shape:
        raise ValueError(f"Logit shape {tuple(logits.shape)} does not match labels {tuple(labels.shape)}.")
    elementwise = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (elementwise * observed).sum() / observed.sum().clamp_min(1.0)


def reconstruction_loss(
    reconstructed: Dict[str, torch.Tensor],
    encoded: Dict[str, torch.Tensor],
    natural_mask: torch.Tensor,
    effective_mask: torch.Tensor,
) -> torch.Tensor:
    total = next(iter(reconstructed.values())).new_zeros(())
    count = next(iter(reconstructed.values())).new_zeros(())
    for modality in MODALITIES:
        target_rows = natural_mask[:, MODALITY_TO_INDEX[modality]] & ~effective_mask[:, MODALITY_TO_INDEX[modality]]
        if target_rows.any():
            per_patient = (reconstructed[modality] - encoded[modality].detach()).pow(2).mean(dim=(1, 2))
            total = total + per_patient[target_rows].sum()
            count = count + target_rows.sum()
    return total / count.clamp_min(1.0)


def subset_consistency_loss(subset_fused: torch.Tensor, full_fused: torch.Tensor) -> torch.Tensor:
    target = full_fused.detach()
    similarity = F.cosine_similarity(subset_fused, target, dim=1, eps=1e-8)
    return (1.0 - similarity).mean()


def total_training_loss(
    model,
    output: dict,
    batch: dict,
    discretizer: TimeDiscretizer,
    lambda_reconstruction: float,
    lambda_consistency: float,
    lambda_baseline: float,
    baseline_smoothness: float,
) -> dict:
    surv = discrete_survival_loss(output["logits"], batch["time"], batch["event"], discretizer)
    rec = reconstruction_loss(
        output["reconstructed"],
        output["encoded"],
        output["natural_mask"],
        output["effective_mask"],
    )
    cons = subset_consistency_loss(output["fused"], output["full_fused"])
    base = model.survival_head.baseline_regularization(baseline_smoothness)
    total = surv + lambda_reconstruction * rec + lambda_consistency * cons + lambda_baseline * base
    return {
        "total": total,
        "survival": surv,
        "reconstruction": rec,
        "consistency": cons,
        "baseline": base,
    }
