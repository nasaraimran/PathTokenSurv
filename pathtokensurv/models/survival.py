from __future__ import annotations

import torch
from torch import nn


class CancerStratifiedDiscreteSurvivalHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_time_bins: int,
        num_cancers: int,
        dropout: float,
        use_cancer_deviation: bool = True,
    ) -> None:
        super().__init__()
        self.num_time_bins = int(num_time_bins)
        self.num_cancers = int(num_cancers)
        self.use_cancer_deviation = bool(use_cancer_deviation)
        self.patient_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_time_bins),
        )
        self.global_baseline = nn.Parameter(torch.zeros(num_time_bins))
        self.cancer_deviation = nn.Embedding(num_cancers, num_time_bins)
        nn.init.zeros_(self.cancer_deviation.weight)

    def forward(self, fused: torch.Tensor, cancers: torch.Tensor) -> torch.Tensor:
        logits = self.patient_head(fused) + self.global_baseline
        if self.use_cancer_deviation:
            logits = logits + self.cancer_deviation(cancers.long())
        return logits

    def baseline_regularization(self, smoothness: float) -> torch.Tensor:
        if not self.use_cancer_deviation:
            return self.global_baseline.new_zeros(())
        deviation = self.cancer_deviation.weight
        magnitude = deviation.pow(2).mean()
        if deviation.shape[1] > 1:
            adjacent = (deviation[:, 1:] - deviation[:, :-1]).pow(2).mean()
        else:
            adjacent = deviation.new_zeros(())
        return magnitude + float(smoothness) * adjacent

    @staticmethod
    def survival_from_logits(logits: torch.Tensor) -> torch.Tensor:
        hazards = torch.sigmoid(logits)
        return torch.cumprod(1.0 - hazards.clamp(max=1.0 - 1e-7), dim=1)
