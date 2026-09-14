from __future__ import annotations

import torch
from torch import nn


class IntraModalEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_layers: int,
        ffn_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ffn_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.final_norm(self.encoder(tokens))
