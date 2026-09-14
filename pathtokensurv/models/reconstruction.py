from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from pathtokensurv.constants import MODALITIES, MODALITY_TO_INDEX


class ReconstructionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_multiplier, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        source: torch.Tensor,
        source_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.cross_attention(
            query=query,
            key=source,
            value=source,
            key_padding_mask=source_padding_mask,
            need_weights=False,
        )
        hidden = self.norm1(query + attended)
        return self.norm2(hidden + self.ffn(hidden))


class ReconstructionDecoder(nn.Module):
    def __init__(
        self,
        target_tokens: int,
        d_model: int,
        n_heads: int,
        ffn_multiplier: int,
        dropout: float,
        num_modalities: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.query_tokens = nn.Parameter(torch.empty(target_tokens, d_model))
        self.availability_projection = nn.Linear(num_modalities, d_model)
        self.blocks = nn.ModuleList(
            [
                ReconstructionBlock(d_model, n_heads, ffn_multiplier, dropout)
                for _ in range(num_layers)
            ]
        )
        nn.init.normal_(self.query_tokens, std=0.02)

    def forward(
        self,
        source: torch.Tensor,
        source_padding_mask: torch.Tensor,
        availability: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = source.shape[0]
        availability_embedding = self.availability_projection(availability.float()).unsqueeze(1)
        query = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1) + availability_embedding
        for block in self.blocks:
            query = block(query, source, source_padding_mask)
        return query


class CrossModalReconstructor(nn.Module):
    def __init__(
        self,
        token_counts: Dict[str, int],
        d_model: int,
        n_heads: int,
        ffn_multiplier: int,
        dropout: float,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.token_counts = dict(token_counts)
        self.decoders = nn.ModuleDict(
            {
                modality: ReconstructionDecoder(
                    target_tokens=count,
                    d_model=d_model,
                    n_heads=n_heads,
                    ffn_multiplier=ffn_multiplier,
                    dropout=dropout,
                    num_modalities=len(MODALITIES),
                    num_layers=num_layers,
                )
                for modality, count in token_counts.items()
            }
        )

    def _source_sequence(
        self,
        encoded: Dict[str, torch.Tensor],
        effective_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = torch.cat([encoded[m] for m in MODALITIES], dim=1)
        padding_parts: List[torch.Tensor] = []
        for modality in MODALITIES:
            available = effective_mask[:, MODALITY_TO_INDEX[modality]]
            padding_parts.append(
                (~available).unsqueeze(1).expand(-1, self.token_counts[modality])
            )
        padding_mask = torch.cat(padding_parts, dim=1)
        if padding_mask.all(dim=1).any():
            raise ValueError("At least one modality must remain available for every patient.")
        return source, padding_mask

    def forward(
        self,
        encoded: Dict[str, torch.Tensor],
        effective_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        source, padding_mask = self._source_sequence(encoded, effective_mask)
        return {
            modality: decoder(source, padding_mask, effective_mask)
            for modality, decoder in self.decoders.items()
        }
