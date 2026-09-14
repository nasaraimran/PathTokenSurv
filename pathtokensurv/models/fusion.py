from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from pathtokensurv.constants import MODALITIES, MODALITY_TO_INDEX


class PathwayBiasedSelfAttention(nn.Module):
    """Multi-head self-attention with a learned and biologically informed bias."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_modality_ids: int,
        modality_ids: torch.Tensor,
        pathway_ids: torch.Tensor,
        pathway_adjacency: torch.Tensor,
        dropout: float,
        use_pathway_bias: bool,
        same_bias_init: float,
        graph_bias_init: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = float(dropout)
        self.use_pathway_bias = bool(use_pathway_bias)

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.modality_pair_bias = nn.Parameter(
            torch.zeros(n_heads, num_modality_ids, num_modality_ids)
        )
        self.same_pathway_scale = nn.Parameter(
            torch.full((n_heads,), float(same_bias_init))
        )
        self.graph_pathway_scale = nn.Parameter(
            torch.full((n_heads,), float(graph_bias_init))
        )
        self.register_buffer("modality_ids", modality_ids.long())
        self.register_buffer("pathway_ids", pathway_ids.long())
        self.register_buffer("pathway_adjacency", pathway_adjacency.float())

        # v1.5.6: cache the static structural masks once. They are marked
        # non-persistent so v1.5.5 checkpoints remain strict-load compatible.
        valid = pathway_ids.long() >= 0
        same = (
            (pathway_ids.long()[:, None] == pathway_ids.long()[None, :])
            & valid[:, None]
            & valid[None, :]
        ).float()
        graph = torch.zeros((pathway_ids.numel(), pathway_ids.numel()), dtype=torch.float32)
        if valid.any():
            valid_positions = torch.nonzero(valid, as_tuple=False).squeeze(1)
            pids = pathway_ids.long()[valid_positions]
            subgraph = pathway_adjacency.float()[pids[:, None], pids[None, :]]
            graph[valid_positions[:, None], valid_positions[None, :]] = subgraph
        self.register_buffer("_same_pathway_mask", same, persistent=False)
        self.register_buffer("_graph_pathway_mask", graph, persistent=False)

    def _attention_bias(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        modality_ids = self.modality_ids
        pair_bias = self.modality_pair_bias[:, modality_ids[:, None], modality_ids[None, :]]
        if not self.use_pathway_bias:
            return pair_bias.unsqueeze(0).to(dtype=dtype, device=device)

        same_scale = F.softplus(self.same_pathway_scale).view(self.n_heads, 1, 1)
        graph_scale = F.softplus(self.graph_pathway_scale).view(self.n_heads, 1, 1)
        same = self._same_pathway_mask.to(dtype=dtype, device=device)
        graph = self._graph_pathway_mask.to(dtype=dtype, device=device)
        bias = pair_bias + same_scale * same.unsqueeze(0) + graph_scale * graph.unsqueeze(0)
        return bias.unsqueeze(0).to(dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        qkv = self.qkv(x).reshape(batch_size, sequence_length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        bias = self._attention_bias(q.dtype, q.device)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, sequence_length, self.d_model)
        return self.out_projection(attended)


class PathwayBiasedTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_multiplier: int,
        dropout: float,
        num_modality_ids: int,
        modality_ids: torch.Tensor,
        pathway_ids: torch.Tensor,
        pathway_adjacency: torch.Tensor,
        use_pathway_bias: bool,
        same_bias_init: float,
        graph_bias_init: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = PathwayBiasedSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            num_modality_ids=num_modality_ids,
            modality_ids=modality_ids,
            pathway_ids=pathway_ids,
            pathway_adjacency=pathway_adjacency,
            dropout=dropout,
            use_pathway_bias=use_pathway_bias,
            same_bias_init=same_bias_init,
            graph_bias_init=graph_bias_init,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_multiplier, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))
        return x


class PathwayBiasedFusionTransformer(nn.Module):
    """Fuse all observed or reconstructed token sequences."""

    def __init__(
        self,
        token_counts: Dict[str, int],
        num_pathways: int,
        pathway_adjacency: torch.Tensor,
        d_model: int,
        n_heads: int,
        num_layers: int,
        ffn_multiplier: int,
        dropout: float,
        use_pathway_bias: bool,
        same_bias_init: float,
        graph_bias_init: float,
    ) -> None:
        super().__init__()
        self.token_counts = dict(token_counts)
        self.fusion_token = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.fusion_token, std=0.02)

        modality_ids: List[int] = [0]  # fusion token has its own ID
        pathway_ids: List[int] = [-1]
        for modality in MODALITIES:
            count = token_counts[modality]
            modality_ids.extend([MODALITY_TO_INDEX[modality] + 1] * count)
            if modality == "clinical":
                pathway_ids.extend([-1] * count)
            else:
                # [summary, P pathway tokens, residual]
                if count != num_pathways + 2:
                    raise ValueError(
                        f"Molecular modality '{modality}' has {count} tokens, expected {num_pathways + 2}."
                    )
                pathway_ids.extend([-1, *range(num_pathways), -1])

        modality_ids_t = torch.tensor(modality_ids, dtype=torch.long)
        pathway_ids_t = torch.tensor(pathway_ids, dtype=torch.long)
        self.blocks = nn.ModuleList(
            [
                PathwayBiasedTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ffn_multiplier=ffn_multiplier,
                    dropout=dropout,
                    num_modality_ids=len(MODALITIES) + 1,
                    modality_ids=modality_ids_t,
                    pathway_ids=pathway_ids_t,
                    pathway_adjacency=pathway_adjacency,
                    use_pathway_bias=use_pathway_bias,
                    same_bias_init=same_bias_init,
                    graph_bias_init=graph_bias_init,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    @property
    def sequence_length(self) -> int:
        return 1 + sum(self.token_counts.values())

    def forward(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = next(iter(tokens.values())).shape[0]
        fusion = self.fusion_token.view(1, 1, -1).expand(batch_size, -1, -1)
        sequence = torch.cat([fusion, *[tokens[m] for m in MODALITIES]], dim=1)
        for block in self.blocks:
            sequence = block(sequence)
        sequence = self.final_norm(sequence)
        return sequence[:, 0]
