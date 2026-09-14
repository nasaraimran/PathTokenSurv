from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence

import torch
from torch import nn


class BaseTokenizer(nn.Module, ABC):
    @property
    @abstractmethod
    def token_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class MolecularPathwayTokenizer(BaseTokenizer):
    """Convert one molecular feature vector into summary, pathway, and residual tokens.

    v1.5.6 keeps the exact pathway-attention calculation used in v1.5.5 but
    evaluates multiple pathways together in padded chunks. This removes hundreds
    of tiny Python-dispatched tensor kernels per modality and greatly improves
    CPU/GPU utilization while keeping the biological specification unchanged.

    The legacy per-pathway buffers are retained in the state dict so checkpoints
    produced by v1.5.5 remain load-compatible. The padded/chunked tensors are
    non-persistent implementation buffers and therefore do not alter checkpoints.
    """

    def __init__(
        self,
        num_features: int,
        pathway_indices: Sequence[Sequence[int]],
        residual_indices: Sequence[int],
        d_model: int,
        modality_id: int,
        dropout: float,
        pathway_chunk_size: int = 32,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.num_pathways = len(pathway_indices)
        self.d_model = int(d_model)
        self.modality_id = int(modality_id)
        self.pathway_chunk_size = max(1, int(pathway_chunk_size))

        self.feature_embeddings = nn.Embedding(self.num_features, d_model)
        self.value_projection = nn.Linear(1, d_model)
        self.feature_transform = nn.Linear(d_model, d_model)
        self.pathway_transform = nn.Linear(d_model, d_model, bias=False)
        self.attention_vector = nn.Linear(d_model, 1, bias=False)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pathway_embeddings = nn.Embedding(self.num_pathways, d_model)
        self.modality_embedding = nn.Parameter(torch.empty(d_model))
        self.summary_token = nn.Parameter(torch.empty(d_model))
        self.residual_identity = nn.Parameter(torch.empty(d_model))
        self.empty_pathway_token = nn.Parameter(torch.empty(d_model))
        self.empty_residual_token = nn.Parameter(torch.empty(d_model))
        self.layer_norm = nn.LayerNorm(d_model)

        # Keep v1.5.5 state-dict compatibility.
        self._pathway_buffer_names: List[str] = []
        normalized_indices: List[List[int]] = []
        for idx, indices in enumerate(pathway_indices):
            values = list(map(int, indices))
            normalized_indices.append(values)
            name = f"pathway_indices_{idx}"
            self.register_buffer(name, torch.as_tensor(values, dtype=torch.long))
            self._pathway_buffer_names.append(name)
        self.register_buffer("residual_indices", torch.as_tensor(list(residual_indices), dtype=torch.long))

        max_group = max([len(v) for v in normalized_indices] + [1])
        padded = torch.zeros((self.num_pathways, max_group), dtype=torch.long)
        valid = torch.zeros((self.num_pathways, max_group), dtype=torch.bool)
        for pathway_idx, indices in enumerate(normalized_indices):
            if indices:
                n = len(indices)
                padded[pathway_idx, :n] = torch.as_tensor(indices, dtype=torch.long)
                valid[pathway_idx, :n] = True
        # Implementation-only buffers: they can be regenerated from legacy buffers.
        self.register_buffer("_padded_pathway_indices", padded, persistent=False)
        self.register_buffer("_padded_pathway_valid", valid, persistent=False)
        self.reset_parameters()

    @property
    def token_count(self) -> int:
        return self.num_pathways + 2  # summary + P pathways + residual

    def reset_parameters(self) -> None:
        nn.init.normal_(self.feature_embeddings.weight, std=0.02)
        nn.init.normal_(self.pathway_embeddings.weight, std=0.02)
        nn.init.normal_(self.modality_embedding, std=0.02)
        nn.init.normal_(self.summary_token, std=0.02)
        nn.init.normal_(self.residual_identity, std=0.02)
        nn.init.normal_(self.empty_pathway_token, std=0.02)
        nn.init.normal_(self.empty_residual_token, std=0.02)

    def _pool_feature_group(
        self,
        x: torch.Tensor,
        feature_indices: torch.Tensor,
        identity_embedding: torch.Tensor,
        empty_token: torch.Tensor,
    ) -> torch.Tensor:
        """Legacy/single-group implementation used for the residual token."""
        batch_size = x.shape[0]
        if feature_indices.numel() == 0:
            return empty_token.unsqueeze(0).expand(batch_size, -1) + identity_embedding

        values = x.index_select(1, feature_indices)  # [B, F_group]
        feature_embed = self.feature_embeddings(feature_indices).unsqueeze(0)  # [1, F_group, D]
        value_embed = self.value_projection(values.unsqueeze(-1))
        hidden = feature_embed + value_embed
        identity = identity_embedding.view(1, 1, -1)
        scores = self.attention_vector(
            torch.tanh(self.feature_transform(hidden) + self.pathway_transform(identity))
        ).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * hidden, dim=1)
        return self.output_projection(pooled) + identity_embedding

    def _pool_pathway_chunk(
        self,
        x: torch.Tensor,
        start: int,
        stop: int,
    ) -> torch.Tensor:
        """Vectorized equivalent of repeatedly calling ``_pool_feature_group``."""
        indices = self._padded_pathway_indices[start:stop]  # [C, K]
        valid = self._padded_pathway_valid[start:stop]  # [C, K]
        chunk = int(stop - start)
        batch_size = x.shape[0]

        # Advanced indexing gives [B, C, K]. Padding positions are masked later.
        values = x[:, indices]
        feature_embed = self.feature_embeddings(indices).unsqueeze(0)  # [1, C, K, D]
        value_embed = self.value_projection(values.unsqueeze(-1))
        hidden = feature_embed + value_embed

        identities = (
            self.pathway_embeddings.weight[start:stop] + self.modality_embedding
        )  # [C, D]
        identity = identities.view(1, chunk, 1, self.d_model)
        scores = self.attention_vector(
            torch.tanh(self.feature_transform(hidden) + self.pathway_transform(identity))
        ).squeeze(-1)  # [B, C, K]

        # -inf is safe for every non-empty pathway. Empty pathways are replaced
        # explicitly after pooling, so use a finite floor there to avoid NaNs.
        nonempty = valid.any(dim=1)
        safe_valid = valid.clone()
        if (~nonempty).any():
            safe_valid[~nonempty, 0] = True
        floor = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~safe_valid.unsqueeze(0), floor)
        weights = torch.softmax(scores, dim=2)
        pooled = torch.sum(weights.unsqueeze(-1) * hidden, dim=2)  # [B, C, D]
        token = self.output_projection(pooled) + identities.unsqueeze(0)

        if (~nonempty).any():
            empty_value = self.empty_pathway_token.view(1, 1, -1) + identities.unsqueeze(0)
            token = torch.where(nonempty.view(1, chunk, 1), token, empty_value)
        return self.layer_norm(token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.num_features:
            raise ValueError(
                f"Expected molecular input [batch, {self.num_features}], got {tuple(x.shape)}."
            )
        batch_size = x.shape[0]
        pathway_chunks: List[torch.Tensor] = []
        for start in range(0, self.num_pathways, self.pathway_chunk_size):
            stop = min(start + self.pathway_chunk_size, self.num_pathways)
            pathway_chunks.append(self._pool_pathway_chunk(x, start, stop))
        pathway_tokens = torch.cat(pathway_chunks, dim=1) if pathway_chunks else x.new_empty((batch_size, 0, self.d_model))

        residual_identity = self.residual_identity + self.modality_embedding
        residual = self._pool_feature_group(
            x,
            self.residual_indices,
            residual_identity,
            self.empty_residual_token,
        )
        residual = self.layer_norm(residual)
        summary = self.layer_norm(
            self.summary_token.unsqueeze(0).expand(batch_size, -1) + self.modality_embedding
        )
        return torch.cat([summary.unsqueeze(1), pathway_tokens, residual.unsqueeze(1)], dim=1)


class ClinicalTokenizer(BaseTokenizer):
    """Tokenize continuous and categorical clinical variables."""

    def __init__(
        self,
        num_continuous: int,
        categorical_cardinalities: Dict[str, int],
        d_model: int,
        modality_id: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_continuous = int(num_continuous)
        self.categorical_names = list(categorical_cardinalities.keys())
        self.d_model = int(d_model)
        self.modality_id = int(modality_id)

        self.continuous_projections = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(self.num_continuous)]
        )
        self.categorical_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(int(cardinality), d_model)
                for name, cardinality in categorical_cardinalities.items()
            }
        )
        self.type_embeddings = nn.Embedding(
            self.num_continuous + len(self.categorical_names), d_model
        )
        self.modality_embedding = nn.Parameter(torch.empty(d_model))
        self.summary_token = nn.Parameter(torch.empty(d_model))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.modality_embedding, std=0.02)
        nn.init.normal_(self.summary_token, std=0.02)

    @property
    def token_count(self) -> int:
        return 1 + self.num_continuous + len(self.categorical_names)

    def forward(
        self,
        continuous: torch.Tensor,
        categorical: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if continuous.ndim != 2 or continuous.shape[1] != self.num_continuous:
            raise ValueError(
                f"Expected clinical continuous input [batch, {self.num_continuous}], "
                f"got {tuple(continuous.shape)}."
            )
        batch_size = continuous.shape[0]
        tokens: List[torch.Tensor] = []
        type_index = 0
        for idx, projection in enumerate(self.continuous_projections):
            token = projection(continuous[:, idx : idx + 1])
            token = token + self.type_embeddings.weight[type_index] + self.modality_embedding
            tokens.append(self.layer_norm(self.dropout(token)))
            type_index += 1
        for name in self.categorical_names:
            if name not in categorical:
                raise KeyError(f"Missing categorical clinical tensor '{name}'.")
            token = self.categorical_embeddings[name](categorical[name].long())
            token = token + self.type_embeddings.weight[type_index] + self.modality_embedding
            tokens.append(self.layer_norm(self.dropout(token)))
            type_index += 1
        summary = self.layer_norm(
            self.summary_token.unsqueeze(0).expand(batch_size, -1) + self.modality_embedding
        )
        return torch.stack([summary, *tokens], dim=1)
