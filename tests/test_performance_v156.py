from __future__ import annotations

import torch

from pathtokensurv.models.tokenizers import MolecularPathwayTokenizer


def _legacy_pathway_forward(tokenizer: MolecularPathwayTokenizer, x: torch.Tensor) -> torch.Tensor:
    batch_size = x.shape[0]
    pathway_tokens = []
    for pathway_idx, buffer_name in enumerate(tokenizer._pathway_buffer_names):
        indices = getattr(tokenizer, buffer_name)
        pathway_identity = tokenizer.pathway_embeddings.weight[pathway_idx] + tokenizer.modality_embedding
        token = tokenizer._pool_feature_group(
            x,
            indices,
            pathway_identity,
            tokenizer.empty_pathway_token,
        )
        pathway_tokens.append(tokenizer.layer_norm(token))
    residual_identity = tokenizer.residual_identity + tokenizer.modality_embedding
    residual = tokenizer.layer_norm(
        tokenizer._pool_feature_group(
            x,
            tokenizer.residual_indices,
            residual_identity,
            tokenizer.empty_residual_token,
        )
    )
    summary = tokenizer.layer_norm(
        tokenizer.summary_token.unsqueeze(0).expand(batch_size, -1) + tokenizer.modality_embedding
    )
    return torch.stack([summary, *pathway_tokens, residual], dim=1)


def test_chunked_tokenizer_matches_v155_loop() -> None:
    torch.manual_seed(7)
    pathway_indices = [[], [0], [1, 2, 3], [0, 2, 4, 5], [5, 6]]
    tokenizer = MolecularPathwayTokenizer(
        num_features=9,
        pathway_indices=pathway_indices,
        residual_indices=[7, 8],
        d_model=12,
        modality_id=1,
        dropout=0.0,
        pathway_chunk_size=2,
    ).eval()
    x = torch.randn(4, 9)
    with torch.no_grad():
        legacy = _legacy_pathway_forward(tokenizer, x)
        chunked = tokenizer(x)
    assert legacy.shape == chunked.shape
    assert torch.allclose(legacy, chunked, atol=2e-6, rtol=2e-6)


def test_v155_state_dict_loads_without_new_buffer_keys() -> None:
    tokenizer = MolecularPathwayTokenizer(
        num_features=6,
        pathway_indices=[[0, 1], [2, 3]],
        residual_indices=[4, 5],
        d_model=8,
        modality_id=1,
        dropout=0.0,
        pathway_chunk_size=2,
    )
    keys = set(tokenizer.state_dict().keys())
    assert "_padded_pathway_indices" not in keys
    assert "_padded_pathway_valid" not in keys
    assert "pathway_indices_0" in keys
    assert "pathway_indices_1" in keys
