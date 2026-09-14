from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import torch
from torch import nn

from pathtokensurv.config import ModelConfig
from pathtokensurv.constants import MODALITIES, MOLECULAR_MODALITIES, MODALITY_TO_INDEX
from pathtokensurv.data.pathways import PathwaySpec
from pathtokensurv.models.encoders import IntraModalEncoder
from pathtokensurv.models.fusion import PathwayBiasedFusionTransformer
from pathtokensurv.models.reconstruction import CrossModalReconstructor
from pathtokensurv.models.survival import CancerStratifiedDiscreteSurvivalHead
from pathtokensurv.models.tokenizers import ClinicalTokenizer, MolecularPathwayTokenizer


@dataclass
class ViewOutput:
    fused: torch.Tensor
    reconstructed: Dict[str, torch.Tensor]
    fused_tokens: Dict[str, torch.Tensor]


class ModalityMaskSampler:
    def __init__(
        self,
        drop_probability: float,
        always_keep_clinical: bool = False,
    ) -> None:
        self.drop_probability = float(drop_probability)
        self.always_keep_clinical = bool(always_keep_clinical)

    def __call__(self, natural_mask: torch.Tensor) -> torch.Tensor:
        if natural_mask.dtype != torch.bool:
            natural_mask = natural_mask.bool()
        keep = torch.rand_like(natural_mask.float()) > self.drop_probability
        effective = natural_mask & keep
        if self.always_keep_clinical:
            effective[:, MODALITY_TO_INDEX["clinical"]] = natural_mask[:, MODALITY_TO_INDEX["clinical"]]

        # Guarantee one input modality for each patient.
        empty_rows = ~effective.any(dim=1)
        if empty_rows.any():
            rows = torch.nonzero(empty_rows, as_tuple=False).squeeze(1)
            for row in rows.tolist():
                available = torch.nonzero(natural_mask[row], as_tuple=False).squeeze(1)
                selected = available[torch.randint(0, len(available), (1,), device=natural_mask.device)]
                effective[row, selected] = True
        return effective


class PathTokenSurv(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        pathway_spec: PathwaySpec,
        category_cardinalities: Mapping[str, int],
        num_continuous_clinical: int,
        num_cancers: int,
    ) -> None:
        super().__init__()
        model_config.validate()
        self.config = model_config
        self.pathway_spec = pathway_spec

        d_model = model_config.d_model
        self.tokenizers = nn.ModuleDict()
        self.tokenizers["clinical"] = ClinicalTokenizer(
            num_continuous=num_continuous_clinical,
            categorical_cardinalities=dict(category_cardinalities),
            d_model=d_model,
            modality_id=MODALITY_TO_INDEX["clinical"],
            dropout=model_config.dropout,
        )
        for modality in MOLECULAR_MODALITIES:
            self.tokenizers[modality] = MolecularPathwayTokenizer(
                num_features=len(pathway_spec.feature_names[modality]),
                pathway_indices=pathway_spec.feature_indices[modality],
                residual_indices=pathway_spec.residual_indices[modality],
                d_model=d_model,
                modality_id=MODALITY_TO_INDEX[modality],
                dropout=model_config.dropout,
                pathway_chunk_size=model_config.tokenizer_pathway_chunk_size,
            )

        self.encoders = nn.ModuleDict(
            {
                modality: IntraModalEncoder(
                    d_model=d_model,
                    n_heads=model_config.n_heads,
                    num_layers=model_config.intramodal_layers,
                    ffn_multiplier=model_config.ffn_multiplier,
                    dropout=model_config.dropout,
                )
                for modality in MODALITIES
            }
        )
        self.token_counts = {
            modality: self.tokenizers[modality].token_count
            for modality in MODALITIES
        }
        self.reconstructor = CrossModalReconstructor(
            token_counts=self.token_counts,
            d_model=d_model,
            n_heads=model_config.n_heads,
            ffn_multiplier=model_config.ffn_multiplier,
            dropout=model_config.dropout,
            num_layers=model_config.reconstruction_layers,
        )
        self.missing_tokens = nn.ParameterDict({
            modality: nn.Parameter(torch.empty(count, d_model))
            for modality, count in self.token_counts.items()
        })
        for parameter in self.missing_tokens.values():
            nn.init.normal_(parameter, std=0.02)
        self.state_embedding = nn.Embedding(2, d_model)  # 0 observed, 1 reconstructed
        self.fusion = PathwayBiasedFusionTransformer(
            token_counts=self.token_counts,
            num_pathways=pathway_spec.num_pathways,
            pathway_adjacency=torch.as_tensor(pathway_spec.adjacency, dtype=torch.float32),
            d_model=d_model,
            n_heads=model_config.n_heads,
            num_layers=model_config.fusion_layers,
            ffn_multiplier=model_config.ffn_multiplier,
            dropout=model_config.dropout,
            use_pathway_bias=model_config.use_pathway_bias,
            same_bias_init=model_config.pathway_same_bias_init,
            graph_bias_init=model_config.pathway_graph_bias_init,
        )
        self.survival_head = CancerStratifiedDiscreteSurvivalHead(
            d_model=d_model,
            num_time_bins=model_config.num_time_bins,
            num_cancers=num_cancers,
            dropout=model_config.dropout,
            use_cancer_deviation=model_config.use_cancer_deviation,
        )

    def encode_modalities(self, batch: dict) -> Dict[str, torch.Tensor]:
        tokens: Dict[str, torch.Tensor] = {}
        tokens["clinical"] = self.tokenizers["clinical"](
            batch["clinical_continuous"],
            batch["clinical_categorical"],
        )
        for modality in MOLECULAR_MODALITIES:
            tokens[modality] = self.tokenizers[modality](batch["molecular"][modality])
        return {
            modality: self.encoders[modality](tokens[modality])
            for modality in MODALITIES
        }

    def build_view(
        self,
        encoded: Dict[str, torch.Tensor],
        effective_mask: torch.Tensor,
    ) -> ViewOutput:
        if self.config.use_reconstruction:
            reconstructed = self.reconstructor(encoded, effective_mask)
        else:
            batch_size = effective_mask.shape[0]
            reconstructed = {
                modality: self.missing_tokens[modality].unsqueeze(0).expand(batch_size, -1, -1)
                for modality in MODALITIES
            }
        fused_tokens: Dict[str, torch.Tensor] = {}
        for modality in MODALITIES:
            modality_mask = effective_mask[:, MODALITY_TO_INDEX[modality]].view(-1, 1, 1)
            selected = torch.where(modality_mask, encoded[modality], reconstructed[modality])
            state_ids = (~effective_mask[:, MODALITY_TO_INDEX[modality]]).long()
            selected = selected + self.state_embedding(state_ids).unsqueeze(1)
            fused_tokens[modality] = selected
        fused = self.fusion(fused_tokens)
        return ViewOutput(fused=fused, reconstructed=reconstructed, fused_tokens=fused_tokens)

    def forward(
        self,
        batch: dict,
        effective_mask: torch.Tensor | None = None,
    ) -> dict:
        natural_mask = batch["availability"].bool()
        if effective_mask is None:
            effective_mask = natural_mask
        encoded = self.encode_modalities(batch)
        view = self.build_view(encoded, effective_mask)
        logits = self.survival_head(view.fused, batch["cancer"])
        survival = self.survival_head.survival_from_logits(logits)
        return {
            "encoded": encoded,
            "effective_mask": effective_mask,
            "fused": view.fused,
            "reconstructed": view.reconstructed,
            "fused_tokens": view.fused_tokens,
            "logits": logits,
            "survival": survival,
        }

    def forward_training(
        self,
        batch: dict,
        mask_sampler: ModalityMaskSampler,
    ) -> dict:
        natural_mask = batch["availability"].bool()
        encoded = self.encode_modalities(batch)
        subset_mask = mask_sampler(natural_mask)
        subset_view = self.build_view(encoded, subset_mask)
        logits = self.survival_head(subset_view.fused, batch["cancer"])
        survival = self.survival_head.survival_from_logits(logits)

        # The natural-view branch supplies a stochastic stop-gradient target;
        # it remains active when subset_mask equals natural_mask.
        if self.config.use_subset_consistency:
            with torch.no_grad():
                full_view = self.build_view(encoded, natural_mask)
            full_fused = full_view.fused
        else:
            full_fused = subset_view.fused.detach()

        return {
            "encoded": encoded,
            "natural_mask": natural_mask,
            "effective_mask": subset_mask,
            "fused": subset_view.fused,
            "full_fused": full_fused,
            "reconstructed": subset_view.reconstructed,
            "logits": logits,
            "survival": survival,
        }
