# Code design and manuscript alignment

## Methods-to-code map

| Route C method | Main implementation |
|---|---|
| Study records and modality masks | `data/raw.py`, `data/preprocessing.py` |
| Leakage-controlled preprocessing | `FoldPreprocessor` |
| Pathway vocabulary and feature mapping | `data/pathways.py` |
| Molecular pathway tokens | `MolecularPathwayTokenizer` |
| Clinical variable tokens | `ClinicalTokenizer` |
| Intra-modality transformers | `IntraModalEncoder` |
| Structured modality masking | `ModalityMaskSampler` |
| Cross-modal token reconstruction | `CrossModalReconstructor` |
| Observed/reconstructed state embeddings | `PathTokenSurv.build_view` |
| Pathway-biased multimodal attention | `PathwayBiasedSelfAttention` |
| Fusion-token representation | `PathwayBiasedFusionTransformer` |
| Subset consistency | `subset_consistency_loss` |
| Cancer-stratified discrete hazards | `CancerStratifiedDiscreteSurvivalHead` |
| Total objective | `total_training_loss` |
| Early stopping and checkpointing | `Trainer` |
| Held-out testing | `experiment.run_single_split` |
| External validation | `evaluation.evaluate_artifact` |
| Inference | `inference.run_inference` |
| Deep ensemble | `scripts/run_ensemble.py` |

## Main design choices

The implementation uses composition for the overall model. This is safer than forcing unrelated modules into one inheritance hierarchy. Tokenizers share `BaseTokenizer`, while encoders, reconstruction, fusion, and survival modules remain independent `nn.Module` components.

The pathway tokenizer uses a loop over pathways. This avoids a dense four-dimensional tensor that would consume excessive memory for real TCGA feature counts. The approach is clear and correct, although a custom segment-softmax kernel could later improve speed.

The fusion transformer uses `torch.nn.functional.scaled_dot_product_attention`. The biological bias is broadcast across the batch and added before softmax. The bias includes learned modality-pair effects, matching-pathway effects, and curated pathway-graph relations.

The discrete-time loss excludes the censoring interval when censoring occurs before its right boundary. This follows the observation mask stated in the revised Methods.

## Deliberate limitations of version 0.1

- The package includes leakage-controlled outer cross-validation, but it does not automatically perform a large nested hyperparameter search.
- The internal calibration diagnostic is simple. Final publication values should be checked with a validated survival library.
- Pathway database acquisition and identifier harmonization remain study-specific preprocessing tasks.
- Degree-matched pathway randomization is not generated automatically in this first package; the model accepts any alternative adjacency matrix in `pathways.json`.
- The inference helper loads one model. Deep-ensemble uncertainty is produced by `run_ensemble.py`.
