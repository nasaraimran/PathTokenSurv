# PathTokenSurv v1.5.6 — Performance and Recovery Patch

## Scope

v1.5.6 is an engineering/performance revision. It does **not** change the frozen v1.5.5 scientific protocol: TCGA cohort definition, feature thresholds, pathway resources, pathway selection, modality caps, model dimensions, survival endpoint, loss weights, or outer-fold definitions remain unchanged.

## Changes

- Vectorized molecular pathway tokenization in configurable pathway chunks (`model.tokenizer_pathway_chunk_size`, default 32). The pathway-level attention equation is unchanged.
- Cached static same-pathway and Reactome graph masks used by pathway-biased fusion attention.
- Combined validation loss and validation prediction generation into one forward pass per validation batch.
- Added per-epoch timing fields: `train_seconds`, `validation_seconds`, and `epoch_seconds`.
- `training_history.csv` and `training_summary.json` are now updated after every completed epoch.
- Added `last_model.pt` after every completed epoch, including optimizer state, scheduler state, early-stopping state, and best-score metadata.
- Added `--resume-from` support to `scripts/run_outer_fold.py`.
- Added `--require-cuda` so a manuscript run can abort immediately if the active PyTorch environment is CPU-only.
- Startup now reports device, GPU name, compute capability, GPU memory, and PyTorch CUDA runtime.
- `experiment_manifest.json` now records Python, PyTorch, CUDA runtime, resolved device, and GPU metadata for reproducibility.
- Added `scripts/check_cuda.py` for a real CUDA-kernel smoke test.
- Package version updated to 0.1.5.6.

## Reproducibility note

The chunked tokenizer is mathematically equivalent to the v1.5.5 pathway loop at evaluation time. Training uses the same stochastic dropout distribution, but vectorized execution changes the order in which random masks are generated. For the primary manuscript experiment, start a fresh v1.5.6 fold from epoch 1 rather than combining v1.5.5 and v1.5.6 epochs.
