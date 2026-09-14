# PathTokenSurv v1.5.6 Validation Report

Validation date: 2026-08-10

## Automated tests

Command:

```text
PYTHONPATH=. pytest -q
```

Result:

```text
26 passed in 2.81s
```

The v1.5.6 tests include a direct numerical comparison between the legacy v1.5.5 per-pathway tokenizer loop and the new chunked tokenizer with dropout disabled. Outputs matched within numerical tolerance. Tests also verify that the new implementation-only padded buffers are non-persistent so legacy checkpoint state dictionaries are not polluted by new keys.

## End-to-end synthetic validation

Command:

```text
PYTHONPATH=. python scripts/run_synthetic.py --config configs/synthetic.json --clean
```

Result:

```text
End-to-end validation passed.
```

The smoke workflow covered preprocessing, pathway specification, training, per-epoch persistence, checkpointing, validation, held-out evaluation, reload, and inference. Synthetic performance values are software-validation outputs only and must not be reported as TCGA results.

## Performance microbenchmark

A CPU-only tokenizer microbenchmark with batch size 8, 256 pathways, 2,671 features, 128-dimensional tokens, and 32 features per simulated pathway showed approximately 1.65x lower tokenizer forward time for the chunked implementation than the v1.5.5 Python pathway loop in the validation environment. This is a component benchmark, not a prediction of total TCGA training speed.

## CUDA validation limitation

The release environment used CPU PyTorch 2.10.0, so CUDA execution could not be validated locally. `scripts/check_cuda.py --require` is provided to validate the user's Windows/NVIDIA environment before the TCGA run.
