# PathTokenSurv v1.2

## Fixed

- Fixed `ValueError: assignment destination is read-only` in `pathtokensurv/data/preprocessing.py` on pandas / NumPy configurations that return read-only views from `DataFrame.to_numpy()`.
- Fixed the related PyTorch warning about converting non-writable NumPy arrays with `torch.from_numpy()`.
- NumPy arrays that are modified in-place or passed to PyTorch are now explicitly copied or made contiguous and writable.

## Validation

- `python -m pytest -q`: 4 tests passed.
- `python scripts/run_synthetic.py --config configs/synthetic.json --clean`: completed successfully end-to-end with no read-only-array warning or exception.
