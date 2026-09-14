# Synthetic validation report

## Environment used

- Python: container runtime
- PyTorch: 2.10.0+cpu
- NumPy: 2.3.5
- Pandas: 2.2.3
- scikit-learn: 1.8.0
- Device: CPU

## Automated tests

Command:

```bash
pytest -q
```

Result:

```text
4 passed
```

The tests covered time discretization, survival monotonicity, a perfect C-index case, leakage-controlled preprocessing, model output dimensions, and structured masking.

## End-to-end command

```bash
python scripts/run_synthetic.py --config configs/synthetic.json --clean
```

Validated stages:

- synthetic data generation;
- file loading and patient alignment;
- train-only preprocessing;
- pathway specification construction;
- model initialization;
- training and checkpoint reload;
- held-out prediction and metrics;
- artifact serialization;
- inference artifact reload; and
- one prediction row per patient.

The completed smoke run used 240 synthetic patients, six pathways, 24 features per molecular modality, four cancer types, and eight epochs. The best validation checkpoint occurred at epoch 5. Training took about 6.3 seconds in the local CPU runtime.

Held-out synthetic metrics included a C-index of approximately 0.562 and an IBS of approximately 0.259. These values only confirm that the software runs and produces finite outputs. They are not expected manuscript results.

## Additional entry points tested

- `scripts/evaluate.py` completed on the labeled synthetic cohort.
- `scripts/run_ensemble.py` completed with two one-epoch members.
- `scripts/run_outer_cv.py` completed with two one-epoch outer folds.
- `scripts/run_nested_cv.py` completed with two outer folds, two inner folds, and one candidate.
- `scripts/run_ablation.py` completed all five core variants with one epoch each.
- `scripts/evaluate_robustness.py` completed controlled missingness and all 15 modality subsets.
- `scripts/make_pathway_controls.py` produced shuffled-assignment and degree-matched graph controls.
- Editable installation completed with `pip install -e . --no-build-isolation`.

A PyTorch warning about nested-tensor optimization appears because the intra-modality transformer uses pre-normalization. It is an optimization notice rather than a correctness error.
