# PathTokenSurv v1.4 validation report

## Automated tests

```text
10 passed
```

The tests cover the original survival-time, preprocessing, TCGA-native loading, pathway construction, and model-forward checks plus v1.4 availability-pattern, cancer-missingness, variance-QC, annotation-matching, and pathway-size-filter checks.

## End-to-end synthetic training

`python scripts/run_synthetic.py --config configs/synthetic.json --clean` completed successfully after the v1.4 changes. Training, checkpointing, held-out evaluation, artifact saving, reload, and inference all completed with finite outputs.

## v1.4 QC smoke test

The detailed QC script was run on an 80-patient synthetic multimodal cohort:

```text
python scripts/run_tcga_qc.py --config configs/synthetic.json --output-dir outputs/qc_smoke --outer-folds 3 --fold 1
```

It generated exact availability-pattern counts, cancer-specific missingness, variance diagnostics, annotation validation, pathway coverage, figures, and the supplementary Markdown report.

## v1.4 dry-run smoke test

```text
python scripts/dry_run_preprocessing.py --config configs/synthetic.json --output-dir outputs/dry_smoke --outer-folds 3 --fold 1 --batch-size 2
```

Result:

```text
status: PASS
selected features: 24 mRNA, 24 miRNA, 24 CNV
retained pathways: 6
survival output: finite and within [0, 1]
```

The synthetic numbers are software-validation outputs only. They must not be reported as scientific results.
