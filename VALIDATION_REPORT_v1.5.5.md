# PathTokenSurv v1.5.5 Validation Report

## Automated tests

Command:

```text
python -m pytest -q
```

Result:

```text
24 passed
```

The suite covers TCGA raw loading, leakage-controlled preprocessing, pathway construction, mRNA symbol/Entrez aliases, TarBase parsing, enrichment-based miRNA mapping, v1.5.4 score-aware top-K pathway selection, model forward propagation, survival metrics, and the new v1.5.5 IPCW calibration/per-cancer evaluation utilities.

## Single-outer-fold end-to-end smoke test

A deliberately small synthetic cohort was generated and trained through the new `scripts/run_outer_fold.py` path for two epochs. The workflow completed successfully and produced:

- exact split assignments and cancer/event summaries;
- a versioned experiment manifest with SHA-256 provenance;
- fold-specific preprocessing and pathway artifacts;
- best-model checkpoint and inference checkpoint;
- held-out test predictions and survival metrics;
- per-cancer metrics;
- IPCW calibration table and plot;
- training history and mean-survival plot.

The manifest ended with `status = PASS`.

**All metric values from this smoke test are synthetic software-validation outputs and must not be reported as TCGA study results.**

## Real-data prerequisite already satisfied

The immediately preceding real-data v1.5.4 CPU dry-run reported `PASS` on 10,986 eligible TCGA patients, with 8,788 outer-development patients, 2,198 outer-test patients, 1,542 selected mRNA features, 743 miRNA features, 2,671 CNV features, 256 retained pathways, 20 time bins, and finite survival probabilities within [0,1]. The v1.5.5 training script deliberately reuses the same outer-fold definition.
