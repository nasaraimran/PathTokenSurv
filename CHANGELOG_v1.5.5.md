# PathTokenSurv v1.5.5

## Scientific single-outer-fold training release

- Added `scripts/run_outer_fold.py` for one exact outer fold using the same outer-fold definition as TCGA QC and the CPU dry-run.
- Added explicit 15% early-stopping validation split inside the outer-development partition; the outer test fold remains untouched.
- Added `split_assignments.csv`, `split_summary.json`, and `split_summary_by_cancer_event.csv` for exact patient-level split provenance.
- Added `experiment_manifest.json` with package version, seeds, split sizes, and SHA-256 hashes of the source config and pathway mapping.
- Added held-out `per_cancer_metrics.csv` using cancer-specific training censoring references.
- Added IPCW calibration tables and a representative calibration plot (`calibration_ipcw.csv`, `calibration_ipcw.png`).
- Test predictions now include both numeric cancer ID and cancer-type label.
- Added `FROZEN_PRIMARY_PROTOCOL_v1.5.5.md` to lock the primary data/pathway/model protocol before manuscript-grade training.
- Corrected the dry-run report version tag and default v1.5.5 output directory.
- Package version advanced to `0.1.5.5`.
