# PathTokenSurv

A PyTorch research implementation of:

**Missingness-aware pathway-token cross-modal learning for pan-cancer survival prediction.**

The package covers fold-specific preprocessing, pathway-token construction, modality encoders, structured modality masking, cross-modal token reconstruction, pathway-biased fusion, cancer-stratified discrete-time survival prediction, training, evaluation, outer cross-validation, deep ensembles, and inference.

The tracked repository contains source code, tests, synthetic fixtures, study
protocols, and aggregate figure inputs. Patient-level TCGA data, third-party
annotation tables, checkpoints, and experiment directories are intentionally
excluded; see `data/README.md`, `annotations/README.md`, and `outputs/README.md`.

## 1. What is implemented

- Training-only variance filtering, median filling, and min-max scaling.
- Unknown-category handling for clinical variables and cancer labels.
- Shared pathway vocabulary for mRNA, miRNA, and CNV.
- Memory-conscious pathway tokenization with patient-specific within-pathway attention.
- Separate transformer encoder for each modality.
- Structured modality masking that never removes every available input.
- Supervised token reconstruction only for observed modalities hidden during training.
- Learned missing-token ablation when reconstruction is disabled.
- Additive same-pathway, pathway-graph, and modality-pair attention biases.
- Subset consistency loss with a stop-gradient full-view target.
- Discrete-time survival likelihood with cancer-specific baseline deviations.
- Harrell C-index, IPCW Brier score, integrated Brier score, time-dependent AUC, and a calibration diagnostic.
- Training, held-out testing, external evaluation, inference, outer cross-validation, and deep-ensemble scripts.
- Synthetic data generator and automated unit tests.

`BaseTokenizer` is an abstract base class. `ClinicalTokenizer` and `MolecularPathwayTokenizer` inherit from it. The remaining model is assembled by composition because each component has a different role and interface.

## 2. Project structure

```text
PathTokenSurv/
├── configs/
│   ├── synthetic.json
│   └── real_template.json
├── pathtokensurv/
│   ├── config.py
│   ├── data/
│   │   ├── raw.py
│   │   ├── preprocessing.py
│   │   ├── pathways.py
│   │   ├── time.py
│   │   ├── dataset.py
│   │   ├── splitting.py
│   │   └── synthetic.py
│   ├── models/
│   │   ├── tokenizers.py
│   │   ├── encoders.py
│   │   ├── reconstruction.py
│   │   ├── fusion.py
│   │   ├── survival.py
│   │   └── model.py
│   ├── losses.py
│   ├── metrics.py
│   ├── trainer.py
│   ├── experiment.py
│   ├── evaluation.py
│   └── inference.py
├── scripts/
│   ├── generate_synthetic.py
│   ├── run_synthetic.py
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   ├── run_outer_cv.py
│   ├── run_nested_cv.py
│   ├── run_ensemble.py
│   ├── run_ablation.py
│   ├── evaluate_robustness.py
│   └── make_pathway_controls.py
└── tests/
```

## 3. Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[test]"
```

On Windows PowerShell, prefer `python -m pip` so that `pip` and `python` refer to the same environment:

```powershell
python -m pip install -e ".[test]"
```

Version 1.1 also bootstraps the project root inside every script, so commands such as
`python scripts/run_synthetic.py ...` work directly from an extracted source checkout even
if the editable install step has not yet been run. Installing the package remains recommended
for normal development.

For a CUDA installation, install the PyTorch build recommended for the local CUDA driver before running `pip install -e .`.

## 4. End-to-end synthetic validation

Run this first:

```bash
python scripts/run_synthetic.py --config configs/synthetic.json --clean
```

The command performs all of the following steps:

1. creates a synthetic multimodal survival cohort;
2. creates a synthetic pathway map and graph;
3. splits patients into training, validation, and test partitions;
4. fits every preprocessing operation on the training patients only;
5. trains PathTokenSurv;
6. evaluates the held-out test patients;
7. saves the fitted preprocessing and model artifacts; and
8. reloads the saved model and runs inference on the full synthetic cohort.

Expected output directory:

```text
outputs/synthetic_run/
├── best_model.pt
├── model_for_inference.pt
├── config.json
├── preprocessor.pkl
├── pathway_spec.json
├── time_discretizer.json
├── training_reference.npz
├── training_history.csv
├── training_history.png
├── test_metrics.json
├── test_predictions.csv
├── test_mean_survival.png
└── inference_predictions.csv
```

Synthetic metrics are only software checks. They are not estimates for the paper.

## 5. Required real-data files

The real-data folder must contain wide patient-level tables.

### `outcomes.tsv`

```text
patient_id    time    event    cancer_type
TCGA-XX-0001  850     1        BRCA
TCGA-XX-0002  1230    0        LUAD
```

- `time` must be positive.
- `event` must be 1 for an observed death and 0 for censoring.
- State the time origin and unit in the paper.

### `clinical.tsv`

```text
patient_id    age    sex    race    histology
TCGA-XX-0001  61     Female Group_A Invasive_ductal
```

Cancer type is intentionally excluded from the clinical predictor table. It is supplied through `outcomes.tsv` and used only by the survival baseline.

### Molecular tables

`mrna.tsv`, `mirna.tsv`, and `cnv.tsv` use one row per patient and one column per feature:

```text
patient_id    GENE1    GENE2    GENE3
TCGA-XX-0001  8.21     4.10     6.33
```

A patient may be absent from a molecular table or have an all-missing row. The loader marks that entire modality as unavailable. Sporadic missing cells within an observed row are filled with training-set medians.

### `pathways.json`

```json
{
  "pathways": ["Pathway_A", "Pathway_B"],
  "modality_features": {
    "mrna": {
      "Pathway_A": ["GENE1", "GENE2"],
      "Pathway_B": ["GENE3"]
    },
    "mirna": {
      "Pathway_A": ["hsa-miR-1"],
      "Pathway_B": ["hsa-miR-2"]
    },
    "cnv": {
      "Pathway_A": ["GENE1", "GENE2"],
      "Pathway_B": ["GENE3"]
    }
  },
  "adjacency": [[0, 1], [1, 0]]
}
```

The final TCGA experiment should build this file from documented KEGG/Reactome releases and a documented miRNA-target resource. The code does not download licensed pathway content.

## 6. Train and test on a real cohort

Edit `configs/real_template.json`, especially:

- `data.data_dir`;
- variance thresholds;
- model size;
- batch size;
- output directory.

Then run:

```bash
python scripts/train.py --config configs/real_template.json
```

The training script uses a patient-level train/validation/test split stratified by cancer and event status when the strata are large enough.

## 7. Outer cross-validation

```bash
python scripts/run_outer_cv.py \
  --config configs/real_template.json \
  --folds 5
```

Each outer fold has a separate training-only preprocessor, pathway specification, time discretizer, model, and held-out test set. An internal validation subset controls early stopping.

This script provides leakage-controlled outer cross-validation with a separate early-stopping subset in every fold.

## 8. Nested cross-validation

A compact nested search is included:

```bash
python scripts/run_nested_cv.py \
  --config configs/real_template.json \
  --search configs/nested_search_example.json \
  --outer-folds 5 \
  --inner-folds 3
```

Each candidate is assessed on inner held-out folds. A separate subset of each inner training fold controls early stopping. The selected candidate is then evaluated on the outer held-out fold.

## 9. Deep ensemble

```bash
python scripts/run_ensemble.py \
  --config configs/real_template.json \
  --members 5 \
  --fold 1 \
  --split-assignments outputs/primary_fold_01/split_assignments.csv
```

The script trains all members on the same patient split with different random seeds. It saves mean survival probabilities and between-model survival variance. For study evaluation, pass the saved primary-fold assignments so the ensemble uses the frozen train, validation, and test patients. Completed member directories are reused when a run is restarted.

## 10. Ablations and pathway controls

Run the core architecture ablations on one fixed split:

```bash
python scripts/run_ablation.py --config configs/real_template.json
```

Create negative-control pathway files:

```bash
python scripts/make_pathway_controls.py \
  --input /path/to/pathways.json \
  --output-dir /path/to/pathway_controls \
  --seed 123 \
  --swaps 10000
```

The first output shuffles pathway-feature assignments while preserving pathway sizes. The second randomizes pathway edges through degree-preserving edge swaps. Point `data.pathway_file` to each control file and rerun the model.

## 11. Missingness and modality-subset evaluation

```bash
python scripts/evaluate_robustness.py \
  --artifacts outputs/tcga_route_c \
  --data-dir /path/to/tcga_route_c \
  --output-dir outputs/robustness \
  --rates 0 0.1 0.3 0.5 0.7 \
  --repeats 5
```

This command evaluates all 15 non-empty modality subsets and controlled modality-removal rates.

## 12. External evaluation

Use this command when the new cohort has observed outcomes:

```bash
python scripts/evaluate.py \
  --artifacts outputs/tcga_route_c \
  --data-dir /path/to/external/cohort \
  --output-dir outputs/external_validation
```

The external feature names must use the same identifiers as the training cohort. The saved preprocessor selects and scales the training features without refitting them.

## 13. Inference without outcomes

An inference folder may omit valid time and event values. It must still contain the clinical and molecular files. A `cancer_type` column can be supplied in `outcomes.tsv`; unknown cancers use the global baseline entry.

```bash
python scripts/infer.py \
  --artifacts outputs/tcga_route_c \
  --data-dir /path/to/inference/cohort \
  --output outputs/predictions.csv
```

For one model, the uncertainty column is `NaN`. Use the ensemble script when predictive variance is needed.

## 14. Tests

```bash
python -m pytest
```

The tests check:

- discrete-time labels;
- monotonic survival curves within [0, 1];
- a known perfect C-index example;
- training-only preprocessing; and
- model shapes and structured modality masks.

The complete suite contains 39 tests and is also run by GitHub Actions.

## 15. Important research safeguards

1. Do not reuse the simulated values drafted for the paper.
2. Fit feature filtering, scaling, pathway selection, time bins, and tuning inside the relevant training fold.
3. Keep all aliquots from the same patient in one fold.
4. Freeze the endpoint definition before model development.
5. Check every reported metric against a validated survival-analysis package before submission.
6. Report pooled and within-cancer results because pooled pan-cancer C-index may be driven by between-cancer survival differences.
7. Run true-pathway, shuffled-pathway, and degree-matched random-graph controls.
8. Save patient-level out-of-fold predictions before creating final tables or figures.

## 16. Memory guidance

Fusion attention grows quadratically with the total token count. With about 214 pathways, the combined sequence contains roughly 650 molecular tokens before clinical tokens are added. Start with a batch size of 8 to 16 on a 10 GB GPU. Reduce `d_model`, pathway count, or batch size if memory is limited. Mixed precision can be added after the full-precision pipeline is verified.


## 17. Native TCGA PanCancer files (v1.3)

Version 1.3 can read the native matrices directly. You do not need to transpose the large files manually.
Put the following files in one folder:

```text
data/tcga_pancancer_raw/
├── PanCancer_Clinical.xlsx
├── PanCancer_mRNA.txt
├── PanCancer_miRNA.txt
├── PanCancer_CNV.txt
└── pathways.json
```

The molecular files are expected in the orientation shown by the original PanCancer downloads: features in rows and TCGA samples in columns. The loader:

- reads the `TCGA-CDR` sheet;
- uses `OS` and `OS.time` as event and time;
- keeps cancer type separate from the clinical predictor vector;
- creates clinical predictors `age`, `sex`, `race`, `stage`, and `histology`;
- prefers AJCC pathologic stage and falls back to clinical stage;
- collapses stage subgroups to Stage 0/I/II/III/IV;
- normalizes molecular sample barcodes to 12-character patient IDs;
- uses tumor sample types in priority order `01`, `03`, `09`, `06` by default (primary solid tumor, primary blood-derived tumor, and metastatic tumor when no preferred primary-type sample exists);
- averages multiple primary-tumor aliquots for the same patient by default;
- reads molecular values as float32 to reduce memory;
- keeps patients with a missing molecular modality and represents missingness through the availability mask; and
- excludes zero survival-time records by default because they do not define a positive at-risk interval.

First inspect the clinical workbook without loading the large omics files:

```powershell
python .\scripts\inspect_tcga_raw.py --config .\configs\tcga_pancancer_raw.json --clinical-only
```

Then inspect the complete aligned cohort:

```powershell
python .\scripts\inspect_tcga_raw.py --config .\configs\tcga_pancancer_raw.json --output .\outputs\tcga_raw_qc.json
```

Once `pathways.json` is present, train directly from the native files:

```powershell
python .\scripts\train.py --config .\configs\tcga_pancancer_raw.json
```

For the full paper experiment, use nested validation:

```powershell
python .\scripts\run_nested_cv.py --config .\configs\tcga_pancancer_raw.json --search .\configs\nested_search_example.json --outer-folds 5 --inner-folds 3
```

### Building `pathways.json` from local annotation tables

The package does not redistribute or automatically download pathway databases. Prepare local tables from the exact KEGG/Reactome and validated miRNA-target releases used in the paper, then run:

```powershell
python .\scripts\build_pathways.py `
  --gene-membership .\annotations\gene_pathway_membership.tsv `
  --mirna-targets .\annotations\mirna_targets.tsv `
  --pathway-edges .\annotations\pathway_edges.tsv `
  --output .\data\tcga_pancancer_raw\pathways.json
```

Templates are available in `examples/`.

### Important cohort-count note

The TCGA-CDR sheet contains 11,094 non-missing `OS.time` values, but some have `OS.time = 0`. Version 1.3 excludes zero-time observations by default because the discrete-time likelihood requires a positive follow-up interval. The final manuscript cohort count must therefore be taken from the v1.3 QC report after all matching and eligibility rules are frozen. Set `tcga_exclude_zero_survival_time` to `false` only if you adopt and document another statistically valid treatment of zero-time observations.

## 18. Detailed real-data QC and dry-run preprocessing (v1.4)

Version 1.4 adds a reproducibility-focused QC stage before any real GPU training. It produces exact natural missingness patterns, cancer-specific missingness, training-fold variance diagnostics, pathway/identifier coverage, and a manuscript-ready Markdown report.

### Step A: run detailed QC before training

```powershell
python .\scripts\run_tcga_qc.py `
  --config .\configs\tcga_pancancer_raw.json `
  --output-dir .\outputs\tcga_qc_v1.4 `
  --outer-folds 5 `
  --fold 1
```

If `pathways.json` has not been built yet, run the raw-data and variance checks first:

```powershell
python .\scripts\run_tcga_qc.py `
  --config .\configs\tcga_pancancer_raw.json `
  --output-dir .\outputs\tcga_qc_v1.4_prepathway `
  --outer-folds 5 `
  --fold 1 `
  --skip-pathways
```

The QC directory includes:

```text
upset_ready_patient_availability.csv
modality_pattern_summary.csv
cancer_specific_missingness.csv
cancer_specific_pattern_counts.csv
variance_summary.csv
variance_features_mrna.csv
variance_features_mirna.csv
variance_features_cnv.csv
selected_feature_counts.csv
annotation_validation.csv                 # when pathways.json exists
pathway_coverage_summary.csv              # when pathways.json exists
retained_pathway_sizes.csv                # when pathways.json exists
qc_pathway_spec.json                      # when pathways.json exists
SUPPLEMENTARY_QC_REPORT.md
qc_summary.json
modality_availability.png
availability_patterns.png
cancer_specific_missingness.png
variance_mrna.png
variance_mirna.png
variance_cnv.png
pathway_coverage.png                       # when pathways.json exists
```

`upset_ready_patient_availability.csv` has one row per patient and binary columns for Clinical, mRNA, miRNA, and CNV. It can be imported directly into R/Python UpSet plotting tools or used to reproduce the natural missingness table in the paper.

### Step B: build the pathway file with source pathway-size filtering

Version 1.4 can exclude annotation pathways that are too small or too broad before the model sees them. For example:

```powershell
python .\scripts\build_pathways.py `
  --gene-membership .\annotations\gene_pathway_membership.tsv `
  --mirna-targets .\annotations\mirna_targets.tsv `
  --pathway-edges .\annotations\pathway_edges.tsv `
  --min-genes-per-pathway 5 `
  --max-genes-per-pathway 500 `
  --output .\data\tcga_pancancer_raw\pathways.json
```

The resulting JSON records source pathway counts, retained sizes, unique gene identifiers, miRNA-target coverage, and retained pathway-graph edges in its `metadata` field. Keep the exact annotation releases and the command used to build this file with the study artifacts.

The model applies a second, fold-specific filter after training-only feature selection. A pathway is retained only if the selected training features supply at least `min_features_per_pathway` total features and at least `min_modalities_per_pathway` represented molecular modalities. Per-modality token size is capped by `max_features_per_pathway`.

### Step C: run a CPU dry run before GPU training

```powershell
python .\scripts\dry_run_preprocessing.py `
  --config .\configs\tcga_pancancer_raw.json `
  --output-dir .\outputs\tcga_dry_run_v1.4 `
  --outer-folds 5 `
  --fold 1 `
  --batch-size 2
```

This command performs no parameter optimization. It:

1. loads the real cohort;
2. creates one leakage-controlled outer training/test fold;
3. fits feature filtering, median values, scaling, and categorical vocabularies on training patients only;
4. constructs the fold-specific pathway specification;
5. builds the training-only discrete-time intervals;
6. instantiates PathTokenSurv on CPU; and
7. executes one no-gradient forward pass to validate tensor dimensions and finite survival probabilities.

Do not start the full nested-CV experiment until this dry run reports `"status": "PASS"`.

### Identifier validation in v1.4

Matching is now canonicalized only for annotation lookup while original feature names are preserved in saved artifacts:

- mRNA Entrez IDs: trailing `.0` is removed when present;
- CNV gene symbols: pathway matching is case-insensitive;
- mature miRNA identifiers: pathway matching is case-insensitive.

The QC report separately records the number and proportion of raw and selected features that match the pathway annotation. This is the number that should be reported in the Methods/Supplement, rather than assuming complete mapping.

### Memory improvement

Large-matrix feature statistics are now computed in configurable feature chunks (`preprocessing_feature_chunk_size`, default 2048). This avoids creating a full second median-imputed copy of the mRNA or CNV training matrix during feature selection while preserving the same variance-filtering rule.



## v1.5.2 pathway construction

PathTokenSurv v1.5.2 uses **DIANA-TarBase v9.0 as the primary miRNA-target provider** and retains miRTarBase as an optional alternative. See `PATHWAY_CONSTRUCTION_GUIDE_v1.5.2.md`.

TarBase v9 documents local retrieval through its official web interface, but the publication does not provide a stable direct bulk-file URL. The package therefore does not hard-code an unverified URL. Export/download the Homo sapiens interaction table from the official TarBase v9 portal, save it under `annotations/raw/`, and validate it:

```powershell
python .\scripts\validate_tarbase.py `
    --file .\annotations\raw\TarBase_v9_human.csv
```

Then construct the pathway resource:

```powershell
python .\scripts\construct_pathways.py `
    --download `
    --raw-dir .\annotations\raw `
    --processed-dir .\annotations\processed `
    --output .\data\tcga_pancancer_raw\pathways.json `
    --mirna-target-source tarbase `
    --tarbase-file .\annotations\raw\TarBase_v9_human.csv `
    --min-genes-per-pathway 5 `
    --max-genes-per-pathway 500
```

The TarBase parser is chunked for multi-million-row exports, restricts the TCGA mapping to human `hsa-` miRNAs, harmonizes gene symbols/Entrez IDs through NCBI gene_info, and writes a standardized `mirna_targets.tsv`. Downstream KEGG/Reactome pathway construction is unchanged.

For an optional miRTarBase sensitivity analysis, use `--mirna-target-source mirtarbase --mirtarbase-file <hsa_MTI.csv>`.
## v1.5.3 real-TCGA pathway mapping fix

Real TCGA QC showed that the loaded mRNA matrix uses gene symbols rather than an Entrez-only feature space. v1.5.3 therefore stores both gene-symbol and Entrez aliases for mRNA pathway matching. It also replaces the permissive TarBase any-target rule with pathway-level target enrichment (minimum overlap + per-miRNA BH-FDR) to avoid saturating every pathway at the miRNA feature cap. See `PATHWAY_CONSTRUCTION_GUIDE_v1.5.3.md`.


## v1.5.4 pathway-selection update

For the TCGA PathTokenSurv analysis, v1.5.4 uses score-aware top-K miRNA assignment, modality-specific pathway caps (`mRNA=128`, `miRNA=32`, `CNV=128`), equal-weight normalized multimodal pathway ranking, and conservative unique-arm miRNA alias harmonization. See `PATHWAY_CONSTRUCTION_GUIDE_v1.5.4.md` and `CHANGELOG_v1.5.4.md`.


## v1.5.5: first manuscript-grade outer-fold training run

After the v1.5.4 pathway QC and real-data CPU dry-run pass, the primary preprocessing/pathway protocol is frozen in `FROZEN_PRIMARY_PROTOCOL_v1.5.5.md`.

Train only outer fold 1 while keeping its outer test partition untouched:

```powershell
python .\scripts\run_outer_fold.py `
    --config .\configs\tcga_pancancer_raw.json `
    --outer-folds 5 `
    --fold 1 `
    --validation-fraction 0.15 `
    --output-dir .\outputs\tcga_outer_fold1_v1.5.5
```

The script uses the same outer-fold generator and base seed as `run_tcga_qc.py` and `dry_run_preprocessing.py`. It reserves 15% of the outer-development partition for early stopping, changes only the model random seed, and evaluates the final best checkpoint once on the untouched outer test fold.

Important outputs include:

- `experiment_manifest.json` — exact seeds, split sizes, config hash, pathway-mapping hash, and final metrics.
- `split_assignments.csv` — patient-level train/validation/test membership.
- `split_summary_by_cancer_event.csv` — split balance by cancer and event status.
- `training_history.csv` / `.png` and `training_summary.json`.
- `best_model.pt`, `model_for_inference.pt`, `preprocessor.pkl`, `pathway_spec.json`, `time_discretizer.json`.
- `test_metrics.json` and `test_predictions.csv`.
- `per_cancer_metrics.csv` — cancer-specific held-out metrics using cancer-specific training censoring references.
- `calibration_ipcw.csv` / `.png` — IPCW calibration diagnostics on supported horizons.

The one-fold run is a development milestone, not the final manuscript estimate. Final claims should come from the complete nested/outer validation protocol and, where specified, the deep ensemble.

## v1.5.6 performance/recovery update

v1.5.6 preserves the frozen v1.5.5 scientific protocol while improving execution efficiency and long-run recoverability. Molecular pathway tokenization is vectorized in pathway chunks, static pathway-bias masks are cached, validation is performed in one model pass, and training artifacts are persisted after each epoch. Use `python scripts/check_cuda.py --require` before a long NVIDIA run. `scripts/run_outer_fold.py --require-cuda` prevents accidental CPU-only manuscript training. See `GPU_SETUP_WINDOWS_v1.5.6.md` and `CHANGELOG_v1.5.6.md`.

## v1.5.7: cancer-stratified and missingness-aware held-out evaluation

v1.5.7 adds a frozen evaluation amendment without changing the v1.5.6 training protocol. See `EVALUATION_AMENDMENT_v1.5.7.md`.

Re-evaluate an already completed outer fold without retraining:

```powershell
python .\scripts\evaluate_outer_fold_extended.py `
    --artifacts .\outputs\tcga_outer_fold1_v1.5.6_cuda `
    --data-dir .\data\tcga_pancancer_raw `
    --device cuda `
    --require-cuda
```

Train the cancer-only and clinical-only baselines on exactly the same fold:

```powershell
python .\scripts\run_fold_baselines.py `
    --artifacts .\outputs\tcga_outer_fold1_v1.5.6_cuda `
    --data-dir .\data\tcga_pancancer_raw `
    --baselines cancer_only clinical_only `
    --device cuda `
    --require-cuda
```

Run outer folds 2–5 sequentially (extended evaluation is saved automatically):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_remaining_outer_folds.ps1
```

Add `-RunBaselines` to that PowerShell command if the two frozen-split baselines should be trained immediately after each primary fold.

## v1.5.8: aggregate missingness robustness across outer folds

After all five v1.5.7 extended evaluations are complete, aggregate their natural-missingness and controlled complete-case modality-drop results with:

```powershell
python .\scripts\aggregate_missingness_robustness.py `
    --fold-dirs `
        .\outputs\tcga_outer_fold1_v1.5.6_cuda `
        .\outputs\tcga_outer_fold2_v1.5.7_cuda `
        .\outputs\tcga_outer_fold3_v1.5.7_cuda `
        .\outputs\tcga_outer_fold4_v1.5.7_cuda `
        .\outputs\tcga_outer_fold5_v1.5.7_cuda `
    --output-dir .\outputs\tcga_missingness_robustness_v1.5.8
```

This command is evaluation-only. It does not retrain or modify any fold checkpoint. See `MISSINGNESS_AGGREGATION_PROTOCOL_v1.5.8.md` for the statistical and interpretation rules.

## v1.5.9 frozen architecture ablations

v1.5.9 adds a resume-safe five-fold architecture ablation suite without retraining the completed primary model. See `ARCHITECTURE_ABLATION_PROTOCOL_v1.5.9.md`.

Run all prespecified ablations on Windows/PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_architecture_ablations.ps1
```

The five ablations are `no_structured_masking`, `no_reconstruction_objective`, `no_subset_consistency`, `no_pathway_bias`, and `no_cancer_conditioning`. Every fold reuses the exact split, preprocessor, pathway specification, time discretizer, and model seed from its corresponding frozen primary outer fold. Completed runs are skipped and interrupted runs resume from `last_model.pt` by default.

## v1.6.0 pathway-tokenization control

v1.6.0 adds a secondary five-fold sensitivity analysis that compares the frozen primary biological pathway-token assignment with a structure-preserving feature-label permutation. The control preserves pathway-token sizes, overlap topology, residual size, pathway adjacency, model capacity, preprocessing, patient splits, and model seeds while breaking the biological feature-to-pathway correspondence.

Run all five restart-safe control folds and aggregate them with:

```powershell
powershell -ExecutionPolicy Bypass `
    -File .\scripts\run_pathway_token_control.ps1
```

See `PATHWAY_TOKENIZATION_CONTROL_PROTOCOL_v1.6.0.md` for the exact control definition and interpretation rules. This analysis is labeled secondary because it was motivated after the prespecified v1.5.9 architecture-ablation suite.

## License

PathTokenSurv is available under the MIT License. See `LICENSE`.
