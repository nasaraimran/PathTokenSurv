# PathTokenSurv publication figure manifest

This folder uses the **completed real experiments only**:

- v1.5.7 five-fold outer evaluation and clinical/cancer-only baselines
- v1.5.8 natural missingness and controlled complete-case modality removal
- v1.5.9 prespecified architecture ablations
- v1.6.0 secondary biological pathway-token permutation control
- Kaplan–Meier analysis when its out-of-fold output folder is supplied

No simulated Route C values are used.

## Main-text figures

### Figure 3 — Overall survival prediction performance
Scripts: `fig03_main_performance.py`

Generated panels:
- `Figure3A_Main_Pooled_CIndex`
- `Figure3B_Main_IBS`
- `Figure3C_Main_Mean_tdAUC`
- `Figure3D_Main_CancerStratified_CIndex`

**Caption:**  
**Figure 3. Overall survival prediction performance across five outer folds.**
PathTokenSurv is compared with clinical-only and cancer-only reference models.
Error bars show the two-sided 95% Student-t interval across the five held-out
outer-fold estimates. Higher values indicate better performance for the
C-index and mean time-dependent AUC, whereas lower values indicate better
performance for the integrated Brier score. The cancer-stratified C-index is
reported separately because pooled pan-cancer concordance can reflect broad
survival differences between cancer types.

**Insert:** Results, immediately after the paragraph that first reports the
five-fold PathTokenSurv, clinical-only, and cancer-only performance.

### Figure 4 — Robustness to incomplete molecular inputs
Script: `fig04_missingness.py`

Generated panels:
- `Figure4A_Controlled_Modality_Dropout_CIndex`
- `Figure4B_Controlled_Modality_Dropout_Stratified_CIndex`
- `Figure4C_Natural_Missingness_Stratified_CIndex`
- `Figure4D_Natural_Missingness_IBS`

**Caption:**  
**Figure 4. Robustness of PathTokenSurv to incomplete molecular inputs.**
Controlled complete-case experiments evaluate the same patients after
removing individual molecular modalities or retaining clinical variables
alone. Natural-missingness analyses compare patients with complete and
incomplete molecular profiles. The natural comparison is observational and
should not be interpreted as a causal effect of missing data. Error bars show
cross-fold variability.

**Insert:** Results, "Robustness to missing molecular modalities", directly
after the first paragraph describing the complete/incomplete and controlled
dropout experiments.

### Figure 5 — Architecture ablation analysis
Script: `fig05_ablation.py`

Generated panels:
- `Figure5A_Ablation_Pooled_CIndex_Degradation`
- `Figure5B_Ablation_Stratified_CIndex_Degradation`
- `Figure5C_Ablation_IBS_Degradation`

**Caption:**  
**Figure 5. Matched five-fold ablation analysis of PathTokenSurv.**
Points show the mean degradation after removing one model component, and
horizontal bars show two-sided 95% Student-t intervals across matched outer
folds. Positive degradation indicates that the full model performed better
under the metric-specific direction. Structured modality masking produced
the clearest and most consistent loss in discrimination. The pathway-bias
ablation did not show measurable incremental benefit.

**Insert:** Results, "Ablation study", immediately after the paragraph that
identifies structured masking as the strongest component.

### Figure 6 — Biological pathway-token sensitivity
Script: `fig06_pathway_control.py`

Generated panels:
- `Figure6A_Pathway_Control_Stratified_CIndex`
- `Figure6B_Pathway_Control_Pooled_CIndex`
- `Figure6C_Pathway_Control_Metric_Degradation`

**Caption:**  
**Figure 6. Secondary sensitivity analysis of biological pathway-token
assignment.** The control deterministically permutes feature identities
within each molecular modality while preserving pathway-token sizes,
feature-degree distribution, token-overlap structure, residual-token size,
pathway adjacency, model capacity, and the frozen training protocol.
Biological assignment showed a consistent advantage for cancer-stratified
C-index across the five outer folds. Other performance differences were
smaller and their cross-fold confidence intervals included zero.

**Insert:** Results, "Biological pathway tokenization sensitivity analysis",
after the first paragraph that reports biological versus permuted results.
Describe this as a secondary/post-hoc sensitivity analysis.

### Figure 7 — Kaplan–Meier risk stratification
Script: `fig07_kaplan_meier.py`

**Caption:**  
**Figure 7. Kaplan–Meier overall survival according to PathTokenSurv
out-of-fold risk quartiles.** Patients are ranked by held-out PathTokenSurv
risk score within cancer type and assigned to four risk groups without using
survival outcomes to determine the cut points. Curves show Kaplan–Meier
estimates with 95% confidence intervals and numbers at risk. The plot reports
the global log-rank test and, when cancer type is available, a
cancer-stratified Cox global test.

**Insert:** Results, after the main discrimination/calibration paragraph and
before the Discussion. If the final KM separation is mainly confirmatory,
move this figure to the Supplementary Material instead.

## Supplementary figures

### Figure S1 — Outer-fold stability
`Supplementary_Figure_S1_Outer_Fold_Stability`

Shows fold-wise pooled C-index for PathTokenSurv, clinical-only, and
cancer-only models.

### Figure S2 — Integrated calibration error
`Supplementary_Figure_S2_Integrated_ECE`

Shows five-fold integrated IPCW-ECE. This is a calibration *summary*, not a
replacement for horizon-specific calibration curves.

### Figure S3 — Natural modality-availability patterns
`Supplementary_Figure_S3_Modality_Availability_Patterns`

Shows the number of patients in each observed modality pattern. This figure
documents the real missingness structure of the cohort.

### Figure S4 — Natural missingness across folds
`Supplementary_Figure_S4_Natural_Missingness_Foldwise`

Shows the cancer-stratified C-index of complete and incomplete patient groups
across outer folds. Treat this comparison as observational.

### Figure S5 — Fold-wise ablation effects
`Supplementary_Figure_S5_Ablation_Foldwise`

Shows the individual outer-fold C-index degradation for each prespecified
architecture ablation.

## Optional figures that need patient/horizon-level files not present in the
four supplied aggregate ZIPs

- `optional_per_cancer.py`: cancer-specific C-index plot. Requires a finalized
  cancer-level CSV with `cancer_type`, `c_index`, and `events`.
- `optional_calibration_curves.py`: one calibration plot per horizon. Requires
  a CSV with `horizon`, `predicted_survival`, and `observed_survival`.

These scripts are intentionally optional. The package does not invent
patient-level or horizon-level values from aggregate summaries.
