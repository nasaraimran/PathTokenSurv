# PathTokenSurv publication figure package

## Quick start

From this `figures` directory:

```powershell
python -m pip install -r .\requirements_figures.txt
python .\make_all.py --data-dir .\data --output-dir .\outputs
```

All core figures are written as PDF, SVG, and 600-dpi PNG.

After the Kaplan–Meier analysis has been run:

```powershell
python .\make_all.py `
  --data-dir .\data `
  --output-dir .\outputs `
  --km-dir ..\outputs\tcga_kaplan_meier_oof
```

or run only the KM figure:

```powershell
python .\fig07_kaplan_meier.py `
  --km-dir ..\outputs\tcga_kaplan_meier_oof `
  --output-dir .\outputs
```

## Rebuilding `data/` from your four summary ZIPs

```powershell
python .\prepare_data.py `
  --outer .\tcga_outer_cv_v1.5.7_summary.zip `
  --missingness .\tcga_missingness_robustness_v1.5.8.zip `
  --ablation .\tcga_architecture_ablation_summary_v1.5.9.zip `
  --pathway-control .\tcga_pathway_token_control_summary_v1.6.0.zip `
  --output-dir .\data
```

## Important interpretation rules

1. Use pooled and cancer-stratified discrimination together.
2. Natural complete-vs-incomplete comparisons are observational.
3. Controlled modality-drop experiments use the same complete-case patients
   and are the cleaner missingness stress test.
4. Structured masking is the strongest supported architecture component.
5. Do not claim an independent benefit for pathway attention bias.
6. Treat v1.6.0 pathway permutation as a secondary sensitivity analysis.
7. Do not use the old simulated Route C numbers.
8. The KM risk groups must be defined from held-out OOF risk scores, not from
   survival outcomes.
