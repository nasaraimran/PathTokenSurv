param(
    [string]$Config = ".\configs\tcga_pancancer_raw.json",
    [string]$DataDir = ".\data\tcga_pancancer_raw",
    [switch]$RunBaselines
)

$ErrorActionPreference = "Stop"

foreach ($fold in 2..5) {
    $out = ".\outputs\tcga_outer_fold${fold}_v1.5.7_cuda"
    Write-Host "=== PathTokenSurv outer fold $fold ==="
    python .\scripts\run_outer_fold.py `
        --config $Config `
        --outer-folds 5 `
        --fold $fold `
        --validation-fraction 0.15 `
        --output-dir $out `
        --require-cuda
    if ($LASTEXITCODE -ne 0) { throw "Outer fold $fold failed." }

    if ($RunBaselines) {
        Write-Host "=== Frozen-split baselines for fold $fold ==="
        python .\scripts\run_fold_baselines.py `
            --artifacts $out `
            --data-dir $DataDir `
            --baselines cancer_only clinical_only `
            --device cuda `
            --require-cuda
        if ($LASTEXITCODE -ne 0) { throw "Baselines for fold $fold failed." }
    }
}

Write-Host "All requested outer folds completed."
