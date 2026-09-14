param(
    [string]$Config = ".\configs\tcga_pancancer_raw.json",
    [string]$OutputRoot = ".\outputs\tcga_architecture_ablations_v1.5.9",
    [string]$SummaryDir = ".\outputs\tcga_architecture_ablation_summary_v1.5.9",
    [string[]]$Ablations = @(
        "no_structured_masking",
        "no_reconstruction_objective",
        "no_subset_consistency",
        "no_pathway_bias",
        "no_cancer_conditioning"
    ),
    [switch]$NoResume,
    [switch]$SkipAggregate
)

$ErrorActionPreference = "Stop"

$primary = @{
    1 = ".\outputs\tcga_outer_fold1_v1.5.6_cuda"
    2 = ".\outputs\tcga_outer_fold2_v1.5.7_cuda"
    3 = ".\outputs\tcga_outer_fold3_v1.5.7_cuda"
    4 = ".\outputs\tcga_outer_fold4_v1.5.7_cuda"
    5 = ".\outputs\tcga_outer_fold5_v1.5.7_cuda"
}

foreach ($ablation in $Ablations) {
    foreach ($fold in 1..5) {
        $primaryDir = $primary[$fold]
        $out = Join-Path $OutputRoot (Join-Path $ablation ("fold_{0:D2}" -f $fold))
        $manifest = Join-Path $out "ablation_manifest.json"
        $metrics = Join-Path $out "test_metrics_extended.json"

        $complete = $false
        if ((Test-Path $manifest) -and (Test-Path $metrics)) {
            try {
                $m = Get-Content $manifest -Raw | ConvertFrom-Json
                if ($m.status -eq "PASS") { $complete = $true }
            } catch {
                $complete = $false
            }
        }
        if ($complete) {
            Write-Host "=== SKIP completed: $ablation fold $fold ==="
            continue
        }

        Write-Host ""
        Write-Host "============================================================"
        Write-Host "PathTokenSurv v1.5.9: $ablation / fold $fold"
        Write-Host "============================================================"

        $resumeArgs = @()
        $last = Join-Path $out "last_model.pt"
        if ((-not $NoResume) -and (Test-Path $last)) {
            Write-Host "Resuming partial run from $last"
            $resumeArgs = @("--resume-from", $last)
        }

        $args = @(
            ".\scripts\run_frozen_ablation_fold.py",
            "--config", $Config,
            "--primary-artifacts", $primaryDir,
            "--ablation", $ablation,
            "--output-dir", $out,
            "--device", "cuda",
            "--require-cuda"
        ) + $resumeArgs
        python @args
        if ($LASTEXITCODE -ne 0) {
            throw "Ablation failed: $ablation fold $fold"
        }
    }
}

if (-not $SkipAggregate) {
    Write-Host ""
    Write-Host "=== Aggregating frozen architecture ablations ==="
    python .\scripts\aggregate_architecture_ablations.py `
        --primary-fold-dirs `
            $primary[1] `
            $primary[2] `
            $primary[3] `
            $primary[4] `
            $primary[5] `
        --ablation-root $OutputRoot `
        --output-dir $SummaryDir
    if ($LASTEXITCODE -ne 0) { throw "Architecture ablation aggregation failed." }
}

Write-Host "All requested architecture ablations completed."
