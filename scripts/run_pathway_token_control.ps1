param(
    [string]$Config = ".\configs\tcga_pancancer_raw.json",
    [string]$OutputRoot = ".\outputs\tcga_pathway_token_control_v1.6.0",
    [string]$SummaryDir = ".\outputs\tcga_pathway_token_control_summary_v1.6.0",
    [int]$PermutationSeedBase = 160000,
    [int[]]$Folds = @(1,2,3,4,5),
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

foreach ($fold in $Folds) {
    if (-not $primary.ContainsKey($fold)) { throw "Invalid outer fold: $fold" }
    $primaryDir = $primary[$fold]
    $out = Join-Path $OutputRoot ("fold_{0:D2}" -f $fold)
    $manifest = Join-Path $out "pathway_control_manifest.json"
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
        Write-Host "=== SKIP completed pathway-token control fold $fold ==="
        continue
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "PathTokenSurv v1.6.0: feature-permuted pathway tokens / fold $fold"
    Write-Host "============================================================"

    $resumeArgs = @()
    $last = Join-Path $out "last_model.pt"
    if ((-not $NoResume) -and (Test-Path $last)) {
        Write-Host "Resuming partial run from $last"
        $resumeArgs = @("--resume-from", $last)
    }

    $permutationSeed = $PermutationSeedBase + $fold
    $args = @(
        ".\scripts\run_pathway_token_control_fold.py",
        "--config", $Config,
        "--primary-artifacts", $primaryDir,
        "--output-dir", $out,
        "--permutation-seed", $permutationSeed,
        "--device", "cuda",
        "--require-cuda"
    ) + $resumeArgs
    python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Pathway-token control failed: fold $fold"
    }
}

if (-not $SkipAggregate) {
    if ($Folds.Count -ne 5) {
        Write-Host "Skipping automatic aggregation because a subset of folds was requested."
    } else {
        Write-Host ""
        Write-Host "=== Aggregating pathway-tokenization control ==="
        python .\scripts\aggregate_pathway_token_control.py `
            --primary-fold-dirs `
                $primary[1] `
                $primary[2] `
                $primary[3] `
                $primary[4] `
                $primary[5] `
            --control-root $OutputRoot `
            --output-dir $SummaryDir
        if ($LASTEXITCODE -ne 0) { throw "Pathway-token control aggregation failed." }
    }
}

Write-Host "All requested pathway-token control folds completed."
