# Phase 5 Cleanup - Ablation Study Checkpoints
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Phase 5: Ablation Study Cleanup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$root = "c:\Users\HP\Projects\EDI_Multiple_Sclerosis_detector"
$checkpointsPath = Join-Path $root "checkpoints"

Write-Host "Ablation study checkpoints are experiment results" -ForegroundColor Yellow
Write-Host "from testing different model configurations." -ForegroundColor Yellow
Write-Host ""
Write-Host "These can be safely deleted if:" -ForegroundColor White
Write-Host "  - Results are documented in ablation_results/" -ForegroundColor White
Write-Host "  - You're not re-running these experiments" -ForegroundColor White
Write-Host ""

$ablationDirs = @(
    ".resume_ablation_baseline",
    ".resume_ablation_causal",
    ".resume_ablation_consistency",
    ".resume_ablation_evidential",
    ".resume_ablation_fdr",
    ".resume_ablation_self_correction",
    ".resume_checkpoints_usald",
    ".resume_checkpoints_optimal",
    ".resume_checkpoints"
)

$totalSize = 0
$foundDirs = @()

Write-Host "Found ablation checkpoint directories:" -ForegroundColor White
foreach ($dir in $ablationDirs) {
    $fullPath = Join-Path $checkpointsPath $dir
    if (Test-Path $fullPath) {
        $files = Get-ChildItem $fullPath -File -EA SilentlyContinue
        if ($files) {
            $dirSize = ($files | Measure-Object -Property Length -Sum).Sum
            $dirSizeMB = [math]::Round($dirSize/1MB, 1)
            $totalSize += $dirSize
            $foundDirs += $fullPath
            Write-Host "  $dir : $($files.Count) files ($dirSizeMB MB)" -ForegroundColor White
        }
    }
}

$totalSizeGB = [math]::Round($totalSize/1GB, 2)
Write-Host ""
Write-Host "Total ablation checkpoints: $totalSizeGB GB" -ForegroundColor Cyan
Write-Host ""

if ($totalSizeGB -gt 0) {
    Write-Host "Check ablation_results/ for documented results:" -ForegroundColor Yellow
    $ablationResultsPath = Join-Path $root "ablation_results"
    if (Test-Path $ablationResultsPath) {
        $csvFiles = Get-ChildItem $ablationResultsPath -Filter "*.csv" -EA SilentlyContinue
        if ($csvFiles) {
            Write-Host "  Found $($csvFiles.Count) result CSV files" -ForegroundColor Green
            foreach ($csv in $csvFiles | Select-Object -First 5) {
                Write-Host "    - $($csv.Name)" -ForegroundColor Green
            }
        }
    }
    
    Write-Host ""
    Write-Host "These checkpoint folders contain intermediate training states" -ForegroundColor White
    Write-Host "from ablation experiments (Nov 2025)." -ForegroundColor White
    Write-Host ""
    Write-Host "Your CURRENT models are safely preserved in:" -ForegroundColor Green
    Write-Host "  - ESSENTIAL_MODELS_BACKUP/" -ForegroundColor Green
    Write-Host "  - segmentation/best_model.pth" -ForegroundColor Green
    Write-Host "  - deployment/model.pth" -ForegroundColor Green
    Write-Host ""
    
    $confirm = Read-Host "Delete all ablation study checkpoints? (yes/no)"
    
    if ($confirm -eq "yes") {
        Write-Host ""
        Write-Host "Deleting ablation checkpoints..." -ForegroundColor Yellow
        
        foreach ($dir in $foundDirs) {
            try {
                Remove-Item $dir -Recurse -Force
                $dirName = Split-Path $dir -Leaf
                Write-Host "  Deleted: $dirName" -ForegroundColor Green
            }
            catch {
                $dirName = Split-Path $dir -Leaf
                Write-Host "  Failed: $dirName" -ForegroundColor Red
            }
        }
        
        Write-Host ""
        Write-Host "Deleted $totalSizeGB GB of ablation checkpoints" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
$newSize = (Get-ChildItem $root -Recurse -File -EA SilentlyContinue | Measure-Object -Property Length -Sum).Sum
$newSizeGB = [math]::Round($newSize/1GB, 2)
Write-Host "Current project size: $newSizeGB GB" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
