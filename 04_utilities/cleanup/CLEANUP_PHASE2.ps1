# Phase 2 Cleanup - Remove Duplicate Checkpoints
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Phase 2: Duplicate Checkpoint Cleanup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$root = "c:\Users\HP\Projects\EDI_Multiple_Sclerosis_detector"

Write-Host "This will delete OLD DUPLICATE checkpoint files" -ForegroundColor Yellow
Write-Host "Your newest models are already backed up in ESSENTIAL_MODELS_BACKUP" -ForegroundColor Green
Write-Host ""
Write-Host "Space to free: 7.62 GB" -ForegroundColor Cyan
Write-Host ""

# List of old ablation/resume directories with duplicates
$oldResumeDirs = @(
    ".resume_checkpoints",
    ".resume_checkpoints_optimal",
    ".resume_checkpoints_usald",
    ".resume_ablation_baseline",
    ".resume_ablation_evidential",
    ".resume_ablation_causal",
    ".resume_ablation_self_correction",
    ".resume_ablation_consistency",
    ".resume_ablation_fdr",
    ".resume_checkpoints_frozen_20251218_135038"
)

Write-Host "Directories to delete (contain old duplicate checkpoints):" -ForegroundColor Yellow
foreach ($dir in $oldResumeDirs) {
    $fullPath = Join-Path $root $dir
    if (Test-Path $fullPath) {
        $size = (Get-ChildItem $fullPath -Recurse -File -EA SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $sizeMB = [math]::Round($size/1MB, 1)
        Write-Host "  - $dir ($sizeMB MB)" -ForegroundColor White
    }
}

# Old checkpoint subdirectories with small duplicates
$oldCheckpointSubdirs = @(
    "checkpoints_csrf",
    "checkpoints_cbam",
    "checkpoints_se"
)

Write-Host ""
Write-Host "Old attention mechanism test checkpoints:" -ForegroundColor Yellow
foreach ($dir in $oldCheckpointSubdirs) {
    $fullPath = Join-Path $root $dir
    if (Test-Path $fullPath) {
        $size = (Get-ChildItem $fullPath -Recurse -File -EA SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $sizeMB = [math]::Round($size/1MB, 1)
        Write-Host "  - $dir ($sizeMB MB)" -ForegroundColor White
    }
}

Write-Host ""
Write-Host "PROTECTED (will NOT be deleted):" -ForegroundColor Green
Write-Host "  - ESSENTIAL_MODELS_BACKUP (your current models)" -ForegroundColor Green
Write-Host "  - deployment/model.pth" -ForegroundColor Green
Write-Host "  - segmentation/best_model.pth" -ForegroundColor Green
Write-Host "  - Dataset files" -ForegroundColor Green
Write-Host ""

$confirm = Read-Host "Proceed with deletion? (yes/no)"

if ($confirm -eq "yes") {
    Write-Host ""
    Write-Host "Deleting old checkpoint directories..." -ForegroundColor Yellow
    
    $totalDeleted = 0
    foreach ($dir in ($oldResumeDirs + $oldCheckpointSubdirs)) {
        $fullPath = Join-Path $root $dir
        if (Test-Path $fullPath) {
            try {
                $size = (Get-ChildItem $fullPath -Recurse -File -EA SilentlyContinue | Measure-Object -Property Length -Sum).Sum
                Remove-Item $fullPath -Recurse -Force
                $totalDeleted += $size
                Write-Host "  Deleted: $dir" -ForegroundColor Green
            }
            catch {
                Write-Host "  Failed: $dir" -ForegroundColor Red
            }
        }
    }
    
    $totalDeletedGB = [math]::Round($totalDeleted/1GB, 2)
    Write-Host ""
    Write-Host "Space freed: $totalDeletedGB GB" -ForegroundColor Green
    
    # Calculate new size
    $newSize = (Get-ChildItem $root -Recurse -File -EA SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    $newSizeGB = [math]::Round($newSize/1GB, 2)
    Write-Host "New project size: $newSizeGB GB" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Cleanup complete!" -ForegroundColor Green
}
else {
    Write-Host "Cleanup cancelled" -ForegroundColor Yellow
}
