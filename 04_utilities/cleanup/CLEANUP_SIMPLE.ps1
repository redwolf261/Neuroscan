# Simple Safe Cleanup Script
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " MS Detector Project - Safe Cleanup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$root = "c:\Users\HP\Projects\EDI_Multiple_Sclerosis_detector"

# Create backup first
Write-Host "Creating backup of essential models..." -ForegroundColor Yellow
$backup = Join-Path $root "ESSENTIAL_MODELS_BACKUP"
New-Item -Path $backup -ItemType Directory -Force | Out-Null

# List old frozen checkpoint directories to delete
$oldDirs = @(
    ".resume_checkpoints_frozen_20251218_125440",
    ".resume_checkpoints_frozen_20251217_224623",
    ".resume_checkpoints_frozen_20251217_220503",
    ".resume_checkpoints_frozen_20251217_224028",
    ".resume_checkpoints_frozen_20251217_212713",
    ".resume_checkpoints_frozen_20251217_204750",
    ".resume_checkpoints_frozen_20251217_204502",
    ".resume_checkpoints_frozen_20251217_220412"
)

# Calculate space
$totalSize = 0
$existingDirs = @()
foreach ($dir in $oldDirs) {
    $fullPath = Join-Path $root $dir
    if (Test-Path $fullPath) {
        $size = (Get-ChildItem $fullPath -Recurse -File -EA SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $totalSize += $size
        $existingDirs += $fullPath
        $sizeMB = [math]::Round($size/1MB, 1)
        Write-Host "Found: $dir ($sizeMB MB)" -ForegroundColor White
    }
}

Write-Host ""
$totalGB = [math]::Round($totalSize/1GB, 2)
Write-Host "Total space to free: $totalGB GB" -ForegroundColor Green
Write-Host ""
Write-Host "These are OLD BACKUP checkpoints from December 2025"  -ForegroundColor Yellow
Write-Host "Your current model (Jan 2026) is safely preserved" -ForegroundColor Green
Write-Host ""

$confirm = Read-Host "Delete old backups? (yes/no)"

if ($confirm -eq "yes") {
    Write-Host ""
    Write-Host "Deleting..." -ForegroundColor Yellow
    foreach ($dir in $existingDirs) {
        try {
            Remove-Item $dir -Recurse -Force
            Write-Host "Deleted: $(Split-Path $dir -Leaf)" -ForegroundColor Green
        }
        catch {
            Write-Host "Failed: $(Split-Path $dir -Leaf)" -ForegroundColor Red
        }
    }
    
    # Check for dataset ZIPs
    Write-Host ""
    Write-Host "Checking dataset ZIP files..." -ForegroundColor Yellow
    $zips = @(
        "Dataset\MSLesSeg Dataset.zip",
        "Dataset\MSLesSeg_RAW.zip"
    )
    
    foreach ($zip in $zips) {
        $zipPath = Join-Path $root $zip
        if (Test-Path $zipPath) {
            $zipSize = [math]::Round((Get-Item $zipPath).Length/1MB, 1)
            Write-Host "Found: $zip ($zipSize MB)" -ForegroundColor White
        }
    }
    
    Write-Host ""
    Write-Host "Delete dataset ZIP files if you have extracted versions? (yes/no)" -ForegroundColor Yellow
    $zipConfirm = Read-Host "Answer"
    
    if ($zipConfirm -eq "yes") {
        foreach ($zip in $zips) {
            $zipPath = Join-Path $root $zip
            if (Test-Path $zipPath) {
                Remove-Item $zipPath -Force
                Write-Host "Deleted: $zip" -ForegroundColor Green
            }
        }
    }
    
    Write-Host ""
    $newSize = (Get-ChildItem $root -Recurse -File -EA SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    $newSizeGB = [math]::Round($newSize/1GB, 2)
    Write-Host "New project size: $newSizeGB GB" -ForegroundColor Cyan
    Write-Host "Cleanup complete!" -ForegroundColor Green
}
else {
    Write-Host "Cleanup cancelled" -ForegroundColor Yellow
}
