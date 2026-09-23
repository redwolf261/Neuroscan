# Safe Project Cleanup Script
# This script ONLY removes old backups, duplicates, and cache files
# Your current models and data are PROTECTED

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   SAFE PROJECT CLEANUP - MS Detector Project    " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

$projectPath = "c:\Users\HP\Projects\EDI_Multiple_Sclerosis_detector"

# Step 1: Create backup of essential models
Write-Host "Step 1 of 5 - Backing up essential models..." -ForegroundColor Yellow
$backupPath = Join-Path $projectPath "ESSENTIAL_MODELS_BACKUP"
if (-not (Test-Path $backupPath)) {
    New-Item -Path $backupPath -ItemType Directory -Force | Out-Null
    Write-Host "  Created backup directory" -ForegroundColor Green
}

# Backup current working models
$essentialModels = @(
    ".resume_checkpoints_frozen_20251218_135038\seg_resume.pth",
    ".resume_checkpoints_frozen_20251218_135038\mae_resume.pth",
    "deployment\model.pth",
    "segmentation\best_model.pth",
    "checkpoints_none\best_checkpoint.pth"
)

foreach ($model in $essentialModels) {
    $sourcePath = Join-Path $projectPath $model
    if (Test-Path $sourcePath) {
        $destPath = Join-Path $backupPath (Split-Path $model -Leaf)
        if (-not (Test-Path $destPath)) {
            Copy-Item $sourcePath $destPath -Force
            Write-Host "  ✓ Backed up: $(Split-Path $model -Leaf)" -ForegroundColor Green
        }
    }
}

Write-Host ""

# Step 2: Calculate space to be freed
Write-Host "Step 2 of 5 - Calculating space to be freed..." -ForegroundColor Yellow

$itemsToDelete = @()
$totalSizeToFree = 0

# Old frozen checkpoints (keep the most recent one from Jan 2026)
$oldFrozenDirs = @(
    ".resume_checkpoints_frozen_20251218_125440",
    ".resume_checkpoints_frozen_20251217_224623",
    ".resume_checkpoints_frozen_20251217_220503",
    ".resume_checkpoints_frozen_20251217_224028",
    ".resume_checkpoints_frozen_20251217_212713",
    ".resume_checkpoints_frozen_20251217_204750",
    ".resume_checkpoints_frozen_20251217_204502",
    ".resume_checkpoints_frozen_20251217_220412"
)

foreach ($dir in $oldFrozenDirs) {
    $fullPath = Join-Path $projectPath $dir
    if (Test-Path $fullPath) {
        $size = (Get-ChildItem $fullPath -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $itemsToDelete += @{Path=$fullPath; Size=$size; Type="Old Frozen Checkpoint"}
        $totalSizeToFree += $size
    }
}

# Dataset ZIP files (only if extracted data exists)
$datasetZips = @(
    "Dataset\MSLesSeg Dataset.zip",
    "Dataset\MSLesSeg_RAW.zip",
    "Dataset\MSLesSeg-2024-main.zip",
    "Dataset\mat_files.zip"
)

# Check if dataset is extracted
$datasetExtracted = Test-Path (Join-Path $projectPath "Dataset\*.nii.gz")
if ($datasetExtracted) {
    foreach ($zip in $datasetZips) {
        $fullPath = Join-Path $projectPath $zip
        if (Test-Path $fullPath) {
            $size = (Get-Item $fullPath).Length
            $itemsToDelete += @{Path=$fullPath; Size=$size; Type="Dataset ZIP Archive"}
            $totalSizeToFree += $size
        }
    }
} else {
    Write-Host "  ⚠ Dataset not extracted - keeping ZIP files" -ForegroundColor Yellow
}

# Python cache files
$cacheFiles = Get-ChildItem -Path $projectPath -Include "__pycache__","*.pyc","*.pyo" -Recurse -Force -ErrorAction SilentlyContinue
foreach ($cache in $cacheFiles) {
    $size = if ($cache.PSIsContainer) {
        (Get-ChildItem $cache.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    } else {
        $cache.Length
    }
    $itemsToDelete += @{Path=$cache.FullName; Size=$size; Type="Python Cache"}
    $totalSizeToFree += $size
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  CLEANUP SUMMARY" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Total items to delete: $($itemsToDelete.Count)" -ForegroundColor White
Write-Host "  Total space to free: $([math]::Round($totalSizeToFree/1GB, 2)) GB" -ForegroundColor Green
Write-Host ""

# Show details
Write-Host "Items to be deleted:" -ForegroundColor Yellow
$itemsToDelete | Group-Object Type | ForEach-Object {
    $groupSize = ($_.Group | Measure-Object -Property Size -Sum).Sum
    $sizeInGB = [math]::Round($groupSize/1GB, 2)
    $itemName = $_.Name
    $itemCount = $_.Count
    $gbUnit = " GB"
    $displayText = "  - " + $itemName + ": " + $itemCount + " items (" + $sizeInGB + $gbUnit + ")"
    Write-Host $displayText -ForegroundColor White
}
Write-Host ""

# Step 3: Confirm deletion
Write-Host "Step 3 of 5 - Confirmation required" -ForegroundColor Yellow
Write-Host "  - Your current models are BACKED UP" -ForegroundColor Green
Write-Host "  - Your dataset files (.nii.gz) will NOT be touched" -ForegroundColor Green
Write-Host "  - Your source code will NOT be touched" -ForegroundColor Green
Write-Host ""
$confirmation = Read-Host "Proceed with deletion? (yes/no)"

if ($confirmation -ne "yes") {
    Write-Host ""
    Write-Host "Cleanup cancelled. No files were deleted." -ForegroundColor Yellow
    exit
}

# Step 4: Delete items
Write-Host ""
Write-Host "Step 4 of 5 - Deleting items..." -ForegroundColor Yellow
$deletedCount = 0
$deletedSize = 0

foreach ($item in $itemsToDelete) {
    try {
        if (Test-Path $item.Path) {
            Remove-Item $item.Path -Recurse -Force -ErrorAction Stop
            $deletedCount++
            $deletedSize += $item.Size
            Write-Host "  ✓ Deleted: $(Split-Path $item.Path -Leaf)" -ForegroundColor Green
        }
    } catch {
        Write-Host "  ✗ Failed to delete: $(Split-Path $item.Path -Leaf)" -ForegroundColor Red
    }
}

# Step 5: Final report
Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  CLEANUP COMPLETE" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Items deleted: $deletedCount" -ForegroundColor Green
Write-Host "  Space freed: $([math]::Round($deletedSize/1GB, 2)) GB" -ForegroundColor Green
Write-Host ""

# Calculate new project size
$newSize = (Get-ChildItem -Path $projectPath -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
Write-Host "  New project size: $([math]::Round($newSize/1GB, 2)) GB" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Backup location: $backupPath" -ForegroundColor Yellow
Write-Host ""
Write-Host "Cleanup successful! Your models and data are safe." -ForegroundColor Green
