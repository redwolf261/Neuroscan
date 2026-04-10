# Additional Cleanup - Checkpoints
Write-Host "======================================" -ForegroundColor Cyan
Write-Host " Checkpoint Cleanup Analyzer" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

$root = "c:\Users\HP\Projects\EDI_Multiple_Sclerosis_detector"

Write-Host "Analyzing checkpoint folders..." -ForegroundColor Yellow
Write-Host ""

# Find all .pth files with duplicates
$allCheckpoints = Get-ChildItem $root -Filter "*.pth" -Recurse -File -EA SilentlyContinue

# Group by name to find duplicates
$grouped = $allCheckpoints | Group-Object Name | Where-Object { $_.Count -gt 1 }

Write-Host "DUPLICATE CHECKPOINT FILES:" -ForegroundColor Yellow
Write-Host ""

$totalDuplicateSize = 0
foreach ($group in $grouped) {
    $files = $group.Group | Sort-Object LastWriteTime -Descending
    $newestFile = $files[0]
    $olderFiles = $files[1..($files.Count-1)]
    
    Write-Host "File: $($group.Name) (found $($group.Count) copies)" -ForegroundColor White
    Write-Host "  KEEP (newest): $($newestFile.Directory.Name)\$($newestFile.Name) - $(Get-Date $newestFile.LastWriteTime -Format 'yyyy-MM-dd')" -ForegroundColor Green
    
    foreach ($old in $olderFiles) {
        $sizeMB = [math]::Round($old.Length/1MB, 1)
        $totalDuplicateSize += $old.Length
        Write-Host "  DELETE (older): $($old.Directory.Name)\$($old.Name) - $(Get-Date $old.LastWriteTime -Format 'yyyy-MM-dd') ($sizeMB MB)" -ForegroundColor Red
    }
    Write-Host ""
}

$totalDuplicateGB = [math]::Round($totalDuplicateSize/1GB, 2)
Write-Host "Total space from duplicates: $totalDuplicateGB GB" -ForegroundColor Cyan
Write-Host ""

# Show breakdown by directory
Write-Host "CHECKPOINT FOLDERS:" -ForegroundColor Yellow
$checkpointDirs = @("checkpoints", "segmentation", "deployment", "mae_pretraining", ".resume_checkpoints_optimal", ".resume_checkpoints_frozen_20251218_135038")

foreach ($dir in $checkpointDirs) {
    $fullPath = Join-Path $root $dir
    if (Test-Path $fullPath) {
        $files = Get-ChildItem $fullPath -Filter "*.pth" -File -EA SilentlyContinue
        if ($files) {
            $dirSize = ($files | Measure-Object -Property Length -Sum).Sum
            $sizeMB = [math]::Round($dirSize/1MB, 1)
            Write-Host "  $dir : $($files.Count) files ($sizeMB MB)" -ForegroundColor White
        }
    }
}

Write-Host ""
Write-Host "RECOMMENDATIONS:" -ForegroundColor Yellow
Write-Host "1. Keep your current working model (Jan 2026 checkpoint)" -ForegroundColor White
Write-Host "2. Keep deployment/model.pth if you use it" -ForegroundColor White  
Write-Host "3. Consider deleting intermediate training checkpoints" -ForegroundColor White
Write-Host "4. Archive the .resume_checkpoints_frozen_20251218_135038 folder after confirming training is complete" -ForegroundColor White
Write-Host ""
