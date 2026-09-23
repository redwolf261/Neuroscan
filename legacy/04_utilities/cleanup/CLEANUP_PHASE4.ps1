# Phase 4 Cleanup - Old Checkpoints Analysis
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Phase 4: Checkpoint Folder Analysis" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$root = "c:\Users\HP\Projects\EDI_Multiple_Sclerosis_detector"
$checkpointsPath = Join-Path $root "checkpoints"

Write-Host "Analyzing checkpoints folder..." -ForegroundColor Yellow
Write-Host ""

# Get all subdirectories in checkpoints
$checkpointDirs = Get-ChildItem $checkpointsPath -Directory -EA SilentlyContinue

if ($checkpointDirs) {
    Write-Host "Found checkpoint subdirectories:" -ForegroundColor White
    $dirSummary = @()
    
    foreach ($dir in $checkpointDirs) {
        $files = Get-ChildItem $dir.FullName -File -EA SilentlyContinue
        if ($files) {
            $dirSize = ($files | Measure-Object -Property Length -Sum).Sum
            $dirSizeMB = [math]::Round($dirSize/1MB, 1)
            
            $dirSummary += [PSCustomObject]@{
                Name = $dir.Name
                Files = $files.Count
                SizeMB = $dirSizeMB
                Path = $dir.FullName
            }
            
            Write-Host "  $($dir.Name): $($files.Count) files ($dirSizeMB MB)" -ForegroundColor White
        }
    }
    
    # Show oldest files
    Write-Host ""
    Write-Host "Oldest checkpoint files:" -ForegroundColor Yellow
    $allFiles = Get-ChildItem $checkpointsPath -File -Recurse -EA SilentlyContinue | Sort-Object LastWriteTime
    $oldFiles = $allFiles | Select-Object -First 10
    
    foreach ($file in $oldFiles) {
        $sizeMB = [math]::Round($file.Length/1MB, 1)
        $relPath = $file.FullName.Replace($checkpointsPath + "\", "")
        $date = Get-Date $file.LastWriteTime -Format "yyyy-MM-dd"
        Write-Host "  $date - $relPath ($sizeMB MB)" -ForegroundColor White
    }
    
    # Calculate total size of old checkpoints (before Nov 2025)
    $cutoffDate = Get-Date "2025-11-01"
    $oldCheckpoints = $allFiles | Where-Object { $_.LastWriteTime -lt $cutoffDate }
    
    if ($oldCheckpoints) {
        $oldSize = ($oldCheckpoints | Measure-Object -Property Length -Sum).Sum
        $oldSizeGB = [math]::Round($oldSize/1GB, 2)
        
        Write-Host ""
        Write-Host "OLD CHECKPOINTS (before Nov 2025):" -ForegroundColor Yellow
        Write-Host "  Count: $($oldCheckpoints.Count) files" -ForegroundColor White
        Write-Host "  Size: $oldSizeGB GB" -ForegroundColor White
        Write-Host ""
        
        if ($oldSizeGB -gt 0.5) {
            Write-Host "These are experiment checkpoints from October 2025" -ForegroundColor Cyan
            Write-Host "Your current best models (Nov 2025 - Jan 2026) are preserved" -ForegroundColor Green
            Write-Host ""
            
            $deleteOld = Read-Host "Delete old checkpoints from before November 2025? (yes/no)"
            
            if ($deleteOld -eq "yes") {
                Write-Host ""
                Write-Host "Deleting old checkpoints..." -ForegroundColor Yellow
                foreach ($file in $oldCheckpoints) {
                    try {
                        Remove-Item $file.FullName -Force
                        Write-Host "  Deleted: $($file.Name)" -ForegroundColor Green
                    }
                    catch {
                        Write-Host "  Failed: $($file.Name)" -ForegroundColor Red
                    }
                }
                
                # Clean up empty directories
                $emptyDirs = Get-ChildItem $checkpointsPath -Directory -Recurse -EA SilentlyContinue | 
                    Where-Object { -not (Get-ChildItem $_.FullName -File -EA SilentlyContinue) }
                
                foreach ($emptyDir in $emptyDirs) {
                    Remove-Item $emptyDir.FullName -Force -EA SilentlyContinue
                }
            }
        }
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
$newSize = (Get-ChildItem $root -Recurse -File -EA SilentlyContinue | Measure-Object -Property Length -Sum).Sum
$newSizeGB = [math]::Round($newSize/1GB, 2)
Write-Host "Current project size: $newSizeGB GB" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
