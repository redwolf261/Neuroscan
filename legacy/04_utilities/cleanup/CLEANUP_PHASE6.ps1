# Phase 6 Cleanup - MS Cross Validation
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Phase 6: MS Cross Validation Cleanup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$root = "c:\Users\HP\Projects\EDI_Multiple_Sclerosis_detector"
$msCVPath = Join-Path $root "MS cross validation"

if (Test-Path $msCVPath) {
    Write-Host "Analyzing MS cross validation folder..." -ForegroundColor Yellow
    Write-Host ""
    
    # Get subdirectories
    $subdirs = Get-ChildItem $msCVPath -Directory -EA SilentlyContinue
    
    if ($subdirs) {
        Write-Host "Cross validation runs found:" -ForegroundColor White
        $runsSummary = @()
        
        foreach ($dir in $subdirs) {
            $files = Get-ChildItem $dir.FullName -Recurse -File -EA SilentlyContinue
            if ($files) {
                $dirSize = ($files | Measure-Object -Property Length -Sum).Sum
                $dirSizeMB = [math]::Round($dirSize/1MB, 1)
                $newestFile = ($files | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime
                
                $runsSummary += [PSCustomObject]@{
                    Name = $dir.Name
                    Files = $files.Count
                    SizeMB = $dirSizeMB
                    LastModified = $newestFile
                }
            }
        }
        
        $runsSummary | Sort-Object LastModified -Descending | ForEach-Object {
            $date = Get-Date $_.LastModified -Format "yyyy-MM-dd"
            Write-Host "  $($_.Name) - $date ($($_.SizeMB) MB)" -ForegroundColor White
        }
        
        # Calculate old runs (before Dec 2025)
        $cutoffDate = Get-Date "2025-12-01"
        $oldRuns = $runsSummary | Where-Object { $_.LastModified -lt $cutoffDate }
        
        if ($oldRuns) {
            $oldSize = ($oldRuns | Measure-Object -Property SizeMB -Sum).Sum
            $oldSizeGB = [math]::Round($oldSize/1024, 2)
            
            Write-Host ""
            Write-Host "OLD CROSS VALIDATION RUNS (before Dec 2025):" -ForegroundColor Yellow
            Write-Host "  Count: $($oldRuns.Count) runs" -ForegroundColor White
            Write-Host "  Size: $oldSizeGB GB" -ForegroundColor White
            Write-Host ""
            
            if ($oldSizeGB -gt 0.5) {
                Write-Host "These are older experiment results." -ForegroundColor Cyan
                Write-Host "Recent runs (Dec 2025+) will be kept." -ForegroundColor Green
                Write-Host ""
                
                $deleteOld = Read-Host "Delete old CV runs from before December 2025? (yes/no)"
                
                if ($deleteOld -eq "yes") {
                    Write-Host ""
                    Write-Host "Deleting old cross validation runs..." -ForegroundColor Yellow
                    foreach ($run in $oldRuns) {
                        $runPath = Join-Path $msCVPath $run.Name
                        try {
                            Remove-Item $runPath -Recurse -Force
                            Write-Host "  Deleted: $($run.Name)" -ForegroundColor Green
                        }
                        catch {
                            Write-Host "  Failed: $($run.Name)" -ForegroundColor Red
                        }
                    }
                }
            }
        } else {
            Write-Host ""
            Write-Host "All cross validation runs are recent (Dec 2025+)" -ForegroundColor Green
            Write-Host "No cleanup recommended." -ForegroundColor Green
        }
        
        # Check for large checkpoint files in CV folder
        Write-Host ""
        Write-Host "Checking for large checkpoint files in CV folder..." -ForegroundColor Yellow
        $cvCheckpoints = Get-ChildItem $msCVPath -Filter "*.pth" -Recurse -File -EA SilentlyContinue
        
        if ($cvCheckpoints) {
            $cvCheckpointSize = ($cvCheckpoints | Measure-Object -Property Length -Sum).Sum
            $cvCheckpointSizeGB = [math]::Round($cvCheckpointSize/1GB, 2)
            
            Write-Host "  Found $($cvCheckpoints.Count) checkpoint files ($cvCheckpointSizeGB GB)" -ForegroundColor White
            
            if ($cvCheckpointSizeGB -gt 1) {
                Write-Host ""
                Write-Host "Large checkpoint files in CV folder:" -ForegroundColor Yellow
                $largeCheckpoints = $cvCheckpoints | Where-Object { $_.Length -gt 100MB } | Sort-Object Length -Descending | Select-Object -First 10
                
                foreach ($ckpt in $largeCheckpoints) {
                    $sizeMB = [math]::Round($ckpt.Length/1MB, 1)
                    $relPath = $ckpt.FullName.Replace($msCVPath + "\", "")
                    Write-Host "  $relPath ($sizeMB MB)" -ForegroundColor White
                }
                
                Write-Host ""
                Write-Host "CV checkpoints are usually not needed after experiments complete." -ForegroundColor Cyan
                $deleteCVCheckpoints = Read-Host "Delete checkpoint files from CV folder? (yes/no)"
                
                if ($deleteCVCheckpoints -eq "yes") {
                    Write-Host ""
                    foreach ($ckpt in $cvCheckpoints) {
                        try {
                            Remove-Item $ckpt.FullName -Force
                            Write-Host "  Deleted: $($ckpt.Name)" -ForegroundColor Green
                        }
                        catch {
                            Write-Host "  Failed: $($ckpt.Name)" -ForegroundColor Red
                        }
                    }
                }
            }
        }
    }
} else {
    Write-Host "MS cross validation folder not found." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
$newSize = (Get-ChildItem $root -Recurse -File -EA SilentlyContinue | Measure-Object -Property Length -Sum).Sum
$newSizeGB = [math]::Round($newSize/1GB, 2)
Write-Host "Current project size: $newSizeGB GB" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
