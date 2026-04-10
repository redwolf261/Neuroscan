# Phase 3 Cleanup - Python Cache and Web App
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Phase 3: Safe Cleanup - Cache & Web" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$root = "c:\Users\HP\Projects\EDI_Multiple_Sclerosis_detector"

# Step 1: Python Cache Cleanup (100% safe)
Write-Host "Step 1: Cleaning Python cache files..." -ForegroundColor Yellow
$cacheItems = Get-ChildItem $root -Include "__pycache__","*.pyc","*.pyo" -Recurse -Force -EA SilentlyContinue
$cacheSize = 0
$cacheCount = 0

foreach ($item in $cacheItems) {
    if ($item.PSIsContainer) {
        $size = (Get-ChildItem $item.FullName -Recurse -File -EA SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $cacheSize += $size
    } else {
        $cacheSize += $item.Length
    }
    $cacheCount++
}

$cacheSizeMB = [math]::Round($cacheSize/1MB, 1)
Write-Host "  Found $cacheCount cache items ($cacheSizeMB MB)" -ForegroundColor White

if ($cacheCount -gt 0) {
    Write-Host "  Deleting Python cache..." -ForegroundColor Yellow
    foreach ($item in $cacheItems) {
        Remove-Item $item -Recurse -Force -EA SilentlyContinue
    }
    Write-Host "  Deleted Python cache" -ForegroundColor Green
}

# Step 2: Check ms_detector_webapp
Write-Host ""
Write-Host "Step 2: Analyzing ms_detector_webapp..." -ForegroundColor Yellow
$webappPath = Join-Path $root "ms_detector_webapp"

if (Test-Path $webappPath) {
    $nodeModules = Join-Path $webappPath "node_modules"
    $buildDir = Join-Path $webappPath "build"
    $distDir = Join-Path $webappPath "dist"
    
    $webappSize = 0
    $canDelete = @()
    
    if (Test-Path $nodeModules) {
        $nmSize = (Get-ChildItem $nodeModules -Recurse -File -EA SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $nmSizeGB = [math]::Round($nmSize/1GB, 2)
        Write-Host "  Found node_modules: $nmSizeGB GB" -ForegroundColor White
        Write-Host "    (Can be regenerated with: npm install)" -ForegroundColor Cyan
        $canDelete += @{Path=$nodeModules; Size=$nmSize; Name="node_modules"}
        $webappSize += $nmSize
    }
    
    if (Test-Path $buildDir) {
        $bSize = (Get-ChildItem $buildDir -Recurse -File -EA SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $bSizeMB = [math]::Round($bSize/1MB, 1)
        if ($bSizeMB -gt 10) {
            Write-Host "  Found build/: $bSizeMB MB (build artifacts)" -ForegroundColor White
            $canDelete += @{Path=$buildDir; Size=$bSize; Name="build"}
            $webappSize += $bSize
        }
    }
    
    if (Test-Path $distDir) {
        $dSize = (Get-ChildItem $distDir -Recurse -File -EA SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $dSizeMB = [math]::Round($dSize/1MB, 1)
        if ($dSizeMB -gt 10) {
            Write-Host "  Found dist/: $dSizeMB MB (build artifacts)" -ForegroundColor White
            $canDelete += @{Path=$distDir; Size=$dSize; Name="dist"}
            $webappSize += $dSize
        }
    }
    
    if ($canDelete.Count -gt 0) {
        $webappSizeGB = [math]::Round($webappSize/1GB, 2)
        Write-Host ""
        Write-Host "  Total deletable in webapp: $webappSizeGB GB" -ForegroundColor Cyan
        Write-Host "  These can be regenerated from package.json" -ForegroundColor Green
        Write-Host ""
        $webConfirm = Read-Host "  Delete webapp build artifacts? (yes/no)"
        
        if ($webConfirm -eq "yes") {
            foreach ($item in $canDelete) {
                try {
                    Remove-Item $item.Path -Recurse -Force
                    Write-Host "    Deleted: $($item.Name)" -ForegroundColor Green
                }
                catch {
                    Write-Host "    Failed to delete: $($item.Name)" -ForegroundColor Red
                }
            }
        }
    } else {
        Write-Host "  No large build artifacts found" -ForegroundColor Green
    }
}

# Step 3: Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
$newSize = (Get-ChildItem $root -Recurse -File -EA SilentlyContinue | Measure-Object -Property Length -Sum).Sum
$newSizeGB = [math]::Round($newSize/1GB, 2)
Write-Host "Current project size: $newSizeGB GB" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
