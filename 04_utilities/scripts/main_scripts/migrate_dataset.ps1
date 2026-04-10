# Dataset Migration using PowerShell (More reliable for large folders)
# This uses robocopy which is optimized for large file operations

$source = "G:\My Drive\Dataset"
$destination = "C:\Users\HP\EDI\Dataset"
$logFile = "C:\Users\HP\EDI\dataset_copy_log.txt"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "DATASET MIGRATION - PowerShell Method" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Source: $source" -ForegroundColor White
Write-Host "Destination: $destination" -ForegroundColor White
Write-Host "Log: $logFile`n" -ForegroundColor Gray

if (-not (Test-Path $source)) {
    Write-Host "ERROR: Source not found: $source" -ForegroundColor Red
    exit 1
}

# Calculate source size
Write-Host "Calculating dataset size..." -ForegroundColor Yellow
$size = (Get-ChildItem -Path $source -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host "Dataset size: $([math]::Round($size, 2)) GB" -ForegroundColor Green
Write-Host "Estimated time: $([math]::Ceiling($size * 2))-$([math]::Ceiling($size * 3)) minutes`n" -ForegroundColor Gray

$response = Read-Host "Proceed with copy? (yes/no)"
if ($response -ne "yes") {
    Write-Host "`nCancelled by user." -ForegroundColor Yellow
    exit 0
}

Write-Host "`nStarting copy using robocopy..." -ForegroundColor Cyan
Write-Host "This will show progress and is safe to interrupt/resume`n" -ForegroundColor Gray

# Use robocopy for reliable large file copying
# /MIR = Mirror (copy all, remove if deleted from source)
# /Z = Restartable mode (can resume if interrupted)
# /R:3 = Retry 3 times on failure
# /W:5 = Wait 5 seconds between retries
# /MT:8 = Use 8 threads for faster copying
# /V = Verbose output
# /NP = No percentage (cleaner output)
# /LOG = Log file

robocopy $source $destination /E /Z /R:3 /W:5 /MT:8 /V /NP /TEE /LOG:$logFile

$exitCode = $LASTEXITCODE

# Robocopy exit codes:
# 0 = No files copied
# 1 = Files copied successfully
# 2 = Extra files/folders detected (still success)
# 3 = Files copied + extra files (still success)
# 4+ = Errors occurred

if ($exitCode -lt 4) {
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "COPY COMPLETED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "========================================`n" -ForegroundColor Green
    
    # Verify
    Write-Host "Verifying copy..." -ForegroundColor Yellow
    $destSize = (Get-ChildItem -Path $destination -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1GB
    Write-Host "Source size: $([math]::Round($size, 2)) GB" -ForegroundColor White
    Write-Host "Destination size: $([math]::Round($destSize, 2)) GB`n" -ForegroundColor White
    
    if ([math]::Abs($size - $destSize) -lt 0.01) {
        Write-Host "✅ Verification PASSED - Sizes match!" -ForegroundColor Green
        Write-Host "`nDataset is now available at: $destination" -ForegroundColor Cyan
        Write-Host "`nNEXT STEPS:" -ForegroundColor Yellow
        Write-Host "1. Test training with local dataset" -ForegroundColor Gray
        Write-Host "2. After verification, delete from Google Drive to free $([math]::Round($size, 2)) GB" -ForegroundColor Gray
        Write-Host "3. Update final_model.py DATA_PATH to use local dataset`n" -ForegroundColor Gray
    } else {
        Write-Host "⚠️ WARNING: Size mismatch!" -ForegroundColor Red
        Write-Host "Please verify manually before deleting from Google Drive`n" -ForegroundColor Yellow
    }
    
    Write-Host "Log file: $logFile" -ForegroundColor Gray
} else {
    Write-Host "`n========================================" -ForegroundColor Red
    Write-Host "COPY FAILED - Exit Code: $exitCode" -ForegroundColor Red
    Write-Host "========================================`n" -ForegroundColor Red
    Write-Host "Check log file for errors: $logFile" -ForegroundColor Yellow
    Write-Host "Do NOT delete from Google Drive!`n" -ForegroundColor Red
}
