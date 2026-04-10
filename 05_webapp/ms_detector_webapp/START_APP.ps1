# MS Lesion Detection Web Application
# Quick Start Script for Windows PowerShell

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   MS Lesion Detection Web Application" -ForegroundColor Green
Write-Host "   Model: HybridMiniSwin2.5D with Adaptive Slice Selection" -ForegroundColor Green
Write-Host "   Performance: 82.31% Dice Score (Epoch 12)" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if running from correct directory
if (!(Test-Path "backend\app.py")) {
    Write-Host "ERROR: Please run this script from the ms_detector_webapp directory" -ForegroundColor Red
    Write-Host "Current directory: $PWD" -ForegroundColor Yellow
    exit 1
}

Write-Host "[1/2] Starting Backend API Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\backend'; .\venv\Scripts\python.exe app.py" -WindowStyle Normal

Start-Sleep -Seconds 5

Write-Host "[2/2] Starting Frontend Development Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\frontend'; npm start" -WindowStyle Normal

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "✅ Application Starting!" -ForegroundColor Green
Write-Host "" 
Write-Host "Backend API:  " -NoNewline
Write-Host "http://localhost:5000" -ForegroundColor Blue
Write-Host "Frontend UI:  " -NoNewline
Write-Host "http://localhost:3000" -ForegroundColor Blue
Write-Host ""
Write-Host "Close the PowerShell windows to stop the servers" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Cyan
