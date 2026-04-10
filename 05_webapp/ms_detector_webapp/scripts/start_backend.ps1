# Start Backend Server
# Run this script to start only the backend API

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Starting MS Detector Backend" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to backend directory
Set-Location backend

# Check if virtual environment exists
if (!(Test-Path ".\venv")) {
    Write-Host "✗ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run setup.ps1 first" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Check if model exists
if (!(Test-Path ".\models\model_deployable.pth")) {
    Write-Host "⚠ Warning: Model file not found!" -ForegroundColor Yellow
    Write-Host "Expected location: backend\models\model_deployable.pth" -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") {
        exit 1
    }
}

# Start Flask server
Write-Host ""
Write-Host "Starting Flask server..." -ForegroundColor Green
Write-Host "Backend will be available at: http://localhost:5000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

python app.py
