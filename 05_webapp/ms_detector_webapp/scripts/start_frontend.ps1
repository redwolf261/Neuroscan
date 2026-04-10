# Start Frontend Development Server
# Run this script to start only the frontend

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Starting MS Detector Frontend" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to frontend directory
Set-Location frontend

# Check if node_modules exists
if (!(Test-Path ".\node_modules")) {
    Write-Host "✗ Dependencies not installed!" -ForegroundColor Red
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    npm install
}

# Start development server
Write-Host ""
Write-Host "Starting React development server..." -ForegroundColor Green
Write-Host "Frontend will be available at: http://localhost:3000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

npm start
