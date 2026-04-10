# Start Both Backend and Frontend
# This script launches both servers in separate windows

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Starting MS Detector (Full Stack)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get current directory
$currentDir = Get-Location

# Start backend in new window
Write-Host "Starting backend server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-File", "$currentDir\start_backend.ps1"

# Wait a bit for backend to start
Start-Sleep -Seconds 3

# Start frontend in new window
Write-Host "Starting frontend server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-File", "$currentDir\start_frontend.ps1"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Servers Starting!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Two terminal windows have been opened:" -ForegroundColor White
Write-Host "  1. Backend (Flask API)" -ForegroundColor Cyan
Write-Host "  2. Frontend (React App)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Access the application at:" -ForegroundColor White
Write-Host "  Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host "  Backend API: http://localhost:5000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Close the terminal windows to stop the servers" -ForegroundColor Yellow
