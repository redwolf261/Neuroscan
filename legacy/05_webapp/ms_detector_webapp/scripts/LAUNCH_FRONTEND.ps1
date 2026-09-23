# Launch Frontend Server
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host "  Starting MS Detector Frontend" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Set-Location "C:\Users\HP\EDI\ms_detector_webapp\frontend"

Write-Host "`nStarting React development server..." -ForegroundColor Green
Write-Host "This will open in your browser automatically..." -ForegroundColor Yellow
Write-Host "`nPress Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

npm start
