@echo off
echo ========================================
echo Stopping current validation process...
echo ========================================
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *cross_dataset_validation*" 2>nul
timeout /t 2 /nobreak >nul

echo.
echo ========================================
echo Starting validation with GPU...
echo ========================================
cd /d C:\Users\HP\EDI
python research\cross_dataset_validation_lgg.py

pause
