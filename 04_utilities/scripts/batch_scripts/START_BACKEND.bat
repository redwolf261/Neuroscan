@echo off
echo ======================================================================
echo   Starting MS Lesion Detection Backend
echo   This will take 30-60 seconds to load the 367MB model...
echo ======================================================================
echo.

cd C:\Users\HP\EDI\ms_detector_webapp\backend
C:\Users\HP\EDI\ms_detector_webapp\backend\venv\Scripts\python.exe app.py

pause
