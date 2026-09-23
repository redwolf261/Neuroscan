@echo off
REM Disable Python 3.13 user packages and run training
set PYTHONNOUSERSITE=1
set PYTHONUNBUFFERED=1
set PATH=C:\ProgramData\miniconda3;C:\ProgramData\miniconda3\Scripts;C:\ProgramData\miniconda3\Library\bin;%PATH%

echo Starting training with clean environment...
echo.
python final_model.py
pause
