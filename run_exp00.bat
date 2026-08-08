@echo off
REM Launch Experiment 00 baseline training
REM Usage: run_exp00.bat [epochs] [batch_size]

cd /d "%~dp0"

REM Activate venv
call venv_gpu\Scripts\activate.bat

REM Run training from project root
cd experiments\exp00_neuroscan_baseline
python train.py --epochs %1 --batch_size %2

pause
