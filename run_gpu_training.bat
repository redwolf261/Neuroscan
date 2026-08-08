@echo off
REM GPU Training Launcher for BraTS
REM This script activates the GPU venv and runs training

cd /d "%~dp0"

REM Activate venv
call venv_gpu\Scripts\activate.bat

REM Run training
echo Training on GPU...
python train_baseline_brats_simple.py --epochs 1 --batch_size 8

pause
