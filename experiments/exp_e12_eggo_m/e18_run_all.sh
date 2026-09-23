#!/bin/bash
set -e
PY="c:/Users/Rivan/Projects/Neuroscan/venv_gpu/Scripts/python.exe"
cd "c:/Users/Rivan/Projects/Neuroscan/experiments/exp_e12_eggo_m"

for abl in none freeze_bn freeze_encoder freeze_decoder freeze_seg_head lambda_zero; do
  echo "=========================================="
  echo "Starting ablation: $abl"
  echo "=========================================="
  "$PY" e18_ablation_train.py --ablation "$abl" --epochs 30 --seed 0 --num_workers 4 --run_name "e18_${abl}_seed0"
  echo "[E18] ablation=$abl DONE"
done
echo "=== E18 ALL ABLATIONS COMPLETE ==="
