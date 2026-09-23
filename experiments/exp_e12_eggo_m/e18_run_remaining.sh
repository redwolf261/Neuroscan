#!/bin/bash
set -e
PY="c:/Users/Rivan/Projects/Neuroscan/venv_gpu/Scripts/python.exe"
cd "c:/Users/Rivan/Projects/Neuroscan/experiments/exp_e12_eggo_m"

# Remaining work after the GPU-OOM incident (caused by an earlier,
# incorrectly-parallel freeze_bn watcher script -- see PHASE_E18 doc's
# note on this): freeze_encoder (rerun from scratch, prior attempt
# crashed at epoch 27/30 due to the OOM), freeze_decoder, freeze_seg_head,
# lambda_zero (never started), and freeze_bn (uses the CORRECTED
# e18_ablation_train.py with the 1-epoch-warmup fix, not the original
# buggy momentum=0-at-init version). All run strictly sequentially, one
# process at a time, to avoid repeating the GPU contention crash.
for abl in freeze_encoder freeze_decoder freeze_seg_head lambda_zero freeze_bn; do
  echo "=========================================="
  echo "Starting ablation: $abl"
  echo "=========================================="
  "$PY" e18_ablation_train.py --ablation "$abl" --epochs 30 --seed 0 --num_workers 4 --run_name "e18_${abl}_seed0"
  echo "[E18] ablation=$abl DONE"
done
echo "=== E18 ALL REMAINING ABLATIONS COMPLETE ==="
