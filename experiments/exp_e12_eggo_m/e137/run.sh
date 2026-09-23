#!/bin/bash
cd /c/Users/Rivan/Projects/Neuroscan
echo "=== ARM 1/2: v16 Enhancement-Conditioned Readout ==="
python -u experiments/exp_e12_eggo_m/e137/train_e137_ecr.py \
  --epochs 50 --batch_size 1 --accum_steps 2 --bn_momentum 0.01 \
  --arch v16 --run_name E137_v16_seed0 --fast_val 1
echo "=== E137 v16 COMPLETE ==="
