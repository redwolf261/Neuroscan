#!/bin/bash
cd /c/Users/Rivan/Projects/Neuroscan
echo "=== ARM 1/2: v13 Rank-Adaptive Pooling ==="
python -u experiments/exp_e12_eggo_m/e131/train_e131_rap.py \
  --epochs 50 --batch_size 1 --accum_steps 2 --bn_momentum 0.01 \
  --arch v13 --lam_init 0.05 --run_name E131_v13_seed0 --fast_val 1
echo "=== ARM 2/2: v5 control (same BN fix, same protocol) ==="
python -u experiments/exp_e12_eggo_m/e131/train_e131_rap.py \
  --epochs 50 --batch_size 1 --accum_steps 2 --bn_momentum 0.01 \
  --arch v5 --run_name E131_v5control_seed0 --fast_val 1
echo "=== E131 BOTH ARMS COMPLETE ==="
