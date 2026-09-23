#!/bin/bash
cd /c/Users/Rivan/Projects/Neuroscan
echo "=== ARM 1/2: v5 AMP-matched CONTROL ==="
python experiments/exp_e12_eggo_m/e128/train_e128_capacity_gate.py \
  --epochs 30 --arch v5 --amp 1 --seed 0 --run_name Control_v5amp_seed0
echo "=== ARM 2/2: v12 capacity mutation ==="
python experiments/exp_e12_eggo_m/e128/train_e128_capacity_gate.py \
  --epochs 30 --arch v12 --amp 1 --seed 0 --run_name Capacity_v12_seed0
echo "=== E128 BOTH ARMS COMPLETE ==="
