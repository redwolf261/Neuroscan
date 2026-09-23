#!/bin/bash
set -e
cd /c/Users/Rivan/Projects/Neuroscan
echo "=== E112 3-seed campaign starting at $(date) ==="
for seed in 0 1 2; do
  echo "=== Launching seed $seed at $(date) ==="
  ./venv_gpu/Scripts/python.exe experiments/exp_e12_eggo_m/e112/train_e112_d4d8_a96_combined.py \
    --epochs 30 --run_name D4D8_A96_seed${seed} --seed ${seed}
  echo "=== Seed $seed complete at $(date) ==="
done
echo "=== E112 3-seed campaign ALL DONE at $(date) ==="
