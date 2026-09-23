#!/bin/bash
cd /c/Users/Rivan/Projects/Neuroscan
E=experiments/exp_e12_eggo_m/e130
echo "=== REPLICATION 1/2: v14 ==="
python -u experiments/exp_e12_eggo_m/e133/run_e133_replicate.py \
  --ckpt experiments/exp_e12_eggo_m/e131/runs/E131_v14_seed0/checkpoints/best.pth \
  --eval_json $E/E130_full_eval_E131_v14_seed0_per_subject.json --arch v14 --tag v14
echo "=== REPLICATION 2/2: E130 baseline ==="
python -u experiments/exp_e12_eggo_m/e133/run_e133_replicate.py \
  --ckpt $E/runs/E130_baseline_seed0/checkpoints/best.pth \
  --eval_json $E/E130_full_eval_E130_baseline_seed0_ep32_per_subject.json --arch v5 --tag E130base
echo "=== REPLICATION COMPLETE ==="
