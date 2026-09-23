#!/bin/bash
cd /c/Users/Rivan/Projects/Neuroscan
for RUN in E131_v14_seed0 E131_v5control_seed0; do
  echo "=== FULL EVAL: $RUN ==="
  python -u experiments/exp_e12_eggo_m/e130/eval_e130_full.py \
    --ckpt experiments/exp_e12_eggo_m/e131/runs/$RUN/checkpoints/best.pth \
    --tag $RUN
done
echo "=== E131 EVAL COMPLETE ==="
