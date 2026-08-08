# Quick Start: GPU Training

## Activate GPU Environment

```powershell
cd C:\Users\Rivan\Projects\Neuroscan
.\venv_gpu\Scripts\Activate.ps1
```

You should see `(venv_gpu)` in your terminal.

## Run Training

### 1. Test with 1 Epoch (5 min)
```bash
python train_baseline_brats_simple.py --epochs 1 --batch_size 8
```

### 2. Full Baseline (50 epochs, ~2-3 hours)
```bash
python train_baseline_brats_simple.py --epochs 50 --batch_size 8
```

### 3. NeuroScan Frozen Architecture (coming soon)
```bash
python train_baseline_brats.py --epochs 50 --batch_size 8
```

## Monitor Training

Results saved to:
```
checkpoints/brats_baseline/
├── best_model.pth           # Best checkpoint
├── checkpoint_epoch_*.pth   # All checkpoints
└── history.json             # Training curves (loss, dice)
```

## Expected Performance

| Setting | Time/Epoch | Notes |
|---------|-----------|-------|
| CPU (batch=2) | ~20 min | Previous baseline |
| GPU (batch=8) | ~2-3 min | **10x faster** |
| 50 epochs GPU | ~1.5-2.5 hrs | Full training |

## Troubleshooting

### "Module not found: dataset"
Make sure you're in the project root:
```bash
cd C:\Users\Rivan\Projects\Neuroscan
```

### "CUDA not available"
Check GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If False:
- Check NVIDIA drivers: `nvidia-smi`
- Verify GPU detected: `python -c "import torch; print(torch.cuda.get_device_name(0))"`

### "Out of memory"
Reduce batch size:
```bash
python train_baseline_brats_simple.py --batch_size 4  # Instead of 8
```

## Deactivate Environment

```bash
deactivate
```

---

**Ready to train!** 🚀
