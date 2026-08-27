# GPU Setup Guide for BraTS Training

## Current Status

✅ **CPU-based baseline training completed successfully**
- Simple UNet: 54.5% validation Dice on BraTS
- Data pipeline fully verified
- Checkpoint system working

❌ **GPU support needed** — Python 3.14 too new for PyTorch CUDA wheels

---

## Why GPU?

| Aspect | CPU | GPU |
|--------|-----|-----|
| **Time per Epoch** | ~20 minutes | ~2-3 minutes |
| **50 Epochs** | 16.7 hours | 1.7-2.5 hours |
| **Speed Multiplier** | 1x | **8-10x faster** |

---

## Solution: Python 3.11 Virtual Environment

PyTorch CUDA wheels support up to Python 3.12. We need to:

1. **Install Python 3.11** (separate from Python 3.14)
2. **Create venv** with Python 3.11
3. **Install PyTorch CUDA** in that venv
4. **Run training** from the venv

---

## Step-by-Step Setup

### Step 1: Install Python 3.11

**Windows:**
- Download from: https://www.python.org/downloads/release/python-3111/
- Choose "Windows installer (64-bit)"
- During installation:
  - ✅ Check "Add Python to PATH"
  - ✅ Check "Install pip"
  - Choose installation directory (e.g., `C:\Python311`)

**Or via Windows Store:**
```powershell
winget install Python.Python.3.11
```

### Step 2: Verify Installation

```powershell
python3.11 --version
# Should output: Python 3.11.x
```

### Step 3: Create Virtual Environment

```powershell
# From project root
cd C:\Users\Rivan\Projects\Neuroscan

# Create venv with Python 3.11
python3.11 -m venv venv_gpu

# Activate
.\venv_gpu\Scripts\Activate.ps1

# Verify
python --version  # Should show Python 3.11.x
```

### Step 4: Install PyTorch with CUDA

```powershell
# Activate venv first
.\venv_gpu\Scripts\Activate.ps1

# Install PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install monai nibabel scipy scikit-image pyyaml tqdm tensorboard numpy

# Verify CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Step 5: Run Training with GPU

```powershell
# Make sure venv is activated
.\venv_gpu\Scripts\Activate.ps1

# Run training (no --device needed, defaults to cuda if available)
python train_baseline_brats_simple.py --epochs 10 --batch_size 8
```

---

## Troubleshooting

### CUDA not available after install?

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

If False:
- Check NVIDIA GPU drivers are installed
- Check CUDA Toolkit installed
- Restart Python

### Which GPU do I have?

```powershell
# Windows CMD/PowerShell
nvidia-smi

# Should show GPU name, memory, driver version
```

### Need GPU drivers?

Download from: https://www.nvidia.com/Download/driverDetails.aspx

---

## Expected Performance After GPU Setup

### Current (CPU, batch_size=2)
- Time per epoch: ~20 minutes
- Training speed: ~2 batches/sec

### Expected (GPU, batch_size=8)
- Time per epoch: ~2-3 minutes
- Training speed: ~20-30 batches/sec
- **10x speedup**

### Full Training Timeline
- **10 epochs**: 20-30 minutes (vs 3 hours on CPU)
- **50 epochs**: 1.5-2.5 hours (vs 16 hours on CPU)

---

## Recommended Next Steps

1. **Install Python 3.11** (see Step 1 above)
2. **Create GPU venv** (Steps 2-4)
3. **Run full baseline** (50 epochs) to establish convergence curve
4. **Then integrate NeuroScan architecture** (frozen model)
5. **Finally: Optimizer integration** (the core research contribution)

---

## Files Ready to Use

Once GPU is set up, no code changes needed:
- `train_baseline_brats_simple.py` — Just change `--device cuda` or use default
- `configs/brats.yaml` — Already configured for training
- `dataset/brats_dataset.py` — Data pipeline ready

---

## Current Git State

```
Status: ✅ Clean — ready to commit
Files created:
  - dataset/brats_dataset.py
  - preprocessing/brats_preprocess.py
  - configs/brats.yaml
  - train_baseline_brats_simple.py
  - train_baseline_brats.py
  - checkpoints/brats_baseline/best_model.pth (54.5% Dice baseline)
```

Consider committing once GPU setup confirmed to work.

---

**Questions?** Once you install Python 3.11, come back and I'll help you set up the venv and verify GPU works.
