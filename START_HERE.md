# 🔬 Start Here: NeuroScan Optimization Research

**Status**: Phase A in progress  
**Current Time**: 2026-07-29, ~20:45 UTC  
**Exp 00**: Training baseline (1 epoch test)

---

## You Are Here

You've transitioned from **infrastructure building** to **algorithm research**.

The data pipeline works. PyTorch is set up. NeuroScan compiles. Now you're answering research questions, not debugging tools.

---

## What's Running Right Now

**Experiment 00**: NeuroScan baseline on BraTS
- Location: `experiments/exp00_neuroscan_baseline/`
- Status: Training 1 epoch for verification
- ETA: ~30 min (CPU) to complete test run
- Next: Will run full 50 epochs when ready

---

## Your 6-Week Research Plan

| Week | Phase | Task | Status |
|------|-------|------|--------|
| 1 | A | Baseline: NeuroScan on BraTS | 🔄 In progress |
| 2 | B | Gradient diagnostics | ⏳ Next |
| 3 | C | Optimizer implementation | ⏳ Next |
| 4 | D | Ablation studies | ⏳ Next |
| 5 | E | Competitor comparisons | ⏳ Next |
| 6 | — | Analysis, writing, figures | ⏳ Next |

---

## What You Need to Know

### Key Files

| File | Read This For |
|------|---------------|
| `RESEARCH_MODE.md` | Quick overview of research mode (2 min) |
| `RESEARCH_PROGRAM.md` | Full program details (10 min) |
| `experiments/README.md` | How to run experiments (5 min) |
| `neuroscan_frozen.py` | Model code (frozen, no edits) |
| `experiments/exp00_neuroscan_baseline/train.py` | Training script for Phase A |

### Key Directories

```
Neuroscan/
├── neuroscan_frozen.py              # Model (frozen)
├── dataset/                         # Data loader
├── preprocessing/                   # Preprocessing
├── configs/                         # Configs
├── experiments/                     # All research experiments
│   ├── exp00_neuroscan_baseline/   # Phase A (baseline)
│   ├── exp01_diagnostics/          # Phase B (gradients)
│   ├── exp02_ema/                  # Phase C ablation 1
│   ├── exp03_alpha/                # Phase C ablation 2
│   ├── exp04_delta/                # Phase C ablation 3
│   ├── exp05_full_optimizer/       # Phase C full ABO
│   └── exp06_comparisons/          # Phase E (competitors)
└── venv_gpu/                        # Python 3.11 environment
```

---

## Right Now: Phase A

### What It Is
Train **frozen NeuroScan** on **BraTS** with **zero modifications**.

- Model: HybridMiniSwin2D5_CBAM
- Dataset: BraTS 2023 GLI (1,251 subjects)
- Modality: FLAIR only (single channel)
- Loss: HybridLoss (focal + evidential)
- Optimizer: AdamW + cosine annealing

### What It Answers
Can NeuroScan (trained on MS lesions) transfer to brain tumor segmentation?

### When It's Done
- Validation Dice should be > 30%
- Final Dice will be the **baseline** for all future experiments
- You'll use this to measure improvement from ABO optimizer

### How to Run
```powershell
cd C:\Users\Rivan\Projects\Neuroscan
.\venv_gpu\Scripts\Activate.ps1

cd experiments\exp00_neuroscan_baseline
python train.py --epochs 50
```

**On CPU**: ~25 hours for 50 epochs (train overnight)  
**On GPU** (when fixed): ~2 hours for 50 epochs

---

## Why This Matters

You're not just training models.

You're **building evidence** that:
1. Your data pipeline works (Exp 00)
2. Gradient conflicts exist (Exp 01)
3. ABO solves those conflicts (Exp 02-05)
4. ABO beats existing methods (Exp 06)

Each experiment **must answer one question**. Each question **must be answered with evidence**. That's research.

---

## What Success Looks Like

### This Week
- [ ] Exp 00 runs without errors
- [ ] Baseline validation Dice > 30%
- [ ] Results saved to `checkpoints/best.pth`
- [ ] Config saved to `config.yaml`
- [ ] Summary written to `README.md`

### This Month
- [ ] All 5 phases complete
- [ ] Clear evidence that ABO helps
- [ ] Comparisons against PCGrad/GradNorm
- [ ] Paper figures ready

### This Semester
- [ ] Thesis written
- [ ] Publication submitted
- [ ] Research complete

---

## Common Commands

### Activate environment
```powershell
.\venv_gpu\Scripts\Activate.ps1
```

### Run Exp 00 (baseline)
```bash
cd experiments\exp00_neuroscan_baseline
python train.py --epochs 50
```

### Check results
```bash
cat checkpoints\history.json  # See loss/Dice curves
ls checkpoints\best.pth       # Confirm checkpoint saved
```

### Compare experiments (Week 5)
```bash
python analyze_experiments.py
```

---

## Next Steps

**Today** (now):
- Exp 00 should complete in ~30 min (test run)
- Verify it works without errors

**Tomorrow**:
- Run full Exp 00 (50 epochs)
- Let it train overnight

**This week**:
- Extract final Dice
- Document baseline results
- Plan Phase B (gradient logging)

**Next week**:
- Phase B: Gradient diagnostics
- Understand what's happening

**Week 3+**:
- Phases C, D, E
- Build optimizer, ablate, compare

---

## The Big Picture

**Your contribution**: Adaptive Branch Optimization (ABO)

A dynamic loss-weighting algorithm that learns optimal balance between:
- Main task loss (Focal Tversky)
- Uncertainty task loss (Evidential Beta)

**Why it matters**: Medical AI models often train on multiple objectives. Static weights are suboptimal. ABO learns the right balance.

**Your proof**: 
- Baseline (Exp 00): X% Dice
- With ABO (Exp 05): X% + 2-4% Dice
- Better than PCGrad/GradNorm (Exp 06): ✓ Confirmed

That's your paper. That's your contribution. That's what you're building.

---

## Philosophy

From now on:

✅ **DO**:
- Document research questions before running experiments
- Save all results (never overwrite)
- Compare only against clean baseline
- Ablate systematically
- Measure carefully

❌ **DON'T**:
- Change the architecture
- Edit final_model.py
- Use ad-hoc hyperparameter tuning
- Skip documentation
- Make claims without evidence

---

## Questions?

Refer to:
- `RESEARCH_MODE.md` — Overview
- `RESEARCH_PROGRAM.md` — Full program
- `experiments/README.md` — How to run experiments

---

## Current Status

🟢 **Phase A - ACTIVE**

Exp 00 is training. Baseline will be ready in 1-2 days.

Then: Phase B (gradient analysis).  
Then: Phases C-E (ABO development, ablation, comparison).

Let's go build something that matters. 🚀

---

**Last Updated**: 2026-07-29 20:50 UTC  
**Next Check**: In 30 min (when Exp 00 test completes)
