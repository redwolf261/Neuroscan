# NeuroScan Research Mode: Activated

**Date**: 2026-07-29  
**Status**: 🔬 ALGORITHM RESEARCH MODE ACTIVE  
**Current Phase**: A — Establishing Baseline

---

## What Changed

You've shifted from **"building models"** to **"conducting algorithm research"**.

### Before (Infrastructure Building)
- ✓ Built BraTS data loader
- ✓ Set up Python 3.11 GPU environment
- ✓ Created training infrastructure
- ✓ Validated simple UNet baseline (54.5% Dice)

### Now (Algorithm Research)
- Every experiment answers a research question
- Every result is reproducible from saved configs
- Every comparison is against a clean baseline
- Every ablation is deliberate and documented

---

## Your Research Program

### Phase A: Establish Baseline (THIS WEEK)
**Experiment**: `exp00_neuroscan_baseline`  
**RQ**: Can frozen NeuroScan transfer to BraTS?  
**Status**: Training now

```bash
cd experiments/exp00_neuroscan_baseline
python train.py --epochs 50
# Wait ~25 hours (CPU) or ~2 hours (GPU when CUDA fixed)
```

**When done**:
- Save `checkpoints/best.pth`
- Extract validation Dice from `checkpoints/history.json`
- Document in `README.md`

### Phase B: Gradient Diagnostics (WEEK 2)
**Experiment**: `exp01_diagnostics`  
**RQ**: What drives convergence? Gradient conflicts?

### Phase C: Optimizer Implementation (WEEK 3)
**Experiments**: `exp02_ema`, `exp03_alpha`, `exp04_delta`, `exp05_full_abo`  
**RQ**: Does ABO help? By how much?

### Phase D: Ablations (WEEK 4)
**Analysis**: Which components matter?

### Phase E: Competitor Comparison (WEEK 5)
**Comparison**: ABO vs PCGrad, GradNorm, CAGrad

---

## Key Files

| File | Purpose |
|------|---------|
| `RESEARCH_PROGRAM.md` | Full program outline (5 phases) |
| `experiments/README.md` | How to run experiments |
| `neuroscan_frozen.py` | Frozen NeuroScan model (no changes) |
| `experiments/exp00_neuroscan_baseline/train.py` | Phase A training script |

---

## How to Run Experiments

### Activate Environment
```powershell
cd C:\Users\Rivan\Projects\Neuroscan
.\venv_gpu\Scripts\Activate.ps1
```

### Run Exp 00 (Baseline)
```bash
cd experiments/exp00_neuroscan_baseline
python train.py --epochs 50
# Saves to: checkpoints/best.pth, checkpoints/history.json
```

### Run All Experiments (Later)
```bash
cd experiments/exp02_ema && python train.py
cd experiments/exp03_alpha && python train.py
cd experiments/exp04_delta && python train.py
cd experiments/exp05_full_abo && python train.py
```

### Compare Results (After all experiments)
```python
python analyze_experiments.py
# Output:
# exp00_neuroscan_baseline: Dice=53.2%
# exp02_ema: Dice=53.7%
# exp03_alpha: Dice=54.6%
# exp05_full_abo: Dice=56.4%
```

---

## Measuring Success

### Phase A (NOW)
- ✓ Exp 00 runs without errors
- ✓ Validation Dice > 30%
- ✓ Training curves look reasonable

### Phase B
- ✓ Identify which loss component dominates
- ✓ Measure gradient conflicts
- ✓ Document layer-wise gradients

### Phase C
- ✓ ABO implementation compiles
- ✓ Gradient damper prevents NaN/Inf
- ✓ EMA tracks gradient trends

### Phase D
- ✓ Ablation results clearly ranked
- ✓ Full ABO > baseline by 2-3% Dice

### Phase E
- ✓ ABO > PCGrad, GradNorm, CAGrad
- ✓ Difference is statistically significant

---

## Critical Rules

**NEVER break these**:

1. **Never edit `final_model.py` again**
   - It's frozen at `baseline-frozen` tag
   - Use `neuroscan_frozen.py` wrapper instead

2. **Never delete experiment checkpoints**
   - You might need to reproduce results 6 months later
   - Every checkpoint is precious

3. **Always save config with results**
   - Hyperparams must be reproducible
   - `config.yaml` goes in each exp directory

4. **Document RQ before running**
   - Write the research question first
   - Analyze results after, not before

5. **Compare only against baseline**
   - Exp 00 is ground truth
   - All improvements measured relative to it

---

## What You're Building

**Adaptive Branch Optimization (ABO)**

```
Standard Multi-Task Training:
  Loss = α·L_focal + (1-α)·L_evidential
         ↑ static weights
         
ABO:
  Loss = α_t·L_focal + (1-α_t)·L_evidential
         ↑ dynamic weights based on gradients
         ↑ + gradient damping
         ↑ + EMA smoothing
```

**Why it matters**: Current methods use fixed loss weights. ABO learns optimal balance during training. For medical AI with multi-task losses (main segmentation + uncertainty quantification), this should improve convergence.

---

## Timeline

```
Week 1: Phase A (Baseline)        ← YOU ARE HERE
Week 2: Phase B (Diagnostics)
Week 3: Phase C (Optimizer)
Week 4: Phase D (Ablations)
Week 5: Phase E (Comparisons)
Week 6: Analysis, writing, figures
```

**Next 24 hours**:
1. Exp 00 should finish training (on CPU: ~25h, on GPU when fixed: ~2h)
2. Extract final validation Dice
3. Document results in `exp00_neuroscan_baseline/README.md`
4. Confirm baseline is > 30% (indicates convergence)

---

## Quick Reference Commands

```powershell
# Activate GPU environment
cd C:\Users\Rivan\Projects\Neuroscan
.\venv_gpu\Scripts\Activate.ps1

# Run baseline experiment
cd experiments\exp00_neuroscan_baseline
python train.py --epochs 50

# Check results
cat checkpoints\history.json  # See loss/Dice curves
ls checkpoints\best.pth       # Confirm checkpoint saved

# Compare all experiments (Week 5)
python analyze_experiments.py
```

---

## Questions to Guide Each Phase

**Phase A (Now)**: 
- Does it run? 
- Does it converge?
- What's the final Dice?

**Phase B**: 
- Which loss dominates gradients?
- Is there gradient conflict?
- How do gradients evolve during training?

**Phase C**: 
- Does ABO code run?
- Does it improve baseline?

**Phase D**: 
- Which ABO components matter most?
- Can we remove any?

**Phase E**: 
- Is ABO better than existing methods?
- By how much?
- Is it statistically significant?

---

## Your Next Task

**RIGHT NOW**:
1. Check if Exp 00 is training (it is, in background)
2. Wait for it to complete (~25 hours on CPU)
3. When done: extract validation Dice
4. Write Phase A summary

**Tomorrow**:
- Start Phase B (gradient diagnostics planning)
- Instrument training loop for gradient logging

**This week**:
- Finish Phase B
- Start Phase C (optimizer implementation)

---

## Remember

You're not just training models anymore. You're **conducting research**.

Every experiment must:
- ✓ Answer one clear research question
- ✓ Be reproducible from saved configs
- ✓ Be compared fairly against baseline
- ✓ Be documented properly

**This is PhD-level work.** The infrastructure is done. Now comes the science.

🚀 Phase A in progress. Let's build something that matters.
