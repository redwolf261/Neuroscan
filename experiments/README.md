# NeuroScan Optimization Research Experiments

This directory contains all experiments for the Adaptive Branch Optimization (ABO) algorithm research.

**Key Principle**: Every experiment answers a specific research question. All results are reproducible from saved configurations.

---

## Experiment Progression

### Exp 00: NeuroScan Baseline *(CURRENT)*
**Status**: Setting up  
**RQ**: Can frozen NeuroScan transfer to BraTS tumor segmentation?

```bash
cd exp00_neuroscan_baseline
python train.py --epochs 50
```

**Deliverables**:
- `checkpoints/best.pth` — Best model (target: ~50% Dice)
- `checkpoints/history.json` — Loss/Dice curves
- `config.yaml` — Exact hyperparameters used

**Success Criteria**:
- ✓ Model converges (validation loss decreasing)
- ✓ Validation Dice > 30% (shows learning)
- ✓ Baseline established for comparison

---

### Exp 01: Diagnostic Framework
**RQ**: What gradient dynamics drive NeuroScan convergence on BraTS?

**What to measure**:
- Gradient norm per loss component (focal vs evidential)
- Gradient norm per architecture branch
- Cosine similarity between gradient vectors
- Layer-wise gradient magnitudes
- Gradient variance over training

**Deliverables**:
- Gradient logs for every batch
- Visualizations: gradient norms vs epoch
- Analysis: which branch dominates? Is grad alignment good?

---

### Exp 02-05: Optimizer Ablations

| Exp | Components | Goal |
|-----|------------|------|
| 02  | EMA only | Test if temporal averaging helps |
| 03  | EMA + α | Test adaptive weighting |
| 04  | EMA + δ | Test gradient damping |
| 05  | Full ABO | Test all components together |

Each compares against Exp 00 baseline.

**Metric**: Δ Dice = (Exp Dice) - (Baseline Dice)

**Success**: If Δ Dice > 2-3%, the component matters.

---

### Exp 06: Optimizer Comparisons

Compare against:
- AdamW (baseline)
- PCGrad (gradient conflict resolution)
- GradNorm (gradient magnitude normalization)
- Others as practical

**Metric**: Final validation Dice

---

## Running Experiments

### Quick Test (1 epoch)
```bash
cd exp00_neuroscan_baseline
python train.py --epochs 1
```

### Full Run (50 epochs, ~25 hours on CPU)
```bash
cd exp00_neuroscan_baseline
python train.py --epochs 50 --batch_size 4
```

### From Python 3.11 venv
```powershell
# Activate
cd C:\Users\Rivan\Projects\Neuroscan
.\venv_gpu\Scripts\Activate.ps1

# Run
cd experiments\exp00_neuroscan_baseline
python train.py --epochs 50
```

---

## Saving Results

**NEVER overwrite**. Each run creates a new checkpoint:
```
checkpoints/
├── epoch_000.pth
├── epoch_010.pth
├── best.pth          ← Always points to best validation
└── history.json
```

To compare experiments, check:
```bash
cat exp00_neuroscan_baseline/checkpoints/history.json
cat exp02_ema/checkpoints/history.json
# Extract final validation Dice from each
```

---

## Analysis Tools

Create `analyze_experiments.py`:
```python
import json
from pathlib import Path

results = {}
for exp_dir in Path("experiments").glob("exp*/"):
    hist_file = exp_dir / "checkpoints" / "history.json"
    if hist_file.exists():
        with open(hist_file) as f:
            hist = json.load(f)
        results[exp_dir.name] = {
            "val_dice_final": hist["val_dice"][-1],
            "train_dice_final": hist["train_dice"][-1],
            "val_loss_final": hist["val_loss"][-1]
        }

for exp, metrics in sorted(results.items()):
    print(f"{exp}: Dice={metrics['val_dice_final']:.4f}")
```

---

## Documentation Template

Each experiment should have a `README.md`:

```markdown
# Experiment XX: [Short Name]

## Research Question
[What are we testing?]

## Hypothesis
[What do we expect?]

## Method
[How are we testing it?]

## Results
[What happened?]

## Interpretation
[What does this mean for the optimizer?]
```

---

## Key Rules

1. **Never delete checkpoints** — you might need to reproduce Fig 3 six months later
2. **Always save config** — hyperparams matter
3. **One experiment per directory** — clear separation
4. **Document RQ before running** — avoid post-hoc rationalization
5. **Compare against baseline** — Exp 00 is the gold standard

---

## Research Timeline

- **Week 1**: Exp 00 (baseline) ← YOU ARE HERE
- **Week 2**: Exp 01 (diagnostics)
- **Week 3**: Exp 02-05 (ablations)
- **Week 4**: Exp 06 (comparisons)
- **Week 5-6**: Analysis, writing, plots

---

**Next immediate task**: Run Exp 00 to establish baseline Dice on BraTS with frozen NeuroScan.
