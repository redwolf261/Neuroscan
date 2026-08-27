# NeuroScan Optimization Research Program

**Last Updated**: 2026-07-29  
**Status**: Phase A (Baseline) — In Progress  
**Infrastructure**: ✅ Complete. Research mode: ACTIVE.

---

## Research Goal

Design and validate **Adaptive Branch Optimization (ABO)**, a gradient-based algorithm that dynamically reweights loss components to improve medical image segmentation model training.

**Why this matters**: Medical AI models must converge efficiently on limited annotated data. Most multi-task learning methods use static weights. We propose an adaptive approach that learns optimal loss balance during training.

---

## Research Phases

### Phase A: Establish True Baseline (THIS WEEK)

**Objective**: Get frozen NeuroScan architecture running on BraTS with zero modifications.

**Why first**: We need a clean, reproducible baseline. The simple UNet validated infrastructure; NeuroScan validates that the algorithm research setup is correct.

**Experiment**: `exp00_neuroscan_baseline`
- Train HybridMiniSwin2D5_CBAM on BraTS
- No architectural changes
- Original loss function (HybridLoss with evidential)
- Original optimizer (AdamW + cosine annealing)
- Binary segmentation (tumor vs background)
- FLAIR modality only

**Deliverable**: 
```
experiments/exp00_neuroscan_baseline/
├── checkpoints/best.pth          # Model
├── checkpoints/history.json      # Loss/Dice curves
└── config.yaml                   # Exact hyperparams
```

**Success Criteria**:
- ✓ Model converges (val loss decreasing)
- ✓ Validation Dice > 30%
- ✓ Training reproducible from config

**Timeline**: 1-2 days

---

### Phase B: Re-run Gradient Diagnostics (WEEK 2)

**Objective**: Understand gradient dynamics in BraTS training (not PediMS).

**Why second**: Your previous gradient analysis came from a model that barely learned (Dice ~0.006). BraTS data is different. Need to revalidate core assumptions.

**Experiment**: `exp01_diagnostics`

**Measurements**:
```
Per-batch logging:
├── Loss values
│   ├── Focal Tversky loss
│   ├── Evidential Beta loss
│   └── Total HybridLoss
├── Gradient norms
│   ├── Per layer
│   ├── Per component (evidential branch, main branch)
│   └── Global gradient norm
├── Gradient statistics
│   ├── Cosine similarity (gradient vectors)
│   ├── Gradient variance
│   └── Gradient angle between loss components
└── Layer-wise analysis
    ├── Gradient magnitude per layer
    ├── Gradient saturation
    └── Activation magnitude
```

**Research Questions**:
1. Does evidential branch still dominate gradient flow?
2. Is Dice loss aligned with Focal Tversky?
3. How does gradient behavior change from early to late training?
4. Which layers have vanishing/exploding gradients?
5. Do loss components conflict (negative cosine similarity)?

**Visualization**:
```
plots/
├── gradient_norms_vs_epoch.png       # Trend over training
├── gradient_cosine_similarity.png    # Component alignment
├── layer_wise_gradients.png          # Per-layer analysis
└── loss_contributions.png             # Which loss dominates?
```

**Success Criteria**:
- ✓ Identify gradient dominance pattern
- ✓ Quantify loss component conflict
- ✓ Document gradient saturation points

**Timeline**: 3-4 days

---

### Phase C: Implement Optimizer (WEEK 3)

**Objective**: Build Adaptive Branch Optimization (ABO).

**Design**:
```
Forward Pass
    ↓
Compute all losses (focal, evidential, total)
    ↓
Backward (compute gradients)
    ↓
EMA Controller
    • Track exponential moving average of gradient norms
    • Smooth out noise, detect trends
    ↓
Adaptive Weight α_t
    • Adjust loss component weights based on gradient magnitude
    • Formula: α_focal = 1 - β·(||∇L_evid|| / ||∇L_focal||)
    ↓
Gradient Damper δ_t
    • Clip extreme gradients
    • δ = min(1.0, τ / ||∇L||)
    ↓
Merge Rule
    • Combine clipped gradients
    • α_focal * ∇L_focal + (1-α_focal) * ∇L_evid
    ↓
AdamW Step (unchanged optimizer)
    • Standard parameter update
```

**Experiment**: `exp02_ema`, `exp03_alpha`, `exp04_delta`, `exp05_full_abo`

**Key design decisions**:
- EMA window: 5-10 batches (smooth without lag)
- α update frequency: Every N batches (e.g., 10)
- δ threshold τ: Adaptive (e.g., 95th percentile)
- No weight clipping after damping (only gradient clipping)

**Success Criteria**:
- ✓ Compiles and runs without errors
- ✓ Gradient damper prevents NaN/Inf
- ✓ EMA controller tracks gradient trends correctly

**Timeline**: 4-5 days

---

### Phase D: Ablation Studies (WEEK 4)

**Objective**: Prove which components of ABO actually matter.

**Experiments**:

| Exp | Config | Expected Δ Dice |
|-----|--------|-----------------|
| 00  | Baseline (no ABO) | 0% |
| 02  | EMA only | +0.5-1% |
| 03  | EMA + α | +1-2% |
| 04  | EMA + δ | +1-2% |
| 05  | Full ABO | +2-4% |

**Analysis**:
- Which component contributes most?
- Are components synergistic or redundant?
- Interaction effects?

**Visualization**:
```
plots/
├── ablation_comparison.png       # All experiments on one plot
├── component_contribution.png    # Bar chart of Δ Dice
└── ablation_table.csv            # Exact numbers
```

**Success Criteria**:
- ✓ Exp 05 > Exp 00 (ABO helps)
- ✓ Clear ranking of component importance
- ✓ Statistically significant improvements

**Timeline**: 3-4 days

---

### Phase E: Competitor Comparisons (WEEK 5)

**Objective**: Show ABO is better than existing approaches.

**Baselines**:
- AdamW (vanilla optimizer)
- PCGrad (gradient projection for conflicts)
- GradNorm (balance by magnitude)
- CAGrad (conflict avoidance, if practical)

**Experiment**: `exp06_comparisons`

**Metric**:
```
Final validation Dice:
├── AdamW: X.XX%
├── PCGrad: Y.YY%
├── GradNorm: Z.ZZ%
├── ABO: W.WW%
```

**Table**:
```
| Method    | Val Dice | Improvement |
|-----------|----------|-------------|
| AdamW     | 53.2%    | baseline    |
| PCGrad    | 54.1%    | +0.9%       |
| GradNorm  | 54.7%    | +1.5%       |
| ABO       | 56.4%    | +3.2% ✓     |
```

**Success Criteria**:
- ✓ ABO > all baselines
- ✓ Improvement is statistically significant
- ✓ Fair comparison (same hyperparams, seeding, data)

**Timeline**: 3-4 days

---

## Experimental Infrastructure

### Directory Structure
```
experiments/
├── README.md                           # This guide
├── exp00_neuroscan_baseline/
│   ├── train.py                       # Training script
│   ├── config.yaml                    # Hyperparams (saved)
│   └── checkpoints/
│       ├── best.pth                   # Best model
│       ├── epoch_*.pth                # All checkpoints
│       └── history.json               # Loss/Dice curves
├── exp01_diagnostics/
│   ├── train.py                       # Instrumented training
│   ├── logs/                          # Gradient logs
│   └── plots/                         # Visualizations
└── exp02-06/
    └── [similar structure]
```

### Configuration Management
Every experiment saves its exact config:
```yaml
# exp00_neuroscan_baseline/config.yaml
dataset:
  root_dir: "Dataset/Training"
  val_split: 0.1
  target_shape: [64, 64, 64]
training:
  epochs: 50
  batch_size: 4
  learning_rate: 0.0001
model:
  in_channels: 1
  out_channels: 1
```

To reproduce Exp 00:
```bash
cd exp00_neuroscan_baseline
python train.py --epochs 50
# Reads config.yaml automatically
```

### Result Tracking
```bash
# Compare all experiments
python analyze_experiments.py

# Output:
# exp00_neuroscan_baseline: Dice=53.2%
# exp02_ema: Dice=53.7%
# exp03_alpha: Dice=54.6%
# exp05_full_abo: Dice=56.4%
```

---

## Metrics & Success Criteria

### Primary Metric
**Validation Dice on BraTS**

- Baseline (Exp 00): ~50-55%
- With ABO (Exp 05): ~52-58% (target: +2-3%)

### Secondary Metrics
- Training time (should not increase significantly)
- Convergence speed (should improve or stay same)
- Gradient statistics (documented in Phase B)
- Ablation clear ranking (documented in Phase D)

### Statistical Rigor
- Run each experiment 3x with different seeds
- Report mean ± std for all metrics
- Use paired t-tests for comparisons
- Confidence intervals: 95%

---

## Key Research Questions

**By end of Phase A**:
- Can NeuroScan transfer to BraTS?
- What is the clean baseline Dice?

**By end of Phase B**:
- How do gradients flow through multi-task losses?
- Is there gradient conflict?
- Which components dominate?

**By end of Phase C**:
- Does ABO algorithm run without errors?
- Does it improve baseline?

**By end of Phase D**:
- Which ABO components matter most?
- Are there redundancies?

**By end of Phase E**:
- Is ABO better than PCGrad/GradNorm/CAGrad?
- By how much?
- Is improvement significant?

---

## Writing the Paper

**Structure**:
1. **Motivation** — Medical AI, limited data, need for efficient training
2. **Background** — Multi-task learning, gradient conflict, existing solutions
3. **Method** — ABO algorithm, design choices, theoretical justification
4. **Experiments** — Phase A-E results
5. **Analysis** — What works, why, ablation insights
6. **Comparisons** — ABO vs PCGrad/GradNorm/etc
7. **Discussion** — Limitations, future work, broader impact
8. **Conclusion** — Key contributions, call to action

**Figures**:
- Fig 1: Architecture diagram (NeuroScan)
- Fig 2: Gradient dynamics (Phase B diagnostics)
- Fig 3: ABO algorithm flowchart
- Fig 4: Training curves (baseline vs ABO)
- Fig 5: Ablation results
- Fig 6: Comparison with baselines
- Fig 7: Case studies (visualization of improved segmentations)

---

## Timeline Summary

| Week | Phase | Tasks |
|------|-------|-------|
| 1 | A | Baseline training, establish Dice |
| 2 | B | Gradient diagnostics, analysis |
| 3 | C | ABO implementation, testing |
| 4 | D | Ablation studies (5 experiments) |
| 5 | E | Competitor comparisons |
| 6 | — | Analysis, paper writing, figures |

---

## Current Status

✅ **Week 1 — Phase A In Progress**
- Exp 00 training started
- Baseline should be ready in 1-2 days
- Next: Establish validation Dice

---

## Next Immediate Actions

1. **Complete Exp 00** (1-2 days)
   - Check if baseline training converges
   - Record final validation Dice
   - Save checkpoint and config

2. **Document results** (0.5 day)
   - Write exp00_neuroscan_baseline/README.md
   - Extract key metrics
   - Create baseline plot

3. **Plan Phase B** (0.5 day)
   - Design diagnostic logging
   - Identify which layers to instrument
   - Plan visualizations

**Philosophy**: Every experiment answers one question. No shortcuts, no ad-hoc tuning. Clean research.

---

**Status**: Phase A active. Infrastructure ready. Research mode engaged. 🚀
