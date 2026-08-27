# Phase A.5: Baseline Convergence Study — FROZEN (two-head architecture)

**Status**: ✅ **COMPLETE — this is the reference baseline for all future experiments**

**Supersedes**: an earlier single-head run (archived at
`experiments/exp00b_baseline_convergence/archive_single_head_baseline/`,
90.92% ± 0.05% Dice). That run used a model with one output head, where the
"evidential loss" was an algebraic reweighting of the same segmentation
tensor (α=100p+1, β=100(1-p)+1) — not a genuinely separate branch. It was
retired once that was identified, since it gave ABO's future gradient
diagnostics nothing real to measure. See "Architecture change" below.

## Architecture change: single-head → two-head

`neuroscan_3d_fixed.py`'s `UNet3D` now has a shared encoder-decoder trunk
that splits into two independent heads after `dec1` (32ch, full resolution):

```
dec1 (shared, 32ch, B×32×64×64×64)
  ├─→ seg_head:        Conv3d(32→1) → Sigmoid           → probs
  └─→ evidential_head:  Conv3d(32→2) → chunk → softplus+1 → alpha, beta
```

- **Zero parameter overlap** between the two heads (verified directly by
  comparing `id()` of each head's parameters).
- Both heads receive independent, nonzero gradient on backward (verified:
  seg_head grad norm 0.168 vs evidential_head grad norm 0.049 on a random
  batch — different magnitudes, not scaled copies of one signal).
- `EvidentialBetaLoss` now consumes the *real* learned `(alpha, beta)`
  instead of reconstructing them from `probs`. It implements the standard
  evidential-deep-learning formulation (Sensoy et al. 2018, adapted to
  binary Beta): expected BCE under the Beta mean, plus a KL-to-Beta(1,1)
  regularizer that zeroes out evidence for the *correct* class before
  computing KL — so it only penalizes confident evidence on voxels the
  model got wrong, not blanket uncertainty everywhere.
- `dec1` (the shared trunk) receives gradient contributions from both
  downstream paths merged — this is the point where Phase B's gradient
  conflict measurement becomes meaningful, since the two paths are now
  genuinely different computation subgraphs, not reweightings of one.

## Config (locked — must not change in Phase B/C/D/E)

```yaml
model: UNet3D (neuroscan_3d_fixed.py, two-head)
loss: HybridLoss (FocalTversky[seg_head] + EvidentialBeta[evidential_head], 0.5/0.5)
optimizer: AdamW
scheduler: CosineAnnealingLR (eta_min=1e-6)
batch_size: 8
learning_rate: 0.0004
weight_decay: 0.00001
num_workers: 4
epochs: 50 (no early stop triggered, patience=15)
device: cuda (RTX 5050)
```

Dataset: `Dataset/Training` only. 1,251 subjects, patient-level split (seeded
shuffle, seed=42, in `Dataset/brats_dataset.py`), 1,126 train / 125 val,
disjoint, FLAIR-only, binary tumor-vs-background labels.

`Dataset/Validation` (219 subjects, official BraTS 2023 GLI blind leaderboard
set) has **no segmentation labels** — imaging only. Cannot be used for local
Dice/IoU/F1/HD95. Not used in this study.

## Results (3 seeds, 50 epochs each, full metric suite)

| Seed | Best Epoch | Dice | IoU | Precision | Recall | F1 | HD95 | Avg epoch time | Peak GPU mem |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 50 | 0.9107 | 0.8370 | 0.9128 | 0.9096 | 0.9107 | 1.44 | 162.1s | 4406 MB |
| 1 | 49 | 0.9095 | 0.8349 | 0.9149 | 0.9053 | 0.9095 | 1.38 | 142.6s | 4408 MB |
| 2 | 37 | 0.9097 | 0.8353 | 0.9196 | 0.9010 | 0.9097 | 1.09 | 136.8s | 4405 MB |

**Mean Dice ± Std: 0.9100 ± 0.0005** (91.00% ± 0.05%)

Statistically indistinguishable from the retired single-head baseline
(90.92% ± 0.05%) — adding a genuine second branch and switching to the
real Beta-evidential KL loss did not cost segmentation quality. Seed
variance remains extremely low (same order as before), so this is again a
stable, reproducible number, not a lucky run. Precision/recall balanced
across seeds. Peak GPU memory ~4.41GB (52% of 8.5GB), consistent with the
single-head run's ~4.39GB — the extra head adds negligible memory overhead.
Epoch time is slightly higher (137-162s vs 116-140s before), from the extra
head's forward/backward and the digamma/lgamma-based KL computation.

## Data leakage check (passed, unchanged from single-head run)

- Train/val split is patient-level (one 3D volume = one subject = one sample),
  not slice-level — no possibility of the same patient appearing in both sets.
- Split is a seeded random shuffle (not alphabetical by subject ID), so val
  isn't biased by BraTS ID ordering/acquisition site.
- No data augmentation applied (deterministic resize + normalize only), so
  there's no augmented-duplicate leakage risk either.
- `validate()` correctly uses `model.eval()` + `torch.no_grad()`.
- Metrics computed only on the held-out val split, never on train.

## Why batch_size=8 / lr=4e-4 (not larger)

Isolated GPU throughput test on the UNet3D model (single-head variant, but
architecture-level memory footprint is materially unchanged by the second
head — confirmed post-hoc: peak memory 4.41GB now vs 4.34GB then):

| Batch size | Peak memory | Throughput | Verdict |
|---|---|---|---|
| 8  | 4.34 GB (51%) | 13.66 samples/sec | **fastest, stable** |
| 12 | 6.50 GB (76%) | 3.73 samples/sec | slower — allocator pressure |
| 16 | 8.66 GB (102%) | 1.47 samples/sec | over budget, thrashing |

Larger batch sizes counter-intuitively got *slower* once GPU memory pressure
rose past ~50-60%, due to allocator fragmentation/retries — confirmed by
clean, repeated multi-iteration timing (not a single noisy sample). LR scaled
linearly with the 4x batch increase from the original batch_size=2 baseline
(1e-4 → 4e-4); validated stable across all 3 seeds on both the single-head
and two-head models, no collapse either time.

## What this baseline is for

This is the frozen reference point for:
- **Phase B**: gradient diagnostics — now measuring genuine conflict between
  `seg_head`'s FocalTversky gradient and `evidential_head`'s Beta-KL gradient
  at the shared `dec1` trunk, not two reweightings of one signal.
- **Phase C**: Adaptive Branch Optimizer (ABO) implementation
- **Phase D**: ablation studies (EMA, alpha scheduling, gradient damping)
- **Phase E**: comparison against PCGrad, GradNorm, CAGrad

**Every one of those experiments must use this exact config** (batch_size=8,
lr=4e-4, this two-head architecture, same data split seed logic) so that any
Dice difference observed is attributable to the optimizer/loss change being
tested, not a confounded hyperparameter or architecture change.

**Parameter groups note for Phase C**: the optimizer currently uses a single
flat `AdamW(model.parameters())` group — no `encoder`/`decoder`/`seg_head`/
`evidential_head` separation. If ABO requires group-level control (per the
original design), the optimizer construction will need to be restructured
to expose these groups explicitly; this hasn't been done yet.

## Files

| File | Purpose |
|---|---|
| `neuroscan_3d_fixed.py` | Model (UNet3D, two-head) + loss classes (FocalTverskyLoss, EvidentialBetaLoss, HybridLoss) |
| `experiments/exp00b_baseline_convergence/train.py` | Convergence study script (seeds, early stopping, full metrics) |
| `experiments/exp00b_baseline_convergence/metrics.py` | Dice/IoU/Precision/Recall/F1/HD95 implementations |
| `experiments/exp00b_baseline_convergence/seed_{0,1,2}/checkpoints/best.pth` | Best model weights per seed (two-head) |
| `experiments/exp00b_baseline_convergence/seed_{0,1,2}/results.json` | Full per-epoch history per seed |
| `experiments/exp00b_baseline_convergence/summary.json` | Cross-seed mean/std |
| `experiments/exp00b_baseline_convergence/archive_single_head_baseline/` | Retired single-head results, kept for reference |
| `configs/brats.yaml` | Locked training config |
| `Dataset/brats_dataset.py` | Data loader (seeded patient-level split) |

---

**Frozen**: 2026-08-02
