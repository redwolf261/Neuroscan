# Phase E61 — Bottleneck Effective-Dimensionality Audit

## Purpose

Test the narrower hypothesis raised after E59/E60 (synergy formulation killed): does
small-lesion segmentation depend on a **higher-effective-dimension** bottleneck
representation than large-lesion segmentation? No training, no new architecture, no
loss/hyperparameter changes — a pure representation-analysis audit.

## 1. Checkpoint

- **v5/E46 attention-gate checkpoint** (`experiments/exp_e12_eggo_m/e46/runs/AttnGate_seed0/checkpoints/best.pth`, val_dice=0.9102).
- Chosen because it is the **exact** checkpoint E48's CausalDrop numbers were computed
  from (required for the quantitative test in step 11).
- **Limitation, disclosed up front:** E58 and E59 (the octant/interaction results E61
  cross-checks against in step 12) actually used the plain **v3/D4-only** checkpoint —
  not v5. E48 and E58/E59 are themselves on different architectures. E61 could not
  match both simultaneously from one checkpoint; per explicit sign-off, it matched E48
  (the quantitative target) and treated the E59 cross-check as **qualitative/directional
  only**, never a cross-architecture correlation.
- Validation population: same 125-subject val split (`val_split=0.1`) used by every
  E43–E60 script.

## 2–3. Representation construction

- Clean forward pass per subject to `B ∈ R^(256,8,8,8)`, no GT involved in extraction.
- Sanity check: bottleneck-extraction path reproduces `model.forward()`'s trunk
  bit-for-bit (max abs diff = 0.0).
- Primary matrix: `X_n ∈ R^(512×256)` (spatial cells as observations).
- Secondary/transposed matrix: `X_n^T ∈ R^(256×512)` (channels as observations).
- Centering: **subject-wise**, declared before any result was viewed — each subject's
  own per-feature mean subtracted before SVD, identical procedure for every subject and
  both groups.

## 4. Metrics (all three, pre-declared, none cherry-picked)

Effective rank (Shannon entropy of the singular-value distribution), stable rank
(`‖X‖_F² / ‖X‖_2²`), and participation ratio, computed on both the spatial and channel
orientations, plus a scale-normalized (unit-Frobenius-norm) variant.

## 5. Lesion-regime definition

E35's five native-size bins (S1–S5, built for **per-component** size) do not
discriminate at E48's **per-subject total lesion burden** granularity — all 125 E48
subjects fall in S5 under those edges. Per explicit sign-off, the primary split is a
**median split on E48's own native_size field**: small = below median (n=63), large =
above median (n=62), median = 90,377 voxels.

## Results

| Metric | mean(small) | mean(large) | diff | perm p |
|---|---|---|---|---|
| R_eff (spatial) | 95.145 | 98.148 | **−3.003** | 1.000 |
| R_stable (spatial) | 2.384 | 2.371 | +0.013 | 0.292 |
| Participation ratio (spatial) | 5.014 | 5.010 | +0.004 | 0.459 |
| R_eff (channel) | 99.230 | 102.559 | **−3.330** | 1.000 |
| R_eff (scale-normalized) | 95.145 | 98.148 | −3.003 | 1.000 |

**The direction is reversed from the hypothesis in the metric that shows any effect at
all**: large-lesion subjects have *higher* effective rank, not small. The two other
primary metrics (stable rank, participation ratio) show no significant difference in
either direction. 0/3 primary metrics support "small > large."

### Activation-scale control (step 7)
Small-lesion subjects have systematically *lower* activation magnitude across every
scale statistic (Frobenius norm, spectral norm, mean/var activation; all p<0.005) — a
real, non-trivial confound direction. After scale normalization the R_eff gap is
unchanged (still −3.00, still large > small), so the reversed direction is not an
artifact of scale, but it also means scale normalization does not rescue the original
hypothesis.

### Lesion-burden control (step 8)
`R²(R_eff ~ log(native_size)) = 0.383` — size alone explains most of the group
variance in R_eff, confirming R_eff and lesion size are tightly coupled (in the
large>small direction). After residualizing R_eff on log(native_size), the small/large
gap disappears entirely (p=0.45) — the raw group difference is essentially just this
size relationship, not an independent dimensionality effect.

### Spatial vs. channel locus (step 10)
Neither orientation shows a significant small>large effect (both False) — moot given
the primary direction is reversed anyway.

### Randomized-control null (step 9)
Real bottlenecks show **much lower** effective rank than the feature-shuffled null
(96.6 vs 233.3, p=2.96e-22) — confirms real cross-channel/cross-cell structure exists
and the pipeline is measuring something real, just not the hypothesized group effect.

### E48 CausalDrop relationship (step 11)
Spearman(R_eff, CausalDrop) = **−0.202** (perm p=0.022) — significant, but in the
direction opposite the hypothesis's composed prediction (which required a positive
relationship, since both small-lesion status and higher R_eff were predicted to align
with higher CausalDrop). After controlling for log(native_size), the partial
correlation flips sign and loses significance (ρ=+0.081, p=0.368) — the raw
correlation is a secondary consequence of the same size confound, not an independent
mechanistic link.

### E59 qualitative cross-check (step 12)
E59's own verdict was itself MIXED/non-definitive. E61's direction (large, not small,
shows higher R_eff) does not qualitatively agree with the small-lesion-synergy framing
E59 explored. Cross-architecture, informal only, per the disclosed limitation.

### Robustness (step 13)
0/2 preprocessing variants (subject-wise vs. scale-normalized) show the hypothesized
direction at p<0.05 — the finding is not preprocessing-dependent because there is no
positive finding to be dependent.

## Decision rule (step 14) — all 7 pre-declared criteria

| # | Criterion | Result |
|---|---|---|
| 1 | Small > large consistently | **False** (large > small, where significant) |
| 2 | Survives activation-scale control | False |
| 3 | Survives lesion-burden control | False |
| 4 | ≥2/3 primary metrics agree | False (0/3) |
| 5 | Survives permutation/bootstrap | False |
| 6 | Associated with E48 CausalDrop (post size-control) | False |
| 7 | Not a trivial consequence of size/Dice | False |

## Verdict: **KILL**

Small lesions do **not** require a higher-effective-dimension bottleneck
representation. If anything, the one metric showing a significant, robust effect
(effective rank) points the opposite way — large-lesion subjects' bottlenecks show
higher effective rank — and that effect is fully explained by the lesion-size↔R_eff
regression (R²=0.38), not an independent dimensionality phenomenon. The correlation
with E48's CausalDrop does not survive controlling for lesion size.

This closes the "higher-dimensional bottleneck for small lesions" hypothesis raised
after E59/E60. Combined with E58 (no spatial localization), E59/E60 (synergy
formulation killed), and E47 (routing null), this is now the fourth distinct framing
of "what's special about the bottleneck for small lesions" to return a clean null or
reversed-direction result. The causal fact from E48 (bottleneck ablation hurts small
lesions more) remains real and significant — but its mechanism is still not
explained by localization, synergy, or dimensionality.

## Artifacts

- `experiments/exp_e12_eggo_m/e61/run_e61_bottleneck_dimensionality_audit.py`
- `experiments/exp_e12_eggo_m/e61/E61_bottleneck_dimensionality_table.json` (125 subject records)
- `experiments/exp_e12_eggo_m/e61/E61_summary.json`
