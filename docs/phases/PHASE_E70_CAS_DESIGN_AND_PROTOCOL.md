# Phase E70 — Correspondence-Aware Skip (CAS): Design & Pre-Registered Protocol

**Written before results were available.** Design, hypotheses, and success
criteria are recorded here in advance so the analysis cannot be rationalised
after the fact.

## 1. What is being tested

Two conditions, differing in exactly one variable:

| Condition | Architecture | Input |
|---|---|---|
| `MM` | `UNet3D_v3` | 4 channels (T1, T1ce, T2, FLAIR) |
| `MM_CAS` | `UNet3D_v10` = v3 + CAS at the `enc1` skip | 4 channels (identical) |

Everything else is identical and inherited from this project's established
recipe (`e25/train_deep_sup.py`): AdamW, `CosineAnnealingLR(T_max=30,
eta_min=1e-6)`, lr/weight-decay from `configs/brats.yaml`, gradient clipping
1.0, 30 epochs, D4-only deep supervision (`lambda_ds3=0.9927`, E25b's own
calibration), `seg_loss = 0.5·FocalTversky + 0.5·EvidentialBeta`, identical
train/val split (seed-42 shuffle, same 125 validation subjects as every prior
phase).

## 2. The CAS mechanism

At the `enc1 → dec1` junction, replacing the raw concatenation:

$$
\delta = \texttt{MAX\_OFFSET}\cdot\tanh\big(W_{\text{off}}([\,U(D_2),\,E_1\,])\big)
\qquad
g = \sigma\big(W_{\text{gate}}([\,U(D_2),\,E_1\,])\big)
$$
$$
E_1^{\text{aligned}} = \texttt{grid\_sample}\big(E_1,\ \mathcal{I} + \delta\big)
\qquad
E_1^{\text{out}} = (1-g)\cdot E_1 + g\cdot E_1^{\text{aligned}}
$$

with `MAX_OFFSET = 4.0` voxels, $\mathcal{I}$ the identity sampling grid, and
$W_{\text{off}}, W_{\text{gate}}$ zero-initialised.

**Verified properties** (tested, not assumed):
- **Exact identity at initialisation**: $\delta = 0 \Rightarrow E_1^{\text{aligned}} = E_1
  \Rightarrow E_1^{\text{out}} = E_1$ regardless of $g$. Measured max
  |difference| vs. `UNet3D_v3` with matched weights: **5.96 × 10⁻⁸** (float
  precision). MM_CAS therefore *starts from exactly MM's function* and must
  learn any deviation.
- **Bounded displacement**: $\|\delta\|_\infty \le 4.0$ voxels by construction
  (verified by forcing head saturation).
- **Parameter overhead**: 29,428 params (**+0.52 %** over v3's 5,605,414).

## 3. Why this design, from this project's own evidence

E65's causal decomposition of the `enc1` skip (n=125, v3 checkpoint, four
independent interventions, each pre-registered):

| Intervention on the skip tensor | Mean Dice drop |
|---|---|
| **3-voxel translation** | **0.214** |
| Channel permutation | 0.101 |
| 3×3×3 smoothing | 0.028 |
| 2×2×2 local derangement | 0.027 |

E64 separately established, via a split intervention on two checkpoints, that
this effect lives **entirely in the skip path** (Δ_pool = exactly 0.0) and not
in pooling — correcting the earlier E62/E63 attribution.

CAS is designed directly against that measurement: the dominant causal
requirement at this junction is *absolute spatial correspondence*, so the
module provides an explicit, learned, bounded correspondence correction, with
a gate that can express "no correction needed" exactly. The 4-voxel bound is
set by the displacement range over which E65 actually measured the damage
curve (2–4 voxels).

## 4. Prior art — stated plainly

Learned deformable-offset correction at encoder–decoder skip junctions is
**not new**: Dynamic U-Net's DCU module (arXiv:2403.07303, 2024, 2D abdominal
CT) computes $\Delta = \mathcal{F}_{\text{conv}}([F, S])$ and applies it via
modulated deformable convolution. Related work includes MPLSeg
(Fourier magnitude/phase decoupling of semantics vs. localisation) and
offset-fidelity supervision for deformable alignment in video super-resolution
(Chan et al.).

CAS differs in three concrete, evidence-traceable ways:
1. **3D volumetric** formulation with trilinear resampling (DCU is 2D).
2. **Bounded** offset (tanh-scaled) rather than unconstrained — justified by
   the measured damage regime and by known instability of unconstrained
   deformable offsets.
3. **Gated residual blend** rather than always-on resampling, making CAS a
   strict superset of the baseline skip (gate → 0 recovers it exactly). DCU's
   formulation cannot express "leave this junction alone", so a mis-learned
   offset can only degrade an already-correct junction.

The honest novelty claim is therefore: *a bounded, gated, 3D correspondence
correction whose form and displacement range are derived from a pre-registered
causal audit of the specific junction it modifies* — not "we invented
deformable skip alignment."

## 5. Pre-registered analysis

Primary endpoint: **per-subject mean Dice** (E56's corrected protocol).
Secondary: pooled Dice, reported for continuity with pre-E56 numbers.

Three comparisons, each against a named reference:

| # | Comparison | Question answered |
|---|---|---|
| 1 | `MM` vs. FLAIR-only baseline | Gain from modality expansion |
| 2 | **`MM_CAS` vs. `MM`** | **Gain attributable to CAS alone** |
| 3 | `MM_CAS` vs. FLAIR-only baseline | Total system gain |

Comparison **2 is the ablation that isolates the novel component** and is the
one that must be reported for any claim about CAS specifically. Statistics:
paired t-test and Wilcoxon signed-rank on per-subject deltas, plus a
10,000-sample bootstrap 95 % CI over subjects.

## 6. Pre-declared interpretation rules

- A gain in comparison 1 is **expected and is not a contribution** — adding
  three modalities to a FLAIR-only baseline should help; it is reported as
  context, not as a result.
- CAS is supported **only** by comparison 2 with a bootstrap CI excluding 0.
  If comparison 3 shows ≥1 pp but comparison 2 does not, the honest reading
  is "the system improves, driven by modality expansion; CAS is not
  demonstrated" — and it must be written that way.
- Single seed is **exploratory**. Per this project's own multi-seed policy
  (established after the CCABA variance finding, and reinforced by the
  E45/E54 3-seed result where single-seed effects dissolved), any confirmed
  claim requires ≥3 seeds. Seed 0 alone cannot confirm CAS.

## 7. Status

Both conditions launched, 30 epochs each, seed 0. Results and their honest
interpretation to be recorded in a separate results document.
