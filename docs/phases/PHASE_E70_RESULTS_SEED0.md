# Phase E70 — Results (Seed 0)

Evaluated against the protocol pre-registered in
`PHASE_E70_CAS_DESIGN_AND_PROTOCOL.md` **before** these runs finished.
Primary endpoint: per-subject mean Dice (E56's corrected protocol),
n = 125 validation subjects.

## Headline numbers

| Model | Pooled Dice | Per-subject Dice |
|---|---:|---:|
| FLAIR-only baseline (`e24/A_baseline_seed0`) | 0.9063 | 0.8842 |
| MM (4-modality, `UNet3D_v3`) | 0.9175 | 0.9023 |
| **MM_CAS (4-modality + CAS, `UNet3D_v10`)** | **0.9200** | **0.9052** |

## The three pre-registered comparisons

| # | Comparison | Mean Δ | Bootstrap 95 % CI | paired-t | Wilcoxon | % improved |
|---|---|---:|---|---:|---:|---:|
| 1 | MM vs FLAIR-only (modality gain) | **+1.81 pp** | [+0.80, +3.13] ✅ | 2.9e-03 | 2.0e-07 | 72.0 % |
| 2 | **MM_CAS vs MM (CAS ablation)** | **+0.30 pp** | [+0.07, +0.53] ✅ | 1.2e-02 | 5.6e-03 | 60.8 % |
| 3 | MM_CAS vs FLAIR-only (total system) | **+2.11 pp** | [+1.04, +3.54] ✅ | 1.2e-03 | 1.0e-09 | 76.8 % |

All three bootstrap CIs exclude zero.

## Against the project's historical targets

| Target | Threshold | MM_CAS | Status |
|---|---|---:|---|
| Corrected per-subject (E56) | ≥ 0.8942 | 0.9052 | ✅ clears by 1.10 pp |
| **Original pooled (pre-E56)** | ≥ 0.9163 | **0.9200** | ✅ **clears by 0.37 pp** |

The pooled target is the bar that E45, E46, E49, E50, E51, E54 and E55 all
failed to reach. The multimodal system clears it.

## Convergence stability

Neither result is a single-epoch spike. MM's final six epochs span
0.9005–0.9023 per-subject (range 0.0018); MM_CAS's final seven epochs are
consistently ahead of MM at the matched epoch (+0.15 to +0.39 pp), having
converged from an early-epoch lead that had closed by epoch 8.

## Honest interpretation

**Comparison 1 is real but is not a contribution.** Adding T1/T1ce/T2 to a
FLAIR-only model is expected to help; every published BraTS method uses all
four modalities. The single-modality restriction was a self-imposed project
constraint, not a scientific baseline. This was stated in the pre-registered
protocol before the numbers existed, and it does not change now that the
number is large.

**Comparison 2 is the CAS claim, and on this evidence it is suggestive but
NOT confirmed.** The CI excludes zero and both tests are significant, but
+0.30 pp sits inside this project's own demonstrated between-seed noise
range (CCABA ≈ 0.32 pp; E45 ≈ 0.55 pp). This project has twice seen a
single-seed effect with *stronger* statistics fail to replicate:

| Mechanism | Seed-0 result | 3-seed outcome |
|---|---|---|
| E45 (D4+D8) | +0.97 pp, both tests significant | not confirmed |
| E54 (A96) | +1.02 pp, both tests significant | not confirmed |
| **CAS** | **+0.30 pp, both tests significant** | **unknown — 1 seed only** |

By the project's own multi-seed standard, comparison 2 requires seeds 1 and
2 (≈ 10 GPU-hours) before it can support a claim.

**A methodological caveat on comparison 1 and 3:** the FLAIR-only reference
is a prior run from a different phase — identical recipe and identical 125
validation subjects, but not a seed-matched control trained alongside these.
Comparison 2 has no such caveat: same seed, same recipe, same data, and CAS
is provably the exact identity at initialisation (5.96 × 10⁻⁸), so the two
conditions differ in exactly one thing.

## Gate diagnostic: the mechanistic prediction FAILED

`analyze_cas_gates.py`, seed 0, n = 125. Three analyses, all pre-registered.

**A. Did CAS learn a non-trivial correction? — Yes.**

| Quantity | Value |
|---|---:|
| Gate mean over subjects | 0.722 |
| Gate max over subjects | 0.971 |
| Mean \|offset\| | 1.53 voxels |
| Max \|offset\| | 6.50 voxels (bound 6.93) |
| Collapsed to identity? | **No** |

So the +0.30 pp is not explained by "the module did nothing."

**B. Does the gate agree with E65's causal sensitivity map? — NO. It
anti-correlates.**

| Quantity | Value |
|---|---:|
| Mean per-subject Spearman ρ(gate, translation-sensitivity) | **−0.388** |
| Median ρ | −0.388 |
| Subjects with ρ > 0 | **0 / 125 (0.0 %)** |
| t-test p | 8.2e-114 |
| Wilcoxon p | 3.0e-22 |

The pre-registered prediction was that the gate would be **elevated** where
E65 measured the network to be most correspondence-sensitive. The observed
relationship is the **opposite**, unanimously across all 125 subjects.

**C. Is the gate concentrated on lesions? — No, the reverse.**

| Region | Mean gate |
|---|---:|
| Lesion | 0.493 |
| Background | 0.724 |
| Paired Δ | **−0.231** (p = 3.6e-134) |

CAS applies its heaviest resampling to **background** and largely leaves
**lesion** regions alone.

### What this means

CAS is **not** performing the function it was designed to perform. It learned
an active, non-trivial transformation, but that transformation is
anti-aligned with the causally measured correspondence structure and is
concentrated away from the lesions. The mechanistic hypothesis behind the
module is refuted on this seed.

This removes the strongest novelty argument for the design (a module trained
only on segmentation loss independently recovering an independently measured
causal structure). That claim cannot be made — the data says the opposite.

It also **weakens** the +0.30 pp result: a small Dice gain with no working
mechanistic explanation is more consistent with the extra 29 k parameters
providing incidental optimisation benefit than with correspondence
correction. The seed-noise hypothesis becomes more plausible, not less.

## Follow-up: what IS the gate actually tracking?

`investigate_cas_gate.py`, seed 0, n = 125. Tested the most parsimonious
alternative hypothesis: the gate tracks local `enc1` **activation
statistics**, not correspondence need.

| Test | Mean ρ | frac negative | p |
|---|---:|---:|---:|
| Gate vs. local \|enc1\| magnitude | **−0.566** | 125/125 (100 %) | 2.8e-133 |
| Gate vs. local 3×3×3 variance | **−0.555** | 125/125 (100 %) | 8.9e-123 |
| Gate vs. E65-sensitivity, **partialled on magnitude** | −0.087 | — | 1.6e-44 |

The magnitude/variance correlations are **stronger and equally unanimous**
(100 % vs. the original 100 % in the wrong direction for the correspondence
hypothesis). Controlling for local magnitude shrinks the gate↔sensitivity
correlation from −0.388 to **−0.087** — a ~78 % reduction in effect size.

### Interpretation

CAS learned a **magnitude/variance-gated smoothing operator**: it resamples
heavily where the skip tensor is quiet and homogeneous (background) and
leaves it alone where activations are strong and structured (lesion
boundaries, informative tissue). This single, simple mechanism explains all
three original observations at once — the background/lesion split, and most
of the anti-correlation with the causal sensitivity map, which is itself
plausibly elevated in the same high-magnitude, information-dense regions
CAS avoids.

This is a coherent, sensible thing for gradient descent to discover — a
conservative "only touch what's quiet" editing rule — but it is **not**
correspondence correction, and it is not what the module was designed to do.

## Status

- Seed 0 complete for both conditions; all statistics computed.
- Gate diagnostic complete — **mechanistic prediction falsified**.
- Follow-up investigation complete — **root cause identified**: CAS is a
  magnitude-gated smoothing operator, not a correspondence-correction
  operator.
- Seeds 1 and 2 **not** launched (user paused).

## Where this leaves the contribution

| Claim | Status |
|---|---|
| Multimodal system reaches 0.9200 pooled / 0.9052 per-subject | ✅ solid |
| +2.11 pp over the prior FLAIR-only baseline | ✅ solid, but expected (modality gain) |
| CAS improves Dice (+0.30 pp) | ⚠️ single seed, inside noise range, unreplicated |
| **CAS works by correspondence correction** | ❌ **falsified by the gate diagnostic** |
| Causal-audit protocol (measure → design → verify) | ✅ intact — it produced a falsifiable prediction, and the verification step caught the failure |

## Artifacts

- `experiments/exp_e12_eggo_m/e70/E70_evaluation.json`
- `experiments/exp_e12_eggo_m/e70/runs/MM_seed0/`, `runs/MM_CAS_seed0/`
- `neuroscan_3d_v10.py`, `Dataset/brats_dataset_multimodal.py`
