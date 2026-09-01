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

## Status

- Seed 0 complete for both conditions.
- Gate diagnostic (`analyze_cas_gates.py`) running — tests whether CAS's
  learned gate recovers E65's independently measured correspondence
  sensitivity. A near-zero gate would indicate MM_CAS ≈ MM functionally and
  would undercut the +0.30 pp as mechanism rather than noise.
- Seeds 1 and 2 **not** launched (user paused).

## Artifacts

- `experiments/exp_e12_eggo_m/e70/E70_evaluation.json`
- `experiments/exp_e12_eggo_m/e70/runs/MM_seed0/`, `runs/MM_CAS_seed0/`
- `neuroscan_3d_v10.py`, `Dataset/brats_dataset_multimodal.py`
