# Phase E54 — A96 Whole-Volume Resolution Training: Result

## Context

E29 established geometrically (no training) that the median native GT lesion vanishes under the 64³ resize, with monotonic recovery at 96³/128³, but explicitly cautioned the expected training-time gain would be "real but likely moderate... not dramatic," since the recovery concentrates in a narrow lesion-size band (~13% of the population) and most tiny "components" are single-voxel annotation noise unrecoverable at any resolution. E54 ran the training experiment E29 proposed but never executed: `UNet3D_v3`, completely unchanged, `target_shape=96³` instead of 64³, gradient accumulation (physical batch=2 × 4 steps = effective batch 8, since 96³/batch=8 measured ~15.6GB, far over the 8.15GB budget).

## Result (seed 0)

**Best val Dice: 0.9066** (epoch 30, final epoch). Full 30-epoch trajectory clean throughout — no instability, no NaN/Inf, smooth monotonic climb (train Dice 0.28→0.93, val Dice 0.54→0.91), HD95 improved steadily to 1.72 (notably better boundary-distance metric than typical 64³ runs at a comparable point, though not compared head-to-head with a matched 64³ run in this same session).

## Verdict against success criteria

- **Canonical baseline (0.9063) → 0.9066: +0.03pp.** Essentially a dead tie, not a real improvement.
- **D4-only (0.9096) → −0.30pp.** Below the established secondary bar.
- **Required: ≥1.0pp (≥0.9163). NOT MET, not close.**

## Decision

**Per the pre-declared plan: seed 0 shows no real, non-trivial signal, so this does not proceed to seeds 1–2.** Running 2 more seeds to "confirm" a result already indistinguishable from baseline would burn ~7 hours of compute (3.4hrs/seed × 2) without a plausible path to changing the conclusion — the gap to even D4-only is too large to be seed noise alone (this project's own measured seed-to-seed std, from CCABA/IECG/CCAG, tops out around 0.05–0.30pp; +0.30pp below D4-only is outside that band).

## Interpretation

This is a clean, informative negative result, directly consistent with E29's own honest prediction. Simply increasing whole-volume resolution, with no other architectural change, did not translate into a meaningful Dice gain — most likely because:
1. The resolution-recovery benefit E29 measured concentrates narrowly (5–150-native-voxel lesions, ~13% of the population); a pooled/whole-volume Dice metric can easily wash out a real, localized gain in a small subpopulation.
2. Batch=2 (even with gradient accumulation preserving the optimizer's effective batch=8) still gives BatchNorm much noisier per-forward-pass statistics than every prior batch=8 condition — a real, disclosed cost of running at this resolution on this GPU, which may itself be absorbing some of whatever gain resolution alone would otherwise provide.
3. Whole-volume high-resolution training treats every subject/voxel equally expensively, without any mechanism to concentrate the resolution benefit specifically where E48 showed it's causally needed (small lesions, which depend disproportionately on context) — this is exactly the gap the more targeted E55 design (dual-resolution local refinement, gated on this result per the approved plan) is built to address.

## Next step

Per the approved plan (`ok-what-would-you-iridescent-rossum.md`), this result — a real but sub-threshold/flat outcome — motivates proceeding to **E55**: a mechanism that targets resolution recovery specifically where E48's causal evidence says it matters (small lesions, via a local high-resolution pathway) while preserving the global context E48 showed is essential, rather than paying whole-volume high-resolution cost uniformly across all subjects regardless of lesion size.

## Artifacts on disk

- `experiments/exp_e12_eggo_m/e54/profile_memory_96.py`, `E54_memory_profile_96.json`
- `experiments/exp_e12_eggo_m/e54/train_e54_a96_resolution.py`
- `experiments/exp_e12_eggo_m/e54/e54_smoke_log.txt`, `e54_seed0_run_log.txt`
- `experiments/exp_e12_eggo_m/e54/runs/A96_seed0/epoch_metrics.csv` (30 rows), `checkpoints/` (disk only, gitignored)
