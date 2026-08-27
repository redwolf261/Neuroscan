# Phase E54 — A96 Whole-Volume Resolution Training: Design

## Motivation

E29 (`PHASE_E29_RESIZE_SURVIVAL.md`) established, purely geometrically (no training), that the median native-space GT lesion component **vanishes entirely** under the current 64³ whole-volume resize, with survival improving monotonically at 96³/128³. E29 proposed the obvious follow-up — an "A64/A96/A128" whole-volume training comparison — but never ran it. This phase runs that experiment: no architecture change, just training `UNet3D_v3` (unchanged) at 96³ instead of 64³, to establish empirically whether resolution alone moves Dice, and by how much.

## Success criteria (user-specified, this session)

- ≥+1.0pp Dice over canonical baseline (0.9063) → target ≥0.9163.
- Full 125-subject validation set.
- 3-seed reproducibility with mean/variance reported, once a real signal is confirmed on seed 0.
- Fits GPU budget, practical inference cost.

## Memory profiling (measured, not assumed)

`profile_memory_96.py` measured real peak GPU memory for one forward+backward pass at 96³, `UNet3D_v3`, full loss stack (seg + boundary + D4/D2 aux):

| Batch size | Peak memory | Fits 8.15GB budget (with 1GB margin)? |
|---|---:|---|
| 8 | 15.626 GB | No — OOM |
| 6 | 11.731 GB | No |
| 4 | 7.872 GB | No (too tight) |
| 2 | 3.974 GB | **Yes** |

96³ is 3.375× the voxel count of 64³ (which measured 4.97GB at batch=8, PHASE_E27_PROJECT_AUDIT.md); activation memory scaled noticeably worse than the naive voxel-ratio estimate would suggest, confirming this needed direct measurement rather than assumption.

**Batch=2 is the only size that fits.** Batch=2 alone would make BatchNorm3d statistics (used in every `Conv3DBlock`) considerably noisier than every prior condition's own batch=8 runs, and would change the optimizer's per-step update dynamics (4× more steps per epoch, noisier gradients). **Fix: gradient accumulation** — physical batch=2 per forward pass, accumulate over 4 steps before `optimizer.step()`, giving the optimizer an effective batch of 8 (matching every prior condition's own update dynamics). BatchNorm itself still only sees 2 samples per forward pass — a real, disclosed limitation of this GPU's memory budget at 96³, not hidden or assumed away.

## Training protocol

- `UNet3D_v3`, completely unchanged.
- `target_shape=(96,96,96)` via the existing `target_shape` parameter in `Dataset/brats_dataset.py`/`create_brats_loaders` (never previously exercised above 64³).
- Physical batch=2, gradient accumulation over 4 steps (effective batch=8).
- AdamW, lr=4e-4, weight_decay=1e-5, CosineAnnealingLR(eta_min=1e-6), 30 epochs — unchanged from every prior condition.
- Same loss stack as every D4/D2-supervised condition: `seg_loss` (FocalTversky+Evidential, 0.5/0.5) + `mu*boundary_loss` (mu=0.1) + `lambda_ds3*aux3_loss` (0.9927) + `lambda_ds2*aux2_loss` (1.0014) — all weights reused unchanged, since no new loss term is introduced here.

## Pre-declared gates

- **Smoke test**: 2 epochs, seed 0. Sanity check only — no NaN/Inf, no early degenerate collapse. Not expected to numerically match 64³ trajectories epoch-for-epoch (different resolution changes early-training dynamics somewhat), but should show the same general shape (climbing Dice, not stuck near 0).
- **Kill condition**: val Dice < 0.5 after epoch 5 → immediate stop, matching every prior condition.
- **Seed policy**: seed 0 first. If seed 0 shows a real, non-trivial improvement (clearing or meaningfully approaching 0.9163), run seeds 1–2 and report mean/variance before any claim, per this project's mandatory ≥3-seed policy.

## What this tells us either way

- If A96 alone clears 0.9163 with 3-seed reproducibility: this is the result. Simple, defensible, no further mechanism needed.
- If A96 shows a real but sub-threshold gain: sets the empirical bar for E55 (the dual-resolution local-refinement mechanism) to beat, and confirms E29's own "moderate, not dramatic" prediction.
- If A96 shows no gain or is worse: resolution alone (without preserving compute efficiency or targeting small lesions specifically) isn't sufficient — motivates E55's more targeted approach directly.

## Artifacts on disk

- `experiments/exp_e12_eggo_m/e54/profile_memory_96.py`, `E54_memory_profile_96.json`
- `experiments/exp_e12_eggo_m/e54/train_e54_a96_resolution.py` (training script)
