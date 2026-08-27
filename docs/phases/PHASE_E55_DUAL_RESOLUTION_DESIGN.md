# Phase E55 — Dual-Resolution Local Refinement: Design

## Motivation

E54 (`PHASE_E54_A96_RESOLUTION_RESULT.md`) showed whole-volume high-resolution training (96³ vs 64³, no architecture change) is flat (0.9066, +0.03pp) — confirming E29's own prediction that resolution recovery concentrates narrowly (5–150-native-voxel lesions) and gets washed out in a pooled Dice metric. E48 (`PHASE_E48_BOTTLENECK_ENCODING_AUDIT_REVERSED_FINDING.md`) showed small lesions depend disproportionately on global/bottleneck context (ρ=−0.454, p<0.001) — so any fix must recover local geometric detail *without* discarding global context.

E55's mechanism: keep the existing 64³ global pathway (`UNet3D_v3`) completely intact, and add a small local pathway that examines a native-resolution crop centered on the region the global pathway currently believes the lesion is, fusing its refined prediction back in. This targets resolution recovery specifically where it's needed (small/uncertain lesions) rather than paying whole-volume cost uniformly.

## Key implementation facts (verified this session, not assumed)

- `Dataset/brats_dataset.py`'s `__getitem__` never touches `flair_nib.affine` — not needed. `flair_data` (native ndarray) is in memory before the resize call. Coordinate mapping is a simple per-axis scalar ratio: `native_coord[i] = resized_coord[i] * (current_shape[i] / target_shape[i])`.
- `scipy.ndimage.zoom`'s indexing convention does not match `grid_sample(align_corners=True)`'s convention. Verified conversion: `native_to_norm(coord, size) = 2*coord/(size-1) - 1`.
- `grid_sample(padding_mode='zeros', align_corners=True)` handles out-of-bounds crop centers gracefully (verified numerically: returns zeros, no crash).
- Native volume GPU memory cost is trivial (~35.7MB/sample, ~143MB at batch=4) — not a binding constraint.
- No `grid_sample`/differentiable-crop code exists anywhere in this codebase — genuinely new.

## Architecture

Two forward passes per training step:
1. Global pathway (`UNet3D_v3`, unchanged) → `probs_coarse`.
2. Differentiable soft centroid from `probs_coarse` (probability-weighted mean voxel-index, epsilon-guarded).
3. Map centroid: 64³ grid → native voxel coords (scalar ratio) → `grid_sample` normalized coords (align_corners=True formula).
4. `grid_sample` a 96³ crop from the native volume (now on GPU) at that location.
5. Local pathway (2–3 `Conv3DBlock`s, capped ~16–32 channels) processes the crop → local refinement logits.
6. Fuse: `logit_fused = logit_coarse + g * scatter(logit_local)`, `g` a learned scalar near 0 init. At `g=0`, exact bit-for-bit match to `UNet3D_v3`.

## Loss

`L_local`: `FocalTverskyLoss` (unmodified) between local prediction and native-resolution GT, cropped via the identical mapping (`mode='nearest'` for the label crop). `lambda_local` calibrated by gradient magnitude (not value), target ratio ~1.0, following `calibrate_lambda_cw_v2.py`'s template.

## Verification order (each gates the next)

1. Coordinate-mapping unit test (synthetic marker, numeric check).
2. Label-crop alignment spot-check (5–10 real subjects).
3. Ablation-safety (`g=0` → bit-for-bit match to v3).
4. Gradient-flow check (`L_local` gradient reaches `probs_coarse`).
5. Memory profile at target batch size.
6. 2-epoch smoke test.

## Success criteria

≥0.9163 Dice, full 125-subject validation set, 3-seed mean/variance if seed 0 shows a real signal, clearly distinguishable from baseline/D4-only given observed seed variance, practical GPU/inference cost.
