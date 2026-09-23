# E180 Stage 0 — Freeze manifest

**Date**: 2026-09-16
**Status**: PRE-REGISTERED. Not run. No scientific computation.

---

## Purpose

E180 tests a new hypothesis (representation-induced decision instability predicts recoverable
segmentation benefit) that must not disturb or be confounded with any prior experiment. This
document fixes, before any E180 code runs, exactly what is frozen and how identity is verified.

## Frozen (not modified by E180)

- **Checkpoint**: `experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth`
  Identity check: `ckpt["best_mean_dice"] == 0.8929357248544694` (abs diff `< 1e-9`), the same
  assertion used verbatim by E165/E167/E169.
- **Dataset**: `Dataset/brats_multimodal_dataset.py::create_multimodal_loaders`, `val_split=0.1`,
  `seed=0` — reproduces the fixed 125-subject validation split (`RandomState(42)` shuffle, last
  10%).
- **Preprocessing**: unchanged, as defined in `BraTSMultimodalDataset`.
- **Sliding-window geometry**: 128³ tiles, Gaussian-blended reassembly
  (`_gaussian_weight`, `sigma_scale=0.125`), fixed `overlap=0.25` for every E180 comparison.
- **Segmentation inference / metric implementation**: unchanged (`model(tile)["probs"]`,
  threshold 0.5, binary Dice).
- **Subject IDs**: the 125 validation subject IDs, recorded here as a hash, never recomputed.
- **Existing experiment artifacts**: `E126_*`, `E160_L_per_subject.json`, `E165_*`, `E167_*`,
  `E169_*` (including `e169/FROZEN/`) — read-only references, never overwritten.

## New experiment family

All E180 code and outputs live under `experiments/exp_e12_eggo_m/e180/` and
`docs/phases/PHASE_E180_*.md`. No file outside these paths is modified by this program.

## Deliverable: `experiments/exp_e12_eggo_m/e180/FROZEN/MANIFEST.json`

```json
{
  "checkpoint_path": "...best.pth",
  "checkpoint_sha256_16": "<16 hex chars>",
  "checkpoint_best_mean_dice": 0.8929357248544694,
  "dataset_root": "Dataset/Training",
  "dataset_val_split": 0.1,
  "dataset_seed": 0,
  "n_val_subjects": 125,
  "subject_id_list_sha256_16": "<16 hex chars>",
  "code_commit": "<git rev-parse HEAD>",
  "python_version": "...",
  "torch_version": "...",
  "cuda_available": true,
  "gpu_name": "...",
  "sliding_window_overlap": 0.25,
  "patch_size": [128, 128, 128],
  "target_stage": "enc3",
  "note": "FROZEN <date>. E180 Stage 0. No modification to E126-E176 artifacts."
}
```

Hash convention matches `e169/FROZEN/MANIFEST.json`: SHA-256 truncated to the first 16 hex
characters (`sha256_16`). `code_commit` and `checkpoint_sha256_16` extend that precedent (not
previously recorded by any experiment) since E180 pre-registers reproducibility more strictly
per the user's Stage 0 requirement.

## Verification

- Script asserts checkpoint identity before hashing.
- Script asserts `len(val_loader.dataset) == 125` before hashing the subject ID list.
- Manifest is written once; any later Stage's script re-reads and re-verifies against it rather
  than recomputing independently.
