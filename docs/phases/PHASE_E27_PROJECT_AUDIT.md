# E27/PERF-NOVELTY Project Audit

**Date**: 2026-08-13. All answers below are read directly from source files and checkpoints in this session (not reconstructed from memory), with the file paths cited so you can re-verify. Where I did not check a value directly, it says `unknown`.

---

## A. Exact model and architecture

### A1. Exact baseline architecture (★)

Class `UNet3D` in [neuroscan_3d_fixed.py](neuroscan_3d_fixed.py) (also called `UNet3D_v2`/`UNet3D_v3` — v2 adds a dormant boundary head, v3 additionally adds the deep-supervision aux heads; the encoder/bottleneck/decoder trunk is byte-identical across all three).

```text
Input (B, 1, 64, 64, 64)  -- FLAIR only
 ↓
enc1: Conv3DBlock(1→32) → Conv3DBlock(32→32)         [Conv3d k3 + BatchNorm3d + ReLU, x2]
 ↓ MaxPool3d(2)
enc2: Conv3DBlock(32→64) → Conv3DBlock(64→64)
 ↓ MaxPool3d(2)
enc3: Conv3DBlock(64→128) → Conv3DBlock(128→128)
 ↓ MaxPool3d(2)
bottleneck: Conv3DBlock(128→256) → Conv3DBlock(256→256)
 ↓
ConvTranspose3d(256→128, k2, s2)  → concat with enc3 (skip) → dec3: Conv3DBlock(256→128)→Conv3DBlock(128→128)   [D/4]
 ↓
ConvTranspose3d(128→64, k2, s2)   → concat with enc2 (skip) → dec2: Conv3DBlock(128→64)→Conv3DBlock(64→64)      [D/2]
 ↓
ConvTranspose3d(64→32, k2, s2)    → concat with enc1 (skip) → dec1: Conv3DBlock(64→32)→Conv3DBlock(32→32)       [full res]
 ↓
seg_head: Conv3d(32→1, k1) → Sigmoid
```

- Encoder type: plain 3D CNN encoder, no attention/transformer blocks anywhere, no CBAM despite earlier project notes mentioning it as planned (not present in this frozen file).
- 3 encoder stages + bottleneck, 3 decoder stages.
- Convolution blocks only: `Conv3d(k=3, padding=1) → BatchNorm3d → ReLU`, two per stage.
- Skip connections: standard U-Net concatenation (encoder feature ⊕ upsampled decoder feature), channel-doubled before each decoder block.
- Upsampling: `ConvTranspose3d(kernel=2, stride=2)` (learned transposed convolution, not interpolation).
- Final segmentation head: `Conv3d(32→1, k=1)` + `Sigmoid`.
- Additional heads sharing the `dec1` trunk: an `evidential_head` (`Conv3d(32→2, k=1)`, producing Beta-distribution α/β via softplus+1), and in v2/v3 a dormant `boundary_head` reading `dec1.detach()`. v3 additionally has `aux_head3` (dec3→1×1×1 conv→sigmoid) and `aux_head2` (dec2→1×1×1 conv→sigmoid), training-time only.

### A2. Exact tensor dimensions at each stage (★)

```text
Input:       (B, 1,  64, 64, 64)
enc1:        (B, 32, 64, 64, 64)
pool1:       (B, 32, 32, 32, 32)
enc2:        (B, 64, 32, 32, 32)
pool2:       (B, 64, 16, 16, 16)
enc3:        (B, 128,16, 16, 16)
pool3:       (B, 128, 8,  8,  8)
bottleneck:  (B, 256, 8,  8,  8)
dec3 (D/4):  (B, 128, 16, 16, 16)
dec2 (D/2):  (B, 64,  32, 32, 32)
dec1 (full): (B, 32,  64, 64, 64)
seg_head:    (B, 1,   64, 64, 64)
```

### A3. Exact parameter count (★)

Computed directly this session:

- `UNet3D_v2` (used for A): **5,602,628** total/trainable parameters.
- `UNet3D_v3` (used for D4-only/D2-only/Both): **5,602,822** total/trainable parameters (194 more, from the two 1×1×1 aux-head convs, each `128→1` and `64→1`, i.e. 129+65=194 params).
- Checkpoint size: A's `best.pth` = **67.32 MB**, D4-only's `best.pth` = **67.33 MB** (includes model+optimizer state, not just weights).
- Approximate inference time: `unknown` — not benchmarked directly; full-volume forward pass at batch=1 on the RTX 5050 is sub-second based on training epoch times (see B5/I2), but this was never isolated as a standalone measurement.

---

## B. Training configuration

### B1. Exact loss for baseline A (★)

From [neuroscan_3d_fixed.py](neuroscan_3d_fixed.py) `HybridLoss`, `FocalTverskyLoss`, `EvidentialBetaLoss`:

$$L_A = 0.5 \cdot L_{FocalTversky} + 0.5 \cdot L_{Evidential}$$

- **Dice/Tversky variant**: Focal Tversky loss, $\alpha=0.5, \beta=0.5$ (i.e. symmetric FP/FN weighting — mathematically reduces to a focal-weighted Dice at these values), $\gamma=4/3$, smooth$=1.0$, predictions clamped to $[10^{-6}, 1-10^{-6}]$ before the ratio. Global (whole-batch) reduction, not per-sample-then-averaged, inside the loss call itself.
- **BCE/CE variant**: none directly in the segmentation term — instead an *evidential Beta* term: expected-BCE-under-the-Beta-distribution (`bce = -(target·log(μ+ε) + (1-target)·log(1-μ+ε))` where `μ=α/(α+β)`) plus a KL-divergence evidence-removal regularizer (KL(Beta(α̃,β̃) ‖ Beta(1,1)), zeroing evidence for the correct class before penalizing), weight=0.5 inside `EvidentialBetaLoss`.
- **Class weighting**: none explicit (no positive-class weighting despite the strong class imbalance noted in D2/D3 below — this is itself a candidate lever, see roadmap).
- **Smoothing/epsilon**: Tversky smooth=1.0; evidential smooth=1e-6.
- **Other terms in the actual training runs (not in the frozen `HybridLoss` itself, added by the gate6/deep-sup training scripts)**: A also adds $\mu \cdot L_{boundary}$ with $\mu=0.1$ (BCEWithLogitsLoss on a dormant boundary head) — so A's *actual trained* loss is $L_A^{train} = L_{seg} + 0.1 \cdot L_{boundary}$ where $L_{seg}$ is the $0.5/0.5$ hybrid above.

### B2. Exact loss for D4-only (★)

From [train_deep_sup.py:229](experiments/exp_e12_eggo_m/e25/train_deep_sup.py#L229):

$$L_{D4only} = L_{seg} + \mu \cdot L_{boundary} + \lambda_{D4} \cdot L_{aux3}$$

with $\mu=0.1$, $\lambda_{D4}=0.9927$ (calibrated in `e25b_calibrate_deep_supervision.py` so that $\lambda \cdot L_{aux} \approx L_{seg}$ at initialization), $L_{aux3} = FocalTversky(aux\_probs3,\ mask_{D/4})$, and $mask_{D/4} = \text{avg\_pool3d}(mask, k=4, s=4)$ (fractional/soft downsampled target, not hard-thresholded). $\lambda_{D4}$ is a **fixed constant**, not scheduled/annealed, throughout all 30 epochs.

### B3. Were auxiliary heads trained against full-resolution ground truth?

No — **average pooling** to the aux head's own resolution (`F.avg_pool3d`, kernel=stride=4 for D/4, kernel=stride=2 for D/2), producing a soft/fractional-valued downsampled target (verified mean-preserving in `test_deep_supervision.py::test_downsampled_gt_construction`). `FocalTverskyLoss` operates directly on these fractional values (no thresholding).

### B4. Optimizer (★)

- Optimizer: **AdamW** (`torch.optim.AdamW`) — note this differs from `configs/brats.yaml`'s stated `"Adam"`; the actual training code overrides the config file.
- Learning rate: **4e-4** (`config["training"]["learning_rate"]`, from `configs/brats.yaml`).
- Weight decay: **1e-5**.
- Betas: PyTorch AdamW defaults (0.9, 0.999) — not overridden anywhere found.
- Scheduler: **CosineAnnealingLR**, `T_max=epochs` (30), `eta_min=1e-6`.
- Warmup: config file specifies `warmup_epochs: 5`, but `unknown`/likely unused — no warmup-scheduler code was found wired into `train_gate6.py`/`train_deep_sup.py`'s actual scheduler construction (only `CosineAnnealingLR` is instantiated); this should be verified directly before relying on it.
- Gradient clipping: **`clip_grad_norm_(max_norm=1.0)`**, applied every step (confirmed in `train_gate6.py:251`).

### B5. Training duration (★)

- Maximum epochs actually run: **30** (not the config file's 50 — code hardcodes `EPOCHS=30` in the orchestrators).
- Actual best epoch for A: **epoch 24 (1-indexed) / 23 (0-indexed)**, val_dice=0.9063 (directly from `epoch_metrics.csv`).
- Early stopping: config specifies `early_stopping_patience: 15` but `unknown`/likely not implemented in the actual training loop — no early-stop-triggering code was found in the scripts checked this session; all runs appear to complete the full fixed 30 epochs regardless of plateauing.
- Checkpoint selection rule: best-so-far by `val["dice"]` (per-epoch, batch-pooled Dice — see the metric-definition caveat in `PHASE_E25_STRUCTURAL_PIVOT_1B_4WAY_MECHANISM_COMPARISON.md` Section 0), saved whenever a new best is found, plus periodic saves every 5 epochs.
- Patience: `unknown` (see early stopping above).

### B6. Randomness

- Random seed: **0**, for every experiment (A, C6-2, C6-3, D4-only, D2-only, Both) — single locked seed throughout the E24/E25 arc, by explicit prior instruction (multi-seed only after a candidate clears meaningful-improvement gates, which has not yet happened).
- Deterministic PyTorch: **no** — `set_seed()` (`train_eggo_m.py:141`) sets `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, but does **not** set `torch.backends.cudnn.deterministic=True` or `torch.use_deterministic_algorithms(True)`.
- Deterministic CUDA: **no** (same reason).
- Dataloader workers: `num_workers=4` in production runs (config default); `num_workers=0` used deliberately for the initial-parameter-equality gate checks only (to avoid worker-process nondeterminism during that specific hash comparison).
- Augmentation randomness: **N/A — no augmentation exists in the training pipeline** (see E1).

---

## C. Data preprocessing

### C1. Exact preprocessing pipeline (★)

```text
raw MRI (BraTS 2023 GLI .nii.gz, native resolution, e.g. 240×240×155)
→ nibabel load, get_fdata(), cast to float32
→ (segmentation only) collapse 4-class labels to binary: any label > 0 → 1
→ scipy.ndimage.zoom to (64, 64, 64)   [order=1 linear for FLAIR, order=0 nearest for mask]
→ min-max normalize FLAIR to [0, 1] (per-volume min/max, post-resize)
→ add channel dim → (1, 64, 64, 64) tensor
→ model
```

No skull stripping, no registration (BraTS data is already co-registered/skull-stripped/bias-corrected by the challenge organizers upstream — not redone here), no explicit bias-field correction step in this codebase, no intensity clipping beyond the [0,1] min-max, no orientation-canonicalization step visible in this loader (relies on however nibabel/the BraTS files are stored). Interpolation: linear (order=1) for the image, nearest-neighbor (order=0) for the mask, both via `scipy.ndimage.zoom`.

### C2. What MRI modalities are used? (★)

**FLAIR only** (T2-FLAIR, `{subject_id}-t2f.nii.gz`). Explicitly single-channel by design ("keep architecture unchanged" per the file docstring); T1/T1c/T2 are present in the dataset on disk but never loaded.

### C3. Exact target resolution (★)

**Resized (resampled) whole volume**, not cropped, not padded, not a patch. Every subject's full native-resolution volume (e.g. 240×240×155) is `scipy.ndimage.zoom`'d directly to 64×64×64 in one shot — anisotropic zoom factors, different per axis and per subject (since native shape can vary slightly). This is a global downsampling of the *entire* brain into 64³, not a crop or patch extraction from a larger volume.

---

## D. Sampling — extremely important

### D1. How is the 64³ training volume/patch selected? (★)

**Neither** random crop, center crop, nor lesion-centered crop — there is no cropping at all. The entire subject volume is resized (whole-volume zoom, see C3) directly to 64³. This is the single most consequential preprocessing decision in the pipeline: a typical BraTS FLAIR volume is roughly 240×240×155 native, so 64³ represents a **~3.75×–3.75×–2.4× downsampling** per axis — small lesions that might occupy dozens of voxels natively can shrink to single-digit-voxel or sub-voxel footprints after resize, which is very likely a first-order cause of the small-lesion detection/quality problems the whole E25 arc has been chasing.

Exact code path: `BraTSDataset._resize_volume()` in `Dataset/brats_dataset.py:121-137`, called unconditionally on every `__getitem__`, no random component at all in the spatial sampling.

### D2. What percentage of training samples contain lesion voxels?

From a random sample of 150/1126 training subjects computed this session: **0% zero-lesion** (0/150) — this BraTS-GLI cohort is curated to be tumor-positive for every subject, so every training volume has *some* lesion signal after resize. (Not the same question as "does every 64³ *patch*" — since there's no patching, this is equivalent to "does every subject volume," and the answer is effectively all of them, modulo the tiny chance a lesion is small enough to vanish entirely under nearest-neighbor downsampling — not directly measured but bounded below 2/150≈1.3% by the "low occupancy" count below.)

### D3. Distribution of lesion occupancy

Same 150-subject sample, fraction of the 64³ volume that is lesion (mask mean):

| Stat | Value |
|---|---:|
| Mean | 0.01009 (~1.0%) |
| Median | 0.00963 (~1.0%) |
| Min | 0.00051 (~0.05%) |
| Max | 0.02729 (~2.7%) |
| Low (<0.001, i.e. <~0.4 voxels) | 2/150 |
| Mid [0.001, 0.01] | 75/150 |
| High (>0.01) | 73/150 |

Severe class imbalance (~99% background) is confirmed directly, consistent with everything the E12-E26 arc has already found about small-lesion difficulty.

### D4. Are multiple patches generated from each subject?

No — one fixed whole-volume-resized sample per subject per epoch (no patching means no multi-patch sampling question applies).

### D5. Is there oversampling of lesion-containing patches?

No — `shuffle=True` in the DataLoader is uniform random over subjects, no lesion-aware sampling weight anywhere in `create_brats_loaders`.

---

## E. Augmentation

### E1. Complete augmentation pipeline (★)

**None.** `BraTSDataset.__getitem__` performs load → binarize → resize → normalize → tensor, with zero stochastic augmentation operations (no flip, rotation, scale, noise, intensity shift, gamma, elastic deformation, or any other transform). Confirmed by reading the full file; also confirmed no separate augmentation module is imported anywhere in the active training scripts (`train_gate6.py`, `train_deep_sup.py`) or referenced in `configs/brats.yaml`. This is a real, previously-undiscussed gap — every one of A/C6-2/C6-3/D4-only/D2-only/Both was trained with **no data augmentation whatsoever**, on a 1126-subject training set that is itself fixed and non-stochastic per epoch.

### E2. Are augmentations applied identically to image and mask?

N/A — no augmentations exist.

### E3. Any MRI-specific augmentation?

None (no bias field, motion, intensity nonuniformity, or acquisition-variation simulation anywhere in this codebase).

---

## F. Dataset split

### F1. Exact counts (★)

```text
Total:      1251 BraTS-GLI subjects on disk
Train:      1126 (90%)
Validation: 125 (10%)
Test:       none — no held-out third split exists anywhere in this pipeline
```

The 125-subject validation set has been used as the *sole* evaluation set for every checkpoint-selection decision, every diagnostic, every mechanism audit, and every go/no-go gate across the entire E12–E26 arc (dozens of analyses). **It has not been held completely untouched** in the strict sense of "never influenced any decision" — it has been the single metric every model-selection and stopping decision in this project has been made against, repeatedly, for months of iteration. This is worth being explicit about: there is currently no truly independent test set left to report a final, unbiased number on. If a paper-quality final number is needed, a fresh held-out split (or k-fold) should be established before final reporting.

### F2. How was the split created? (★)

Deterministic, seeded (`np.random.RandomState(42)`) shuffle of the sorted subject-directory list, then a fixed 90/10 slice (`Dataset/brats_dataset.py:52-63`). Subject-level (each BraTS-GLI folder is one subject, one scan) — not stratified by tumor size/type/anything else, and not the BraTS challenge's own official split (this project constructed its own train/val split from the Training partition only; the official BraTS validation/test sets, which have no public ground truth, are not used at all).

### F3. Any possibility of subject leakage?

Low risk: BraTS-GLI subject folders are one scan per subject (no repeated-timepoint structure observed in the training data naming convention, `BraTS-GLI-XXXXX-000` — the `-000` suffix suggests a timepoint field exists in the naming scheme but `unknown` whether any subject has multiple timepoints present in this specific downloaded copy of the dataset; not checked this session).

---

## G. Baseline error profile

### G1. Baseline A (★)

At best checkpoint (epoch 24/30, per `epoch_metrics.csv`, **batch-pooled** validation definition — see the Section 0 caveat in `PHASE_E25_STRUCTURAL_PIVOT_1B_4WAY_MECHANISM_COMPARISON.md`):

| Metric | Value |
|---|---:|
| Pooled/batch Dice (training's own val_dice) | 0.9063 |
| Mean per-subject Dice (this session's independent recomputation, full 125-subject val set) | 0.8872 |
| Median subject Dice | `unknown` — not computed this session, easily derivable from `deep_sup_4way_comparison.json`'s `dice_A` field if needed |
| Precision | 0.9114 |
| Recall | 0.9022 |
| F1 | 0.9063 |
| HD95 | 1.6707 |
| Boundary error (1–2 voxel layer near GT boundary) | 0.1101 |

### G2. Component statistics

From `PHASE_E25_STRUCTURAL_PIVOT_1A_MECHANISM_AUDIT.md` (full 125-subject set, A only):

| Size bin | n GT components | A detect rate | A component Dice (both-detected subset) |
|---|---:|---:|---:|
| 1–50 vox | 223 | 22.0% | 0.186–0.235 (varies slightly by paired-comparison subset) |
| 50–150 vox | 8 | 75.0% | 0.378 |
| 150–400 vox | 6 | 83.3% | 0.463–0.478 |
| 400–1000 vox | 15 | 100.0% | 0.824 |
| >1000 vox | 111 | 99.1% | 0.907 |

Total GT components: 363 (across 125 subjects). A's total missed components: 178 (mostly concentrated in the 1–50 voxel bin: 174/223 missed).

### G3. What proportion of the total error comes from FN/FP/boundary/missed/fragmented lesions?

Not decomposed into a single clean percentage breakdown anywhere in the existing work, but the qualitative picture is well established across the C6/E25 arc: **the deficit is diffuse, not dominated by one category** — `PHASE_E25_C62_SPATIAL_ERROR_ANALYSIS.md` explicitly found no single dominant structural category (slice depth, boundary distance, detection, size, fragmentation all checked, none dominant) for the A-vs-C6-2 comparison. The one component that *is* precisely quantified: small-lesion (1–50 voxel) components account for the large majority of missed components (174/178, i.e. ~98% of all of A's missed components are in the smallest size bin) — completely-missed small lesions are the single largest identifiable error category.

---

## H. D4-only details

### H1. Exact D4 architecture (★)

```text
dec3 output (B, 128, 16, 16, 16) → Conv3d(128→1, kernel_size=1) → Sigmoid → aux_probs3
```

Defined in `neuroscan_3d_v3.py`: `self.aux_head3 = nn.Sequential(nn.Conv3d(128, out_channels, kernel_size=1), nn.Sigmoid())`. Reads `dec3` **un-detached** (gradient flows back through the shared trunk, unlike the dormant `boundary_head` which reads a detached `dec1`).

### H2. D4 auxiliary loss weight (★)

$\lambda_{D4} = 0.9927$, fixed constant for all 30 epochs (calibrated once, pre-training, in `e25b_calibrate_deep_supervision.py`, targeting $\lambda \cdot L_{aux} \approx L_{seg}$ at initialization; not adapted during training).

### H3. Does the D4 head participate during inference?

No — training-only. `validate()` in `train_deep_sup.py` uses only `model(images)["probs"]` (the primary full-resolution `seg_head` output), never referencing `aux_probs3`/`aux_probs2`. Confirmed by the explicit unit test `test_aux_heads_isolated_from_primary_loss` and by direct code inspection of `validate()`.

### H4. What happens to D4 predictions after upsampling?

Nothing — D4/aux3 predictions are never upsampled or used at inference at all; they exist purely as a training-time auxiliary loss signal at native D/4 resolution, discarded after each training step. No saved D4 prediction volumes exist.

---

## I. Hardware / computational constraints

### I1. GPU (★)

**Correction to your assumption**: this session's `nvidia-smi` shows **NVIDIA GeForce RTX 5050 Laptop GPU, 8151 MiB (8GB)**, not an RTX 2050 4GB. All E24/E25 training (A, C6-2, C6-3, D4-only, D2-only, Both) ran on this RTX 5050. (Project memory also independently confirms this GPU from earlier environment setup — the 2050/4GB figure appears to be outdated or a different machine.)

### I2. Typical training time

A's actual run: **30 epochs, mean 134s/epoch, ~67 minutes total** (directly from `epoch_metrics.csv`). Peak GPU memory: ~4.97 GB (out of 8GB available — real headroom exists for larger batch size or a bigger model, not currently exploited). D4-only/D2-only/Both are architecturally near-identical (194 extra params) so their per-epoch time is `unknown but expected to be ~equal to A's within noise` — not directly compared this session.

### I3. Maximum experiments realistically possible

At ~67 min/run, sequentially on one GPU: roughly **1 full 30-epoch run per ~1.1 hours**, so realistically **~6–10 single-seed runs/day** if run back-to-back with no other GPU usage, or fewer if multi-seed replication is required per candidate (3 seeds × candidate = ~3.3 hours/candidate). This is a meaningfully higher budget than the "3/day, 1/day, 5/week" examples in your question — the constraint is milder than assumed, likely because the actual GPU (RTX 5050) is more capable than the RTX 2050 you were expecting.

---

## J. Existing experiments already trained

Beyond A / C6-2 / C6-3 / Both / D4-only / D2-only, from the E1–E26 arc (this list reconstructed from `experiments/` directory contents + prior phase reports, not exhaustively re-verified line-by-line this session):

```text
exp00/exp00b baseline convergence (seeds 0,1,2) → establishes the frozen UNet3D reproducibility baseline, retained as the architecture-freeze reference
exp_c0 weight ablation (focal_weight/evidential_weight sweep: 0.2/0.8, 0.5/0.5, 0.8/0.2) → 0.5/0.5 retained (best), documented in project memory as part of C0
exp_c1 ABO (active margin/gamma/r_target sweeps, seeds 0-2) → frozen null result, ABO mechanism did not improve Dice (see abo_frozen_lessons_learned memory)
E1-E18 (EGGO-M margin mechanism arc) → margin/SC-TAM family, various freeze/BN/decoder ablations (E18), all converged on "margin mechanism active but null on Dice" before the sign-bug discovery
E19-E21 → layerwise gradient, dec1 update decomposition, transport diagnostics — mechanistic, not new loss/training variants
C6-1 through C6-3 (E22-E25 arc) → SC-TAM / gated SC-TAM margin loss variants, both closed as null on outcome (0.9026-0.9030 vs A's 0.9063)
Deep Supervision "Both", D4-only, D2-only (E25, Structural Pivots 1/1A/1B) → the only variants so far to beat A (0.9091/0.9096/0.9080)
E26 response phenotype → analysis only, no training, found no independent predictor beyond size/baseline-Dice (failure per pre-registered gate)
```

Explicit answers to your named-loss checklist — **none of the following have ever been trained in this codebase**, per the grep-verified fact that `FocalTverskyLoss`+`EvidentialBetaLoss` (`HybridLoss`) is the only segmentation loss ever instantiated in any training script found:

```text
Plain BCE                → never trained
Dice + BCE (no focal)    → never trained
Plain Dice                → never trained
Focal loss (classification-style, not Tversky) → never trained
Tversky (non-focal)       → never trained
Focal Tversky              → TRAINED (this IS the production loss, all experiments) → Dice 0.9063 (A) → retained, it's the frozen baseline
Boundary loss (BCE on boundary head) → present as a SECONDARY term (μ=0.1) in every E24/E25 run, never tested as a primary/sole loss
Hausdorff loss             → never trained
clDice                     → never trained
Lovász                     → never trained
Deep supervision weights (different λ schedules) → only ONE calibration tested (λ≈1.0 at init, fixed); no sweep of λ values, no annealed λ, no alternative calibration target
Different crops            → never tested — there IS no cropping at all (whole-volume resize only), a completely untested axis
Lesion oversampling        → never tested — uniform random subject sampling only
Augmentation variants       → NEVER tested — zero augmentation exists in the current pipeline, a completely open, untested axis
Attention blocks            → never tested — no CBAM/attention module present in the actual trained architecture despite it being mentioned in early project docs
Different encoders          → never tested — only the one 3-stage plain-conv encoder
Different resolutions       → never tested — 64³ is the only resolution ever used; the effect of the ~3.75x downsampling on small-lesion loss has never been directly measured against a higher-resolution or patch-based alternative
```

---

## K. Literature / novelty constraints

### K1. Target journal (★)
`unknown` — not specified to me; please confirm.

### K2. Exact dataset (★)
**BraTS 2023 GLI (Glioma) Challenge, ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData**, 1251 subjects, using only the T2-FLAIR modality and a binarized (tumor-vs-background) collapse of the original 4-class (NCR/ET/ED/background) segmentation labels. This is a materially narrower task than the full BraTS challenge (which is multi-class, multi-modal) — worth stating explicitly in any paper, since "BraTS" alone implies a much broader benchmark than what's actually being solved here.

### K3. Bibliography
`unknown` — none provided to me; please paste titles/DOIs if you want me to cross-check against them before finalizing the roadmap.

### K4. Architecture-change latitude
`unknown` — not specified; the roadmap below assumes moderate latitude (loss/training/sampling changes preferred over full architecture replacement, given the project's own history of "no arch changes" as a stated design decision — see `brats_optimization_strategy` memory) but please confirm explicitly since it changes which roadmap items are viable.

### K5. What counts as "novel" to your professor
`unknown` — not specified; please paste their expectation if you have one, since this materially changes the roadmap's ranking (e.g. a sampling/resolution fix would likely be considered "not novel enough" for a "new algorithmic contribution" bar, but could be the single highest-Dice-impact change available given D1's finding).

---

## L. Code/data availability

### L1. Training script (★)
Yes — already audited directly from source this session: [experiments/exp_e12_eggo_m/e24/train_gate6.py](experiments/exp_e12_eggo_m/e24/train_gate6.py) (baseline A), [experiments/exp_e12_eggo_m/e25/train_deep_sup.py](experiments/exp_e12_eggo_m/e25/train_deep_sup.py) (D4/D2/Both), [Dataset/brats_dataset.py](Dataset/brats_dataset.py) (dataset/dataloader), [neuroscan_3d_fixed.py](neuroscan_3d_fixed.py)/[neuroscan_3d_v2.py](neuroscan_3d_v2.py)/[neuroscan_3d_v3.py](neuroscan_3d_v3.py) (model definitions).

### L2. Existing experiment configs/logs
Yes — `configs/brats.yaml` (base config, though note training code overrides several of its values — epochs, optimizer name); per-run `epoch_metrics.csv` exists for A, D4-only, D2-only, Both under each run's own directory in `experiments/exp_e12_eggo_m/e24/gate6_runs/` and `experiments/exp_e12_eggo_m/e25/deep_sup_runs/`.

### L3. Saved checkpoints for A and D4-only
**Yes** — `best.pth` for both, plus D2-only and Both, all confirmed loadable and already used for this session's inference-only analyses (E25 4-way comparison, E26 phenotype audit).

### L4. Saved per-voxel/per-component predictions for the validation set
Partial — per-subject, per-component structured records (size, detection status, component Dice, GT/pred confusion) for all 125 validation subjects across A/D4-only/D2-only/Both exist in `experiments/exp_e12_eggo_m/e25/deep_sup_4way_comparison_results/deep_sup_4way_comparison.json` and `experiments/exp_e12_eggo_m/e25/e26_response_phenotype_results/e26_response_phenotype.json`. Raw per-voxel prediction volumes (the actual 64³ probability/binary maps) are **not** saved to disk anywhere — they're computed on-the-fly during analysis scripts and discarded, so any new per-voxel question requires a fresh (cheap, inference-only, no-GPU-training-needed) forward pass over the checkpoints, not a full retraining.

---

## M. Strategic priority question

Not answered by me — this is explicitly the user's call, and your assumption of Option 3 (both must be competitive, e.g. ≥0.918 and novel) is noted but needs your explicit confirmation before the roadmap below is finalized, since it changes which of the ranked candidates are worth pursuing vs. discarding.

---

## The single most important audit finding

Sections C3/D1/E1 together surface something the entire E12–E26 diagnostic arc has been working *around* rather than *on*: **every model in this project has been trained on whole-volume-resized 64³ inputs with zero data augmentation and zero lesion-aware sampling**, and small lesions (1–50 voxels post-resize, which is a ~3.75×-downsampled representation of native resolution) are simultaneously (a) the dominant source of missed detections (174/178 of A's total missed components), (b) the size bin where deep supervision's benefit concentrates, and (c) a size bin where a lesion may already be only a few voxels wide **before** any model even sees it, because of the resize step alone. This reframes months of margin-loss (C1–C6) and deep-supervision (structural pivot) mechanism-hunting: those experiments have all been trying to make the model recover signal that the *preprocessing pipeline* may be discarding before training ever starts. A resolution/cropping/augmentation-focused experiment is untested, directly actionable, and targets the exact failure mode (small-lesion loss) that every other mechanism audit in this project has independently converged on.

---

## Files referenced

| File | Section(s) |
|---|---|
| `neuroscan_3d_fixed.py`, `neuroscan_3d_v2.py`, `neuroscan_3d_v3.py` | A1-A3, H1 |
| `Dataset/brats_dataset.py` | C1-C3, D1-D5, F1-F3 |
| `configs/brats.yaml` | B1-B6 |
| `experiments/exp_e12_eggo_m/e24/train_gate6.py` | B1-B6, G1 |
| `experiments/exp_e12_eggo_m/e25/train_deep_sup.py` | B2-B3, H1-H4 |
| `experiments/exp_e12_eggo_m/e24/gate6_runs/A_baseline_seed0/epoch_metrics.csv` | B5, G1, I2 |
| `PHASE_E25_STRUCTURAL_PIVOT_1A_MECHANISM_AUDIT.md` | G2 |
| `PHASE_E25_C62_SPATIAL_ERROR_ANALYSIS.md` | G3 |
| `PHASE_E25_STRUCTURAL_PIVOT_1B_4WAY_MECHANISM_COMPARISON.md` | G1 (metric caveat), L4 |
| `PHASE_E26_D4_RESPONSE_PHENOTYPE.md` | J |
