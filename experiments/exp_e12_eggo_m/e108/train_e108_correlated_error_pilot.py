"""
Phase E108: Correlation-Length x Lesion-Scale x Representation-Depth
Interaction Pilot (real training, reduced factorial per explicit scope
negotiation with user).

CONTEXT: Following the closure of E48-E107 (12 independent candidates,
converging on genuine evidence scarcity as the small-lesion root cause,
not a fixable weighting/regularization/attribution problem), a new
hypothesis was proposed from a broader literature sweep: does the
SPATIAL CORRELATION LENGTH of supervision error interact with lesion
scale and representation depth, specifically becoming most damaging
near the E3->bottleneck compression transition E90-E92 already
localized? Broad novelty claims in this space (spatially correlated
label noise, noise+compression interaction, e.g. AAAI 2026's LaT-IB)
were found occupied; the SPECIFIC scale x correlation x depth
interaction, and its relationship to this project's own E48 causal
signal, was not found in the literature and remains a live, narrow
gap candidate -- NOT YET a novelty claim, per the user's own framing.

SCOPE (negotiated down from the originally proposed 4x3x5x4=240-cell
factorial, which would require ~60-80+ hours of continuous GPU time --
explicitly flagged and reduced by mutual agreement to a REAL-TRAINING
pilot, not a proxy):
  Lesion scale (2 levels): SMALL vs LARGE (median native_size split,
    reusing E48's own established convention exactly).
  Error correlation (2 levels): INDEPENDENT (i.i.d. voxel-wise label
    flips) vs CORRELATED (spatially clustered flips via smoothed-noise
    thresholding), both calibrated to flip the SAME TOTAL VOXEL COUNT
    within the lesion mask -- so only the SPATIAL ORGANIZATION of the
    corruption changes, not its amount (the pre-declared control
    variable from the design).
  Representation depth (2 levels): supervision error injected into
    EITHER the main (full-resolution) segmentation loss OR the aux3
    deep-supervision loss (UNet3D_v5's own existing bottleneck-adjacent
    D4-resolution head, aux_probs3, at 16^3 -- a genuine, already-
    existing depth-differentiated supervision signal in this
    architecture, not an invented mechanism).
  Error magnitude: FIXED at 15% of within-mask lesion voxels flipped
    (0<->1), consistent across all 8 cells.

8 CONDITIONS TOTAL = 2 (scale) x 2 (correlation) x 2 (depth). Each
condition is a REAL, independent training run (15 epochs, matching
E100's own established short-training convention), NOT a zero-training
proxy -- per explicit user decision to prioritize direct performance
evidence over a cheaper diagnostic given the reduced (8-cell, not
240-cell) scope.

CORRUPTION MECHANISM: applied to the GROUND-TRUTH MASK fed to the
relevant loss term (main or aux3), NOT the input image -- this is
"supervision error" (label noise) in the sense the source literature
and this hypothesis actually concern, distinct from input-space
perturbation (which is what E99/E100's augmentation experiments tested).
Corruption is regenerated fresh each epoch (a fixed random corruption
mask would only test one noise realization; per-epoch resampling with a
fixed corruption RATE and RULE tests the mechanism, not one instance of it).

MEASURED: for each condition, per-subject Dice on the held-out CLEAN
(uncorrupted) validation set, stratified by validation subjects' OWN
lesion scale (small/large, same median split) -- the interaction of
interest is whether TRAINING with correlated (vs independent) label
noise disproportionately hurts SMALL-lesion validation Dice specifically
when the corruption targets the DEEPER (aux3/bottleneck-adjacent) head.

PRE-DECLARED INTERACTION STATISTIC (matching the user's own I = (D-C)-
(B-A) design, applied per depth condition):
  For depth d in {main, aux3}:
    A_d = mean small-lesion-val-Dice under LARGE-lesion-training +
          INDEPENDENT-noise cell (approximated here as: for the
          SMALL-lesion validation stratum, comparing INDEPENDENT vs
          CORRELATED training-noise conditions, matching training-scale
          to validation-scale stratification)
    I_d = (Dice_small,correlated - Dice_small,independent)
        - (Dice_large,correlated - Dice_large,independent)
  Kill condition: I_main ~ I_aux3 ~ 0 (no interaction at either depth).
  Interesting condition: |I_aux3| >> |I_main| (interaction specific to
  the deeper/bottleneck-adjacent supervision signal, matching the
  E3->bottleneck localization from E90-E92).

PRE-DECLARED KILL CONDITION: training instability (NaN/Inf, dice
collapse below 0.5 after epoch 5) kills the affected run immediately,
matching every prior condition's own established practice.
"""
import sys
import csv
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage as ndi
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402
import nibabel as nib  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp00b_baseline_convergence"))
from metrics import MetricAccumulator  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m"))
from train_eggo_m import set_seed  # noqa: E402

SEED = 0
CONFIG_PATH = project_root / "configs" / "brats.yaml"
OUT_DIR = Path(__file__).parent
N_EPOCHS = 15
ERROR_RATE = 0.15  # fixed error magnitude: fraction of within-lesion-mask voxels flipped
CLUSTER_SIGMA = 4.0  # smoothing scale for CORRELATED noise field (voxels)


def corrupt_mask(mask_np, correlation, rng, error_rate):
    """Flips `error_rate` fraction of FOREGROUND (lesion) voxels to
    background, either INDEPENDENTLY (i.i.d. random selection) or in a
    spatially CORRELATED pattern (thresholded smoothed noise field),
    both calibrated to flip the SAME TOTAL VOXEL COUNT (the controlled
    variable -- only spatial organization changes, not corruption amount).
    Background voxels are left untouched in both conditions (this
    experiment tests FALSE-NEGATIVE-style supervision error within the
    lesion, matching the "small-lesion suppression" mechanism in the
    hypothesis -- not general label noise everywhere)."""
    fg_mask = mask_np > 0.5
    n_fg = int(fg_mask.sum())
    n_flip = int(round(error_rate * n_fg))
    if n_flip == 0:
        return mask_np.copy()

    corrupted = mask_np.copy()
    fg_idx = np.argwhere(fg_mask)

    if correlation == "independent":
        chosen = rng.choice(len(fg_idx), size=n_flip, replace=False)
        flip_idx = fg_idx[chosen]
        corrupted[tuple(flip_idx.T)] = 0.0
    elif correlation == "correlated":
        # Smoothed random field, restricted to the foreground mask, thresholded
        # to select the TOP n_flip voxels by field value -- a spatially
        # clustered (not scattered) subset of the same size as the independent case.
        field = ndi.gaussian_filter(rng.random(mask_np.shape), sigma=CLUSTER_SIGMA)
        field_fg = np.where(fg_mask, field, -np.inf)
        flat_order = np.argsort(-field_fg.ravel())[:n_flip]
        flip_coords = np.unravel_index(flat_order, mask_np.shape)
        corrupted[flip_coords] = 0.0
    else:
        raise ValueError(f"Unknown correlation mode: {correlation}")

    return corrupted


class E108Experiment:
    def __init__(self, config_path, exp_dir, seed, run_name, depth, correlation,
                 lesion_scale_filter, num_workers=None):
        self.condition_name = run_name
        self.seed = seed
        self.depth = depth  # "main" or "aux3"
        self.correlation = correlation  # "independent" or "correlated"
        self.lesion_scale_filter = lesion_scale_filter  # "small", "large", or None (train on all, filter only at eval)
        set_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.exp_dir = Path(exp_dir) / run_name
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        dataset_root = self.config["dataset"]["root_dir"]
        if not Path(dataset_root).is_absolute():
            dataset_root = project_root / dataset_root
        self.config["dataset"]["root_dir"] = str(dataset_root)

        self.device = torch.device(self.config.get("hardware", {}).get("device", "cpu"))
        self.epoch = 0

        self.model = UNet3D_v5(
            in_channels=self.config["model"]["in_channels"],
            out_channels=self.config["model"]["out_channels"],
        ).to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=N_EPOCHS, eta_min=1e-6)

        self.train_dataset_full = BraTSDataset(
            root_dir=self.config["dataset"]["root_dir"], split="train",
            val_split=self.config["dataset"]["val_split"], target_shape=(64, 64, 64), normalize=True,
        )
        self.val_dataset_full = BraTSDataset(
            root_dir=self.config["dataset"]["root_dir"], split="val",
            val_split=self.config["dataset"]["val_split"], target_shape=(64, 64, 64), normalize=True,
        )

        # Compute native_size for every train subject (for lesion-scale filtering
        # of TRAINING data, per this condition's scale factor) using the same
        # convention as E48 (native-resolution seg voxel count).
        self.train_native_sizes = self._compute_native_sizes(self.train_dataset_full, cache_name="train")
        median_size = float(np.median(list(self.train_native_sizes.values())))
        self.median_size = median_size

        self.train_indices = self._filter_indices(self.train_dataset_full, self.train_native_sizes,
                                                    lesion_scale_filter, median_size)

        batch_size = self.config["training"]["batch_size"]
        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(self.train_dataset_full, self.train_indices),
            batch_size=batch_size, shuffle=True,
            num_workers=(num_workers if num_workers is not None else self.config["training"].get("num_workers", 0)),
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset_full, batch_size=batch_size, shuffle=False,
            num_workers=(num_workers if num_workers is not None else self.config["training"].get("num_workers", 0)),
        )

        # Precompute validation subjects' native_size for stratified eval.
        self.val_native_sizes = self._compute_native_sizes(self.val_dataset_full, cache_name="val")

        print(f"[{self.condition_name} seed{seed}] depth={depth} correlation={correlation} "
              f"lesion_scale_filter={lesion_scale_filter}")
        print(f"[{self.condition_name}] Training subjects (after scale filter): {len(self.train_indices)} "
              f"of {len(self.train_dataset_full)}")
        print(f"[{self.condition_name}] Validation subjects: {len(self.val_dataset_full)}")

        self.f_metrics = open(self.exp_dir / "epoch_metrics.csv", "w", newline="")
        self.w_metrics = csv.writer(self.f_metrics)
        self.w_metrics.writerow([
            "epoch", "train_loss", "val_dice_all", "val_dice_small", "val_dice_large", "epoch_time_sec",
        ])

    @staticmethod
    def _compute_native_sizes(dataset, cache_name=None):
        # Cache to disk keyed by split, since this is an expensive I/O-bound
        # pass (one NIfTI load per subject) that would otherwise be repeated
        # identically on every one of the 8 sweep runs.
        cache_path = None
        if cache_name is not None:
            cache_dir = Path(__file__).parent
            cache_path = cache_dir / f"native_sizes_{cache_name}.json"
            if cache_path.exists():
                with open(cache_path) as f:
                    return json.load(f)
        sizes = {}
        for i in range(len(dataset)):
            subject_dir = dataset.subject_dirs[i]
            subject_id = Path(subject_dir).name
            seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
            seg_data = nib.load(str(seg_path)).get_fdata()
            sizes[subject_id] = int((seg_data > 0).sum())
        if cache_path is not None:
            with open(cache_path, "w") as f:
                json.dump(sizes, f)
        return sizes

    @staticmethod
    def _filter_indices(dataset, native_sizes, scale_filter, median_size):
        if scale_filter is None:
            return list(range(len(dataset)))
        indices = []
        for i in range(len(dataset)):
            subject_id = Path(dataset.subject_dirs[i]).name
            sz = native_sizes[subject_id]
            if scale_filter == "small" and sz <= median_size:
                indices.append(i)
            elif scale_filter == "large" and sz > median_size:
                indices.append(i)
        return indices

    def train_epoch(self):
        self.model.train()
        acc = MetricAccumulator()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"[{self.condition_name}] Epoch {self.epoch+1} [Train]")
        for images, masks, _ in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Corrupt the GROUND-TRUTH MASK used for the loss (not the input image) --
            # this is "supervision error", regenerated fresh each batch/epoch.
            masks_np = masks.squeeze(1).cpu().numpy()
            corrupted_np = np.stack([
                corrupt_mask(masks_np[b], self.correlation, self.rng, ERROR_RATE)
                for b in range(masks_np.shape[0])
            ])
            corrupted_masks = torch.from_numpy(corrupted_np).unsqueeze(1).to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(images)
            probs = out["probs"]
            aux_probs3 = out["aux_probs3"]

            def dice_ce(p, t, eps=1e-7):
                p_flat, t_flat = p.reshape(p.shape[0], -1), t.reshape(t.shape[0], -1)
                inter = (p_flat * t_flat).sum(dim=1)
                denom = p_flat.sum(dim=1) + t_flat.sum(dim=1)
                dice_loss = (1.0 - (2.0 * inter / denom.clamp_min(eps))).mean()
                ce = F.binary_cross_entropy(p.clamp(eps, 1 - eps), t, reduction="mean")
                return dice_loss + ce

            if self.depth == "main":
                # Corruption applied to the MAIN (full-resolution) loss only;
                # aux3 sees the CLEAN mask (downsampled), isolating the effect
                # to the shallow/full-resolution supervision pathway.
                loss_main = dice_ce(probs, corrupted_masks)
                mask_d4_clean = F.avg_pool3d(masks, kernel_size=4, stride=4)
                loss_aux3 = dice_ce(aux_probs3, mask_d4_clean)
            else:  # "aux3"
                # Corruption applied to the aux3 (bottleneck-adjacent, D4-resolution)
                # loss only; main sees the CLEAN mask, isolating the effect to the
                # deeper supervision pathway.
                loss_main = dice_ce(probs, masks)
                mask_d4_corrupted = F.avg_pool3d(corrupted_masks, kernel_size=4, stride=4)
                loss_aux3 = dice_ce(aux_probs3, mask_d4_corrupted)

            loss = loss_main + 0.9927 * loss_aux3  # lambda_ds3, matching this project's established value
            if not torch.isfinite(loss):
                raise RuntimeError(f"[{self.condition_name}] NaN/Inf loss at epoch {self.epoch}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            acc.update(loss.item(), probs, masks, compute_hd95=False)  # eval-style Dice vs CLEAN mask, for monitoring only
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": loss.item()})

        return total_loss / max(1, n_batches)

    def validate(self):
        """Validation is ALWAYS against the CLEAN (uncorrupted) ground truth --
        the corruption is a TRAINING-time supervision-error simulation only,
        never applied at evaluation."""
        self.model.eval()
        dice_by_subject = {}
        idx_counter = 0
        with torch.no_grad():
            for images, masks, subject_ids in tqdm(self.val_loader, desc=f"[{self.condition_name}] Epoch {self.epoch+1} [Val]"):
                images = images.to(self.device)
                masks_np = masks.numpy()
                out = self.model(images)
                probs = out["probs"].cpu().numpy()
                for b in range(images.shape[0]):
                    pred_bin = (probs[b, 0] >= 0.5).astype(np.float32)
                    gt_bin = (masks_np[b, 0] > 0.5).astype(np.float32)
                    denom = pred_bin.sum() + gt_bin.sum()
                    dice = (2 * (pred_bin * gt_bin).sum() / denom) if denom > 0 else 1.0
                    sid = subject_ids[b]
                    dice_by_subject[sid] = float(dice)

        small_ids = [sid for sid, sz in self.val_native_sizes.items() if sz <= self.median_size and sid in dice_by_subject]
        large_ids = [sid for sid, sz in self.val_native_sizes.items() if sz > self.median_size and sid in dice_by_subject]

        dice_all = float(np.mean(list(dice_by_subject.values())))
        dice_small = float(np.mean([dice_by_subject[s] for s in small_ids])) if small_ids else float("nan")
        dice_large = float(np.mean([dice_by_subject[s] for s in large_ids])) if large_ids else float("nan")
        return dice_all, dice_small, dice_large, dice_by_subject

    def train(self, epochs):
        best_dice = 0.0
        for epoch in range(epochs):
            self.epoch = epoch
            t0 = time.time()
            train_loss = self.train_epoch()
            dice_all, dice_small, dice_large, _ = self.validate()
            self.scheduler.step()
            epoch_time = time.time() - t0

            if epoch >= 5 and dice_all < 0.5:
                print(f"\n[{self.condition_name}] INSTABILITY: val_dice={dice_all:.4f} < 0.5 after epoch 5.", flush=True)
                self.f_metrics.close()
                raise RuntimeError(f"Training instability at epoch {epoch}: val_dice={dice_all:.4f}")

            print(f"[{self.condition_name}] Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) | "
                  f"loss={train_loss:.4f} dice_all={dice_all:.4f} dice_small={dice_small:.4f} dice_large={dice_large:.4f}")
            self.w_metrics.writerow([epoch, train_loss, dice_all, dice_small, dice_large, epoch_time])
            self.f_metrics.flush()
            best_dice = max(best_dice, dice_all)

        self.f_metrics.close()
        print(f"\n[{self.condition_name}] Done. Best val_dice_all: {best_dice:.4f}")
        return best_dice


def main(run_name, seed, depth, correlation, lesion_scale_filter):
    exp = E108Experiment(
        config_path=str(CONFIG_PATH), exp_dir=str(OUT_DIR / "runs"), seed=seed,
        run_name=run_name, depth=depth, correlation=correlation,
        lesion_scale_filter=lesion_scale_filter, num_workers=None,
    )
    exp.train(epochs=N_EPOCHS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--depth", type=str, choices=["main", "aux3"], required=True)
    parser.add_argument("--correlation", type=str, choices=["independent", "correlated"], required=True)
    parser.add_argument("--lesion_scale_filter", type=str, choices=["small", "large"], required=True)
    args = parser.parse_args()
    main(args.run_name, args.seed, args.depth, args.correlation, args.lesion_scale_filter)
