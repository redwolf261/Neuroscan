"""
Multi-slice supervision experiment: instead of always supervising the fixed
volume-center index (32 of 64), sample a RANDOM slice index per sample per
batch, and feed the model a k=9 window dynamically centered on that same
index (AdaptiveSliceSelector's new 'dynamic_window' mode). Over many epochs
this exposes the model to a representative sample of all ~22 lesion-positive
slices per volume (613 total across the 28 available samples, measured
directly - see the "how big can we get our dataset" analysis), each with a
context window that's guaranteed to actually contain the target, instead of
1 fixed slice per volume with a context window that included it only ~14%
of the time under the shipped 'adaptive' mechanism.

Uses the same realistic budget as reproduce_attempt.py (fresh MAE up to
60 epochs / patience 20, segmentation up to 80 epochs / patience 20) for a
direct, apples-to-apples comparison against the 'adaptive' (0.00421) and
'center_window' (0.00171) results already on record.

Validation uses a FIXED (seeded, not re-randomized every epoch) set of
random indices so epoch-to-epoch comparisons and early stopping are stable,
plus a separate fixed-index-32 pass for direct comparison with the other two
reproduction attempts, which only ever evaluated at index 32.

Usage (run from this directory):
    python reproduce_attempt_multislice.py
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

import common
from reproduce_attempt import train_mae_with_early_stopping

fm = common.fm


def sample_center_indices(batch_size, max_slices, device, generator=None):
    return torch.randint(0, max_slices, (batch_size,), device=device, generator=generator)


def gather_slice_labels(labels, center_indices):
    """labels: (B,1,D,H,W), center_indices: (B,) -> (B,1,H,W)"""
    return torch.stack([labels[b, :, center_indices[b], :, :] for b in range(labels.shape[0])], dim=0)


def train_segmentation_epoch_multislice(model, loader, criterion, evid_criterion, optimizer, scaler, device,
                                         max_slices=64):
    model.train()
    total_loss, total_dice = 0.0, 0.0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        B = images.shape[0]
        center_indices = sample_center_indices(B, max_slices, device)
        target_label = gather_slice_labels(labels, center_indices)

        with fm.autocast(enabled=fm.use_amp):
            out_dict = model(images, center_indices=center_indices)
            probs = out_dict["probs"]
            loss = criterion(probs, target_label)
            loss_evid = 0.0
            if fm.USALD_ENABLED and "alpha" in out_dict:
                loss_evid = evid_criterion(out_dict["alpha"], target_label)
                loss = loss + loss_evid

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pred_binary = (probs > 0.5).float()
        dice = 2 * (pred_binary * target_label).sum() / (pred_binary.sum() + target_label.sum() + 1e-7)
        total_loss += loss.item()
        total_dice += dice.item()

    n = len(loader)
    return total_loss / n, total_dice / n


def validate_multislice(model, loader, criterion, device, mode, max_slices=64, fixed_index=32, seed=1234):
    model.eval()
    total_loss, total_dice = 0.0, 0.0
    all_preds, all_labels = [], []
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            B = images.shape[0]
            if mode == 'random_seeded':
                center_indices = sample_center_indices(B, max_slices, device, generator=generator)
            elif mode == 'fixed_center':
                center_indices = torch.full((B,), fixed_index, dtype=torch.long, device=device)
            else:
                raise ValueError(mode)
            target_label = gather_slice_labels(labels, center_indices)

            out_dict = model(images, center_indices=center_indices)
            probs = out_dict["probs"]
            loss = criterion(probs, target_label)
            pred_binary = (probs > 0.5).float()
            dice = 2 * (pred_binary * target_label).sum() / (pred_binary.sum() + target_label.sum() + 1e-7)
            total_loss += loss.item()
            total_dice += dice.item()
            all_preds.append(pred_binary.cpu().numpy().flatten())
            all_labels.append(target_label.cpu().numpy().flatten())

    n = len(loader)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    precision = fm.precision_score(all_labels, all_preds, zero_division=0)
    recall = fm.recall_score(all_labels, all_preds, zero_division=0)
    f1 = fm.f1_score(all_labels, all_preds, zero_division=0)
    return {'loss': total_loss / n, 'dice': total_dice / n, 'precision': precision, 'recall': recall, 'f1': f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mae-epochs', type=int, default=60)
    parser.add_argument('--mae-patience', type=int, default=20)
    parser.add_argument('--seg-epochs', type=int, default=80)
    parser.add_argument('--seg-patience', type=int, default=20)
    args = parser.parse_args()

    device = fm.device
    out_dir = os.path.join(common.REPO_ROOT, "research_infra", "reproduction_attempt_multislice")

    t_start = time.time()

    # Phase 1: MAE pretraining, 'adaptive' mode - matching the as-shipped default and
    # the encoder-construction choice already used for the 'adaptive' baseline run, so
    # this experiment isolates the ONE new variable (multi-slice supervision) instead
    # of also changing MAE pretraining at the same time.
    encoder = train_mae_with_early_stopping(
        'adaptive', device, max_epochs=args.mae_epochs, patience=args.mae_patience,
        out_csv_path=os.path.join(out_dir, "mae_log.csv"),
    )

    # Phase 2: segmentation fine-tuning with dynamic_window selection + random per-batch
    # supervision targets.
    model = fm.HybridMiniSwin2D5_CBAM(k_slices=fm.K_SLICES, channels=fm.STAGE_CHANNELS).to(device)
    model.encoder.load_state_dict(encoder.state_dict())
    model.encoder.freeze_adaptive_selector()
    model.encoder.slice_selector.selection_mode = 'dynamic_window'

    optimizer = common.build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.seg_epochs)
    criterion = fm.HybridLoss(lambda1=0.5, lambda2=0.5)
    evid_criterion = fm.EvidentialBetaLoss(lambda_kl=fm.LAMBDA_EVIDENTIAL) if fm.USALD_ENABLED else None
    scaler = fm.GradScaler(enabled=fm.use_amp)

    best_val_dice_random = 0.0
    patience_counter = 0
    rows = []

    for epoch in range(1, args.seg_epochs + 1):
        t0 = time.time()
        train_loss, train_dice = train_segmentation_epoch_multislice(
            model, fm.train_loader, criterion, evid_criterion, optimizer, scaler, device
        )
        val_random = validate_multislice(model, fm.val_loader, criterion, device, mode='random_seeded')
        val_fixed = validate_multislice(model, fm.val_loader, criterion, device, mode='fixed_center')
        scheduler.step()
        elapsed = time.time() - t0

        row = {
            'epoch': epoch, 'train_loss': train_loss, 'train_dice': train_dice,
            'val_loss_random': val_random['loss'], 'val_dice_random': val_random['dice'],
            'val_precision_random': val_random['precision'], 'val_recall_random': val_random['recall'],
            'val_f1_random': val_random['f1'],
            'val_loss_fixed32': val_fixed['loss'], 'val_dice_fixed32': val_fixed['dice'],
            'val_precision_fixed32': val_fixed['precision'], 'val_recall_fixed32': val_fixed['recall'],
            'val_f1_fixed32': val_fixed['f1'],
            'epoch_time_sec': elapsed,
        }
        rows.append(row)
        print(f"[reproduce:SEG:multislice] epoch {epoch}/{args.seg_epochs} "
              f"train_loss={train_loss:.4f} val_dice(random)={val_random['dice']:.5f} "
              f"(best={best_val_dice_random:.5f}) val_dice(fixed@32)={val_fixed['dice']:.5f} "
              f"time={elapsed:.1f}s")

        if val_random['dice'] > best_val_dice_random:
            best_val_dice_random = val_random['dice']
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.seg_patience:
                print(f"[reproduce:SEG:multislice] early stopping at epoch {epoch} "
                      f"(best_val_dice_random={best_val_dice_random:.5f})")
                break

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "segmentation_log.csv"), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total_time = time.time() - t_start
    best_fixed = max(r['val_dice_fixed32'] for r in rows)
    print(f"\n[reproduce:multislice] DONE. Best val Dice (random)={best_val_dice_random:.5f}, "
          f"best val Dice (fixed@32)={best_fixed:.5f}. Total time = {total_time / 60:.1f} min.")


if __name__ == '__main__':
    main()
