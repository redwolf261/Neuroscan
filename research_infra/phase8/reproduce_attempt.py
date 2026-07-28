"""
Reproduction attempt: fresh MAE pretraining + segmentation fine-tuning with REAL
early stopping (matching final_model.py's own __main__ design), comparing:
  --mode adaptive       baseline mechanism as-shipped (empirically: only ~14% of
                         samples include the supervised center slice in their
                         selected 9 - see PHASE_1_CODEBASE_AUDIT.md and this
                         script's sibling check)
  --mode center_window  k=9 CONSECUTIVE slices centered on the supervised index,
                         no scorer involved - guarantees the model actually sees
                         the region it's asked to segment

This does NOT attempt to reproduce the README's 83.99% Dice claim - that used a
45-patient dataset (only 9 patients / 28 samples exist on this machine, confirmed
by direct search, no more data found anywhere on disk). This tests, on the data
actually available, whether the selector/supervision mismatch identified in
Phase 1 was a material, fixable blocker independent of dataset size.

Usage (run from this directory):
    python reproduce_attempt.py --mode adaptive
    python reproduce_attempt.py --mode center_window
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

import common

fm = common.fm


def train_mae_with_early_stopping(selection_mode, device, max_epochs=60, patience=20,
                                   out_csv_path=None):
    encoder = fm.HybridMiniSwin2D5_ResNetEncoder(
        k_slices=fm.K_SLICES, channels=fm.STAGE_CHANNELS, use_adaptive_selection=True
    ).to(device)
    if selection_mode != 'adaptive':
        sel = encoder.slice_selector
        sel.selection_mode = selection_mode
        if selection_mode == 'uniform':
            idx = np.linspace(0, sel.max_slices - 1, sel.k).round().astype(int)
            sel.register_buffer('uniform_indices', torch.tensor(idx, dtype=torch.long, device=device))
        elif selection_mode == 'center_window':
            c = sel.max_slices // 2
            start = max(0, c - sel.k // 2)
            end = min(sel.max_slices, start + sel.k)
            start = max(0, end - sel.k)
            idx = np.array(list(range(start, end)))
            sel.register_buffer('center_window_indices', torch.tensor(idx, dtype=torch.long, device=device))
        print(f"[reproduce] MAE selection_mode={selection_mode} fixed_indices={idx.tolist()}")

    mae_model = fm.MAE_2D5(encoder=encoder, mask_ratio=fm.MAE_MASK_RATIO).to(device)
    optimizer = torch.optim.AdamW(mae_model.parameters(), lr=fm.MAE_LEARNING_RATE, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler = fm.GradScaler(enabled=fm.use_amp)

    best_loss = float('inf')
    patience_counter = 0
    rows = []

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        loss = fm.train_mae_epoch(mae_model, fm.train_loader, optimizer, scaler, device, epoch=epoch)
        scheduler.step()
        elapsed = time.time() - t0
        rows.append({'epoch': epoch, 'mae_loss': loss, 'epoch_time_sec': elapsed})
        print(f"[reproduce:MAE:{selection_mode}] epoch {epoch}/{max_epochs} loss={loss:.4f} time={elapsed:.1f}s")

        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[reproduce:MAE:{selection_mode}] early stopping at epoch {epoch} (best_loss={best_loss:.4f})")
                break

    if out_csv_path:
        os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
        with open(out_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'mae_loss', 'epoch_time_sec'])
            writer.writeheader()
            writer.writerows(rows)

    return encoder


def train_segmentation_with_early_stopping(encoder, selection_mode, device, max_epochs=80,
                                            patience=20, out_csv_path=None):
    model = fm.HybridMiniSwin2D5_CBAM(k_slices=fm.K_SLICES, channels=fm.STAGE_CHANNELS).to(device)
    if selection_mode != 'adaptive':
        # Register the same fixed-index buffer as the source `encoder` BEFORE loading,
        # so state_dict keys line up exactly (these are constant buffers, not learned
        # parameters - the values are recomputed identically either way).
        sel = model.encoder.slice_selector
        sel.selection_mode = selection_mode
        if selection_mode == 'uniform':
            idx = np.linspace(0, sel.max_slices - 1, sel.k).round().astype(int)
            sel.register_buffer('uniform_indices', torch.tensor(idx, dtype=torch.long, device=device))
        elif selection_mode == 'center_window':
            c = sel.max_slices // 2
            start = max(0, c - sel.k // 2)
            end = min(sel.max_slices, start + sel.k)
            start = max(0, end - sel.k)
            idx = np.array(list(range(start, end)))
            sel.register_buffer('center_window_indices', torch.tensor(idx, dtype=torch.long, device=device))
    model.encoder.load_state_dict(encoder.state_dict())
    model.encoder.freeze_adaptive_selector()

    optimizer = common.build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = fm.HybridLoss(lambda1=0.5, lambda2=0.5)
    scaler = fm.GradScaler(enabled=fm.use_amp)

    best_val_dice = 0.0
    patience_counter = 0
    rows = []

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss, train_dice, _, _, _ = fm.train_segmentation_epoch(
            model, fm.train_loader, criterion, optimizer, scaler, device, epoch=epoch
        )
        val_metrics = fm.validate_segmentation(model, fm.val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        row = {
            'epoch': epoch, 'train_loss': train_loss, 'train_dice': train_dice,
            'val_loss': val_metrics['loss'], 'val_dice': val_metrics['dice'],
            'val_iou': common.dice_to_iou(val_metrics['dice']),
            'val_precision': val_metrics['precision'], 'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'], 'epoch_time_sec': elapsed,
        }
        rows.append(row)
        print(f"[reproduce:SEG:{selection_mode}] epoch {epoch}/{max_epochs} "
              f"train_loss={train_loss:.4f} val_dice={val_metrics['dice']:.5f} "
              f"(best={best_val_dice:.5f}) time={elapsed:.1f}s")

        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[reproduce:SEG:{selection_mode}] early stopping at epoch {epoch} "
                      f"(best_val_dice={best_val_dice:.5f})")
                break

    if out_csv_path:
        os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
        with open(out_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return best_val_dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['adaptive', 'center_window'], required=True)
    parser.add_argument('--mae-epochs', type=int, default=60)
    parser.add_argument('--mae-patience', type=int, default=20)
    parser.add_argument('--seg-epochs', type=int, default=80)
    parser.add_argument('--seg-patience', type=int, default=20)
    args = parser.parse_args()

    device = fm.device
    out_dir = os.path.join(common.REPO_ROOT, "research_infra", f"reproduction_attempt_{args.mode}")

    t_start = time.time()
    encoder = train_mae_with_early_stopping(
        args.mode, device, max_epochs=args.mae_epochs, patience=args.mae_patience,
        out_csv_path=os.path.join(out_dir, "mae_log.csv"),
    )
    best_dice = train_segmentation_with_early_stopping(
        encoder, args.mode, device, max_epochs=args.seg_epochs, patience=args.seg_patience,
        out_csv_path=os.path.join(out_dir, "segmentation_log.csv"),
    )
    total_time = time.time() - t_start
    print(f"\n[reproduce:{args.mode}] DONE. Best val Dice = {best_dice:.5f}. "
          f"Total wall-clock time = {total_time / 60:.1f} min.")


if __name__ == '__main__':
    main()
