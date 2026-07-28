"""
Phase 8 shared experiment infrastructure.

This module does NOT reimplement NeuroScan. It imports the real, unmodified
(except for two additive, default-preserving hooks documented in
research_infra/PHASE_1_CODEBASE_AUDIT.md's spirit and inline in final_model.py)
training/validation functions and reuses them directly, so that baseline,
sampling-prototype, and optimization-prototype experiments all run through
the identical underlying model/loss/optimizer code - the only thing that
varies between experiments is the one controlled variable each step
introduces (selection_mode, or a gradient_scale_hook).

All three experiments share:
  - the same MAE-pretrained encoder checkpoint (from the Phase 7 run)
  - the same fixed epoch budget, with early stopping DISABLED (patience is not
    used here at all - every experiment runs the full, identical epoch count,
    so no experiment can "get lucky" and stop earlier/later than another)
  - the same data split (deterministic, unchanged from final_model.py)
  - the same optimizer configuration (two param groups: encoder+cbam @ LR,
    decoder @ LR, both taken from final_model.py's own constants)
"""

import os
import sys
import time
import csv
import json

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(REPO_ROOT, "01_source_code", "models")
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

import final_model as fm  # noqa: E402  (importing runs fm's module-level dataset scan/setup)


# The Phase 7 run's MAE-pretrained encoder, reused as-is by all three Phase 8 experiments
# so the only difference between them is the one controlled variable each introduces.
SHARED_MAE_CHECKPOINT = os.path.join(
    "C:\\Users\\Rivan\\MyDrive_Local\\OptimalModel_FrozenSelector_20260727_012405",
    "mae_pretraining", "mae_best.pth"
)

METRICS_HEADERS = [
    'epoch', 'train_loss', 'train_dice', 'val_loss', 'val_dice', 'val_iou',
    'val_precision', 'val_recall', 'val_f1', 'epoch_time_sec',
    'gpu_memory_mb_peak', 'grad_norm_mean', 'grad_norm_std', 'lr_encoder', 'lr_decoder',
]


def dice_to_iou(dice):
    """IoU = Dice / (2 - Dice), valid whenever both are computed from the same TP/FP/FN."""
    if dice is None:
        return None
    denom = 2.0 - dice
    return dice / denom if denom > 1e-8 else 0.0


def reset_gpu_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def gpu_memory_mb_peak():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def try_estimate_flops(model, sample_input):
    """Best-effort FLOPs estimate using torch's built-in FlopCounterMode. Returns None
    (honestly) if unavailable in this torch build, rather than adding a new dependency."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        model.eval()
        with torch.no_grad():
            with FlopCounterMode(display=False) as fcm:
                model(sample_input)
        model.train()
        return fcm.get_total_flops()
    except Exception as e:
        print(f"[phase8] FLOPs estimation unavailable ({e}); reporting None (not fabricated).")
        return None


def build_segmentation_model(device, selection_mode='adaptive',
                              mae_checkpoint=SHARED_MAE_CHECKPOINT):
    """
    Builds HybridMiniSwin2D5_CBAM exactly as final_model.py's __main__ does (same
    channels, same k_slices), loads the shared MAE-pretrained encoder, freezes the
    adaptive selector (matching baseline behavior exactly), then - only for the
    sampling-prototype experiment - flips the already-loaded selector's selection_mode
    to 'uniform'. This does not reload or change any other weights.
    """
    model = fm.HybridMiniSwin2D5_CBAM(k_slices=fm.K_SLICES, channels=fm.STAGE_CHANNELS).to(device)

    checkpoint = torch.load(mae_checkpoint, map_location=device, weights_only=False)
    model.encoder.load_state_dict(checkpoint['encoder_state_dict'])

    # Matches baseline's own documented behavior exactly (see final_model.py __main__,
    # "CRITICAL FIX: Freeze adaptive selector to preserve MAE-learned knowledge").
    model.encoder.freeze_adaptive_selector()

    if selection_mode == 'uniform':
        sel = model.encoder.slice_selector
        sel.selection_mode = 'uniform'
        uniform_idx = np.linspace(0, sel.max_slices - 1, sel.k).round().astype(int)
        sel.register_buffer('uniform_indices', torch.tensor(uniform_idx, dtype=torch.long, device=device))
        print(f"[phase8] Sampling prototype: fixed uniform indices = {uniform_idx.tolist()}")
    elif selection_mode == 'center_window':
        sel = model.encoder.slice_selector
        sel.selection_mode = 'center_window'
        center = sel.max_slices // 2
        start = max(0, center - sel.k // 2)
        end = min(sel.max_slices, start + sel.k)
        start = max(0, end - sel.k)
        window_idx = np.array(list(range(start, end)))
        sel.register_buffer('center_window_indices', torch.tensor(window_idx, dtype=torch.long, device=device))
        print(f"[phase8] Center-window fix: fixed indices = {window_idx.tolist()}")
    elif selection_mode != 'adaptive':
        raise ValueError(f"Unknown selection_mode: {selection_mode}")

    return model


def build_optimizer(model):
    """Identical to final_model.py __main__'s segmentation optimizer construction."""
    encoder_params = list(model.encoder.parameters()) + list(model.cbam.parameters())
    decoder_params = list(model.decoder.parameters())
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': fm.LEARNING_RATE_ENCODER},
        {'params': decoder_params, 'lr': fm.LEARNING_RATE_DECODER},
    ], weight_decay=0.01)
    return optimizer


EVIDENTIAL_HEAD_SUBMODULES = ['anatomy_head', 'pathology_head', 'noise_head', 'causal_shared']


def make_gradient_scale_hook(scale_factor):
    """
    Phase 8 Step 3 optimization prototype. Justified directly by
    research_infra/PHASE_7_HYPOTHESIS_VALIDATION.md Q4/Q7 (Candidate 1): the
    evidential/causal heads showed 2-5x higher per-parameter gradient magnitude than
    the primary probability head (0.0024/0.0023/0.0018 vs 0.0011 for pathology/anatomy/
    noise heads' weights vs prob_head.weight; 0.0057/0.0050/0.0043 vs 0.0020 for their
    biases vs prob_head.bias). scale_factor=1/3 brings these heads' gradient magnitude
    down toward rough parity with the primary segmentation head, the single most
    directly-justified, minimal change identified in Phase 7 - not gradient clipping,
    which Phase 7 explicitly flagged as the weaker-justified candidate.

    Only decoder.anatomy_head / decoder.pathology_head / decoder.noise_head /
    decoder.causal_shared / decoder.causal_weights are touched. Everything else
    (encoder, CBAM, decoder.prob_head, decoder.up_blocks/skip_convs) is unaffected.
    """
    def hook(model):
        decoder = model.decoder
        for attr in EVIDENTIAL_HEAD_SUBMODULES:
            module = getattr(decoder, attr, None)
            if module is None:
                continue
            for p in module.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale_factor)
        causal_weights = getattr(decoder, 'causal_weights', None)
        if causal_weights is not None and causal_weights.grad is not None:
            causal_weights.grad.mul_(scale_factor)
    return hook


def run_fixed_epoch_experiment(experiment_name, epochs, device, selection_mode='adaptive',
                                gradient_scale_hook=None, out_csv_path=None):
    """
    Runs `epochs` full segmentation-training epochs with NO early stopping, using the
    real, unmodified fm.train_segmentation_epoch / fm.validate_segmentation functions.
    Logs per-epoch metrics to out_csv_path in the schema required by Phase 8's
    baseline/sampling/optimization_metrics.csv deliverables.
    """
    print(f"\n{'=' * 80}\nPHASE 8 EXPERIMENT: {experiment_name}\n{'=' * 80}")

    model = build_segmentation_model(device, selection_mode=selection_mode)
    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = fm.HybridLoss(lambda1=0.5, lambda2=0.5)
    scaler = fm.GradScaler(enabled=fm.use_amp)

    rows = []
    for epoch in range(1, epochs + 1):
        reset_gpu_memory_stats()
        t0 = time.time()

        train_loss, train_dice, _, _, _ = fm.train_segmentation_epoch(
            model, fm.train_loader, criterion, optimizer, scaler, device,
            teacher_model=None, epoch=epoch, tau_pl=0.5,
            gradient_scale_hook=gradient_scale_hook,
        )
        val_metrics = fm.validate_segmentation(model, fm.val_loader, criterion, device)
        scheduler.step()

        epoch_time = time.time() - t0
        peak_mem = gpu_memory_mb_peak()

        # Gradient variance for this epoch, read back from the diagnostic CSV that
        # ENABLE_GRADIENT_DIAGNOSTICS just wrote (reuses existing, already-verified
        # instrumentation rather than adding a parallel measurement path).
        grad_mean, grad_std = _read_epoch_gradient_stats(fm.SEGMENTATION_DIR, epoch)

        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_dice': train_dice,
            'val_loss': val_metrics['loss'],
            'val_dice': val_metrics['dice'],
            'val_iou': dice_to_iou(val_metrics['dice']),
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'epoch_time_sec': epoch_time,
            'gpu_memory_mb_peak': peak_mem,
            'grad_norm_mean': grad_mean,
            'grad_norm_std': grad_std,
            'lr_encoder': optimizer.param_groups[0]['lr'],
            'lr_decoder': optimizer.param_groups[1]['lr'],
        }
        rows.append(row)
        print(f"[{experiment_name}] epoch {epoch}/{epochs}  "
              f"train_loss={train_loss:.4f} val_dice={val_metrics['dice']:.4f} "
              f"time={epoch_time:.1f}s peak_mem={peak_mem:.0f}MB "
              f"grad_norm={grad_mean:.4f}+-{grad_std:.4f}")

    if out_csv_path:
        out_dir = os.path.dirname(out_csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=METRICS_HEADERS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[phase8] wrote {out_csv_path}")

    return rows, model


def _read_epoch_gradient_stats(segmentation_dir, epoch):
    """Reads gradient_metrics.csv (written by diagnostics/gradient_logger.py, which
    Phase 8 runs keep enabled) and returns (mean, std) of total_grad_norm for the
    given epoch's batches."""
    path = os.path.join(segmentation_dir, "gradient_metrics.csv")
    if not os.path.exists(path):
        return float('nan'), float('nan')
    norms = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row['epoch']) == epoch:
                try:
                    norms.append(float(row['total_grad_norm']))
                except ValueError:
                    pass
    if not norms:
        return float('nan'), float('nan')
    arr = np.array(norms)
    return float(arr.mean()), float(arr.std())
