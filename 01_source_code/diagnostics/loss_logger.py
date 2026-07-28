"""
training_metrics.csv writer.

Tracks individual loss components (Dice, FocalTversky, Hybrid, Evidential,
Total) per batch, plus learning rates and the running Dice metric. This is
pure logging: it reads values that `train_segmentation_epoch` /
`train_mae_epoch` already compute, and writes them to disk. It does not
change what is computed, only what is recorded.

Instantiate and call this ONLY when config.ENABLE_LOSS_DIAGNOSTICS is True —
the caller in final_model.py is responsible for that gating, so this module
has no internal flag checks and imposes zero cost when unused (unimported
code paths cost nothing).
"""

import os
import csv
from collections import defaultdict
from datetime import datetime

import numpy as np


TRAINING_METRICS_HEADERS = [
    'epoch', 'iteration', 'timestamp', 'phase', 'patient_ids',
    'loss_dice', 'loss_focal_tversky', 'loss_hybrid',
    'loss_evidential', 'loss_total', 'dice_metric',
    'learning_rate_encoder', 'learning_rate_decoder',
]


class LossLogger:
    """Writes one row per batch to training_metrics.csv."""

    def __init__(self, log_dir, phase_name="segmentation"):
        self.log_dir = log_dir
        self.phase_name = phase_name
        os.makedirs(log_dir, exist_ok=True)

        self.csv_path = os.path.join(log_dir, "training_metrics.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(TRAINING_METRICS_HEADERS)

        self.batch_logs = []
        self.epoch_stats = defaultdict(list)

    def log_batch(self, epoch, batch_idx, losses_dict, dice_metric=None,
                  learning_rates=None, patient_ids=None):
        """
        Args:
            epoch: current epoch number
            batch_idx: batch index within the epoch
            losses_dict: dict with any of 'dice','focal_tversky','hybrid','evidential','total'
            dice_metric: the segmentation Dice metric (not the loss) for this batch, if available
            learning_rates: dict with 'encoder'/'decoder' keys
            patient_ids: list/tuple of patient identifiers for this batch (from the 'case'
                         or 'patient_id' batch field), or None if not threaded through yet
        """
        timestamp = datetime.now().isoformat()

        def f(key):
            v = losses_dict.get(key, 0.0)
            return float(v.item()) if hasattr(v, 'item') else float(v)

        row = {
            'epoch': epoch,
            'iteration': batch_idx,
            'timestamp': timestamp,
            'phase': self.phase_name,
            'patient_ids': ';'.join(str(p) for p in patient_ids) if patient_ids else '',
            'loss_dice': f'{f("dice"):.6f}',
            'loss_focal_tversky': f'{f("focal_tversky"):.6f}',
            'loss_hybrid': f'{f("hybrid"):.6f}',
            'loss_evidential': f'{f("evidential"):.6f}',
            'loss_total': f'{f("total"):.6f}',
            'dice_metric': f'{float(dice_metric):.6f}' if dice_metric is not None else '',
            'learning_rate_encoder': f'{float(learning_rates.get("encoder", 0.0)):.2e}' if learning_rates else '',
            'learning_rate_decoder': f'{float(learning_rates.get("decoder", 0.0)):.2e}' if learning_rates else '',
        }
        self.batch_logs.append(row)
        self.epoch_stats[epoch].append({
            'dice': f('dice'), 'ft': f('focal_tversky'), 'hybrid': f('hybrid'),
            'evid': f('evidential'), 'total': f('total'),
        })

    def flush_epoch(self):
        if not self.batch_logs:
            return
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=TRAINING_METRICS_HEADERS)
            writer.writerows(self.batch_logs)
        self.batch_logs.clear()

    def get_epoch_summary(self, epoch):
        logs = self.epoch_stats.get(epoch)
        if not logs:
            return None
        out = {}
        for key in ('dice', 'ft', 'hybrid', 'evid', 'total'):
            vals = [l[key] for l in logs]
            out[f'mean_{key}'] = float(np.mean(vals))
            out[f'std_{key}'] = float(np.std(vals))
        return out


def extract_loss_components(criterion, model_output, labels):
    """
    Break down HybridLoss(model_output, labels) into its Dice and
    FocalTversky components without changing what gradient the training
    loop actually uses (the caller still calls criterion(...) separately
    for the real backward pass; this is a read-only decomposition for
    logging purposes and is numerically identical to what's inside
    HybridLoss.forward).
    """
    loss_dice = criterion.dice_loss(model_output, labels)
    loss_ft = criterion.focal_tversky(model_output, labels)
    loss_hybrid = criterion.lambda1 * loss_dice + criterion.lambda2 * loss_ft
    return {
        'dice': loss_dice.item() if hasattr(loss_dice, 'item') else float(loss_dice),
        'focal_tversky': loss_ft.item() if hasattr(loss_ft, 'item') else float(loss_ft),
        'hybrid': loss_hybrid.item() if hasattr(loss_hybrid, 'item') else float(loss_hybrid),
    }


def extract_evidential_loss(evid_criterion, alpha, labels):
    loss_evid = evid_criterion(alpha, labels)
    return loss_evid.item() if hasattr(loss_evid, 'item') else float(loss_evid)
