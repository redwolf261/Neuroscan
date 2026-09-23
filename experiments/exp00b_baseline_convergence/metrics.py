"""
Segmentation metrics for Phase A.5 convergence study.

All metrics operate on binarized (threshold=0.5) predictions vs binary targets,
computed per-batch and averaged over the epoch (matching how Dice was tracked
in exp00, so numbers are comparable in kind, not by absolute value since the
val split composition changed — see Dataset/brats_dataset.py seed fix).
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def confusion_counts(pred_binary, target):
    tp = torch.sum(pred_binary * target).item()
    fp = torch.sum(pred_binary * (1 - target)).item()
    fn = torch.sum((1 - pred_binary) * target).item()
    tn = torch.sum((1 - pred_binary) * (1 - target)).item()
    return tp, fp, fn, tn


def dice_score(pred, target, smooth=1e-6):
    pred_binary = (pred > 0.5).float()
    intersection = torch.sum(pred_binary * target)
    union = torch.sum(pred_binary) + torch.sum(target)
    return ((2.0 * intersection + smooth) / (union + smooth + 1e-8)).item()


def iou_score(pred, target, smooth=1e-6):
    pred_binary = (pred > 0.5).float()
    intersection = torch.sum(pred_binary * target)
    union = torch.sum(pred_binary) + torch.sum(target) - intersection
    return ((intersection + smooth) / (union + smooth)).item()


def precision_recall_f1(pred, target, smooth=1e-6):
    pred_binary = (pred > 0.5).float()
    tp, fp, fn, _ = confusion_counts(pred_binary, target)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)
    return precision, recall, f1


def hausdorff_distance_95(pred, target):
    """
    HD95 computed per-volume in the batch, then averaged.
    Falls back to NaN for a volume if either mask is empty
    (HD is undefined when there's nothing to measure against).
    """
    pred_binary = (pred > 0.5).float().cpu().numpy()
    target_np = target.cpu().numpy()

    distances = []
    B = pred_binary.shape[0]
    for b in range(B):
        p = pred_binary[b, 0] > 0.5
        t = target_np[b, 0] > 0.5

        if not p.any() or not t.any():
            continue  # undefined; skip rather than penalize with an arbitrary constant

        dt_t = distance_transform_edt(~t)
        dt_p = distance_transform_edt(~p)

        surface_p_to_t = dt_t[p]
        surface_t_to_p = dt_p[t]

        all_dists = np.concatenate([surface_p_to_t, surface_t_to_p])
        hd95 = np.percentile(all_dists, 95)
        distances.append(hd95)

    if not distances:
        return float("nan")
    return float(np.mean(distances))


class MetricAccumulator:
    """Accumulates per-batch metrics and reports epoch averages."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.sums = {
            "loss": 0.0,
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
        self.hd95_values = []

    def update(self, loss, pred, target, compute_hd95=False):
        self.sums["loss"] += loss
        self.sums["dice"] += dice_score(pred, target)
        self.sums["iou"] += iou_score(pred, target)
        p, r, f1 = precision_recall_f1(pred, target)
        self.sums["precision"] += p
        self.sums["recall"] += r
        self.sums["f1"] += f1
        self.n += 1

        if compute_hd95:
            hd = hausdorff_distance_95(pred, target)
            if not np.isnan(hd):
                self.hd95_values.append(hd)

    def summary(self):
        out = {k: v / max(1, self.n) for k, v in self.sums.items()}
        out["hd95"] = float(np.mean(self.hd95_values)) if self.hd95_values else float("nan")
        return out
