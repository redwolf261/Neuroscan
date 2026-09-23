"""
Expected Calibration Error (ECE) for the evidential branch.

Uses the Beta mean mu = alpha / (alpha + beta) as the model's predicted
confidence that a voxel is tumor. Correctness is defined against the binary
ground truth using the SAME 0.5 threshold as Dice/IoU elsewhere, so this is
directly comparable to the segmentation metrics: "is the voxel classified
correctly, and how confident was the model when it did so?"

ECE cannot be a simple running mean per batch like Dice -- it requires
pooling (confidence, correctness) pairs across the whole validation set
before computing the per-bin accuracy-vs-confidence gap. A single 64^3
volume is 262,144 voxels; holding every voxel raw across ~16 val batches
of batch_size=8 would be ~34M floats, so this accumulates directly into
fixed-width histogram bins (counts + correct-counts + confidence-sum per
bin) instead of storing raw pairs.
"""

import torch


class ECEAccumulator:
    def __init__(self, n_bins=15):
        self.n_bins = n_bins
        self.bin_edges = torch.linspace(0, 1, n_bins + 1)
        self.bin_correct = torch.zeros(n_bins, dtype=torch.float64)
        self.bin_conf_sum = torch.zeros(n_bins, dtype=torch.float64)
        self.bin_count = torch.zeros(n_bins, dtype=torch.float64)

    def update(self, alpha, beta, target):
        """
        alpha, beta, target: tensors of identical shape (B, 1, D, H, W).
        Confidence is taken as max(mu, 1-mu) -- i.e. confidence in whatever
        class the model actually predicts (binary calibration convention),
        not just confidence in the positive class.
        """
        mu = (alpha / (alpha + beta)).detach().flatten().cpu()
        target_flat = target.detach().flatten().cpu()

        pred_positive = (mu > 0.5).float()
        confidence = torch.where(pred_positive == 1, mu, 1 - mu)
        correct = (pred_positive == target_flat).float()

        bin_idx = torch.bucketize(confidence, self.bin_edges[1:-1])

        for b in range(self.n_bins):
            mask = bin_idx == b
            cnt = mask.sum().item()
            if cnt == 0:
                continue
            self.bin_count[b] += cnt
            self.bin_correct[b] += correct[mask].sum().item()
            self.bin_conf_sum[b] += confidence[mask].sum().item()

    def compute(self):
        """
        Returns (ece, per_bin_dict). ECE = sum_b (n_b/N) * |acc_b - conf_b|.
        Empty bins are skipped (undefined accuracy/confidence).
        """
        total = self.bin_count.sum().item()
        if total == 0:
            return float("nan"), {}

        ece = 0.0
        per_bin = {}
        for b in range(self.n_bins):
            cnt = self.bin_count[b].item()
            if cnt == 0:
                continue
            acc_b = self.bin_correct[b].item() / cnt
            conf_b = self.bin_conf_sum[b].item() / cnt
            weight = cnt / total
            ece += weight * abs(acc_b - conf_b)
            per_bin[b] = {"count": cnt, "accuracy": acc_b, "confidence": conf_b}

        return ece, per_bin

    def reset(self):
        self.bin_correct.zero_()
        self.bin_conf_sum.zero_()
        self.bin_count.zero_()
