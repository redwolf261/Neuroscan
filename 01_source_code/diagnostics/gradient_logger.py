"""
gradient_metrics.csv writer.

Measures gradient magnitudes after the backward pass. This is read-only
w.r.t. training: it inspects `.grad` on existing parameters, it does not
create new parameters, change the loss, or alter the optimizer step.

IMPORTANT (AMP correctness): final_model.py uses `torch.cuda.amp.GradScaler`.
`scaler.scale(loss).backward()` multiplies the loss (and therefore every
gradient) by a dynamic scale factor to prevent fp16 underflow. If you read
`.grad` immediately after that `.backward()` without calling
`scaler.unscale_(optimizer)` first, the norms you log are inflated by the
scale factor (which itself changes over training) and are NOT comparable
across batches or epochs. This module assumes `scaler.unscale_(optimizer)`
has already been called by the training loop before `log_batch_gradients`
or `log_loss_specific_gradients` is invoked. See PHASE_3_INTEGRATION.md for
the exact call site.
"""

import os
import csv
import torch
from datetime import datetime


GRADIENT_METRICS_HEADERS = [
    'epoch', 'iteration', 'timestamp',
    'total_grad_norm', 'encoder_grad_norm', 'decoder_grad_norm', 'cbam_grad_norm',
    'grad_norm_loss_dice', 'grad_norm_loss_ft', 'grad_norm_loss_evid',
    'amp_scale_factor',
]

LAYER_WISE_HEADERS = [
    'epoch', 'iteration', 'timestamp', 'layer_name', 'grad_norm', 'param_count', 'grad_norm_per_param',
]


class GradientLogger:
    def __init__(self, log_dir, model, phase_name="segmentation"):
        self.log_dir = log_dir
        self.phase_name = phase_name
        self.model = model
        os.makedirs(log_dir, exist_ok=True)

        self.grad_csv = os.path.join(log_dir, "gradient_metrics.csv")
        self.layer_csv = os.path.join(log_dir, "gradient_metrics_layerwise.csv")

        if not os.path.exists(self.grad_csv):
            with open(self.grad_csv, 'w', newline='') as f:
                csv.writer(f).writerow(GRADIENT_METRICS_HEADERS)
        if not os.path.exists(self.layer_csv):
            with open(self.layer_csv, 'w', newline='') as f:
                csv.writer(f).writerow(LAYER_WISE_HEADERS)

        self.batch_logs = []
        self.layer_logs = []
        self._pending_loss_specific = {}

    @staticmethod
    def compute_gradient_norm(parameters):
        total = 0.0
        for p in parameters:
            if p.grad is not None:
                total += torch.norm(p.grad.detach()).item() ** 2
        return total ** 0.5

    def log_loss_specific_gradients(self, loss_dice, loss_ft, loss_evid, model):
        """
        Measure how much each loss term contributes to gradients, in
        isolation. Requires calling `.backward(retain_graph=True)` on each
        loss separately, then zeroing grads between measurements so they
        don't accumulate into each other. This DOES NOT replace the real
        combined backward pass the training loop performs afterward — call
        this BEFORE the real `optimizer.zero_grad()` / combined backward,
        using loss tensors that still have `retain_graph=True` available
        (i.e. the combined loss has not been backward'd yet with
        retain_graph=False).

        Returns a dict of the three norms; the caller merges this into the
        next log_batch_gradients() call via _pending_loss_specific.
        """
        params = [p for p in model.parameters() if p.requires_grad]

        def isolated_norm(loss_term):
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
            loss_term.backward(retain_graph=True)
            n = self.compute_gradient_norm(params)
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
            return n

        norm_dice = isolated_norm(loss_dice) if torch.is_tensor(loss_dice) else 0.0
        norm_ft = isolated_norm(loss_ft) if torch.is_tensor(loss_ft) else 0.0
        norm_evid = isolated_norm(loss_evid) if torch.is_tensor(loss_evid) else 0.0

        self._pending_loss_specific = {
            'grad_norm_loss_dice': norm_dice,
            'grad_norm_loss_ft': norm_ft,
            'grad_norm_loss_evid': norm_evid,
        }
        return self._pending_loss_specific

    def log_batch_gradients(self, epoch, batch_idx, model, amp_scale_factor=None):
        """
        Call AFTER scaler.unscale_(optimizer) and AFTER the real combined
        backward pass, but BEFORE optimizer.zero_grad() clears gradients
        for the next batch.
        """
        timestamp = datetime.now().isoformat()
        all_params = [p for p in model.parameters() if p.requires_grad]

        total_norm = self.compute_gradient_norm(all_params)
        encoder_norm = self.compute_gradient_norm(
            [p for p in model.encoder.parameters() if p.requires_grad])
        decoder_norm = self.compute_gradient_norm(
            [p for p in model.decoder.parameters() if p.requires_grad])
        cbam_norm = self.compute_gradient_norm(
            [p for p in model.cbam.parameters() if p.requires_grad])

        row = {
            'epoch': epoch,
            'iteration': batch_idx,
            'timestamp': timestamp,
            'total_grad_norm': f'{total_norm:.8f}',
            'encoder_grad_norm': f'{encoder_norm:.8f}',
            'decoder_grad_norm': f'{decoder_norm:.8f}',
            'cbam_grad_norm': f'{cbam_norm:.8f}',
            'grad_norm_loss_dice': f'{self._pending_loss_specific.get("grad_norm_loss_dice", float("nan")):.8f}' if self._pending_loss_specific else '',
            'grad_norm_loss_ft': f'{self._pending_loss_specific.get("grad_norm_loss_ft", float("nan")):.8f}' if self._pending_loss_specific else '',
            'grad_norm_loss_evid': f'{self._pending_loss_specific.get("grad_norm_loss_evid", float("nan")):.8f}' if self._pending_loss_specific else '',
            'amp_scale_factor': f'{amp_scale_factor:.2f}' if amp_scale_factor is not None else '',
        }
        self.batch_logs.append(row)
        self._pending_loss_specific = {}

    def log_layer_wise_gradients(self, epoch, batch_idx, model):
        timestamp = datetime.now().isoformat()
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad_norm = torch.norm(p.grad.detach()).item()
                count = p.numel()
                self.layer_logs.append({
                    'epoch': epoch, 'iteration': batch_idx, 'timestamp': timestamp,
                    'layer_name': name, 'grad_norm': f'{grad_norm:.8f}',
                    'param_count': count,
                    'grad_norm_per_param': f'{(grad_norm / (count ** 0.5)) if count else 0.0:.8f}',
                })

    def flush_batch(self):
        if self.batch_logs:
            with open(self.grad_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=GRADIENT_METRICS_HEADERS)
                writer.writerows(self.batch_logs)
            self.batch_logs.clear()
        if self.layer_logs:
            with open(self.layer_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=LAYER_WISE_HEADERS)
                writer.writerows(self.layer_logs)
            self.layer_logs.clear()
