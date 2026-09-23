"""
gradient_metrics_similarity.csv writer.

Measures cosine similarity between the gradient vectors produced by each
loss term individually. This is the most expensive diagnostic (3+ isolated
backward passes per measured batch) and is gated by
config.GRADIENT_SIMILARITY_EVERY_N_BATCHES so it only runs occasionally.

Interpretation (not applied automatically, just documented):
  similarity > 0.7   -> losses pushing parameters in a similar direction
  similarity 0 to 0.7 -> largely independent
  similarity < 0      -> losses pushing parameters in opposing directions
                         ("conflicting")
"""

import os
import csv
import torch
import torch.nn.functional as F
from datetime import datetime


SIMILARITY_HEADERS = [
    'epoch', 'iteration', 'timestamp',
    'cosine_sim_dice_vs_ft',
    'cosine_sim_dice_vs_evid',
    'cosine_sim_ft_vs_evid',
    'cosine_sim_hybrid_vs_evid',
    'conflict_score',
]


class GradientSimilarityLogger:
    def __init__(self, log_dir, phase_name="segmentation"):
        self.log_dir = log_dir
        self.phase_name = phase_name
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "gradient_metrics_similarity.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(SIMILARITY_HEADERS)
        self.batch_logs = []

    @staticmethod
    def _flatten_grad(parameters):
        parts = [p.grad.flatten() for p in parameters if p.grad is not None]
        if not parts:
            return torch.tensor([])
        return torch.cat(parts)

    def _gradient_vector_for_loss(self, loss_term, parameters):
        for p in parameters:
            if p.grad is not None:
                p.grad.zero_()
        loss_term.backward(retain_graph=True)
        vec = self._flatten_grad(parameters)
        for p in parameters:
            if p.grad is not None:
                p.grad.zero_()
        return vec

    @staticmethod
    def _cosine(vec1, vec2):
        if vec1.numel() == 0 or vec2.numel() == 0:
            return 0.0
        return float(F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0), dim=1).item())

    def log_gradient_similarity(self, epoch, batch_idx, model, loss_dice, loss_ft, loss_evid):
        timestamp = datetime.now().isoformat()
        params = [p for p in model.parameters() if p.requires_grad]

        grad_dice = self._gradient_vector_for_loss(loss_dice, params)
        grad_ft = self._gradient_vector_for_loss(loss_ft, params)
        grad_evid = self._gradient_vector_for_loss(loss_evid, params) if torch.is_tensor(loss_evid) else torch.tensor([])

        sim_dice_ft = self._cosine(grad_dice, grad_ft)
        sim_dice_evid = self._cosine(grad_dice, grad_evid)
        sim_ft_evid = self._cosine(grad_ft, grad_evid)

        grad_hybrid = 0.5 * grad_dice + 0.5 * grad_ft if grad_dice.numel() and grad_ft.numel() else torch.tensor([])
        sim_hybrid_evid = self._cosine(grad_hybrid, grad_evid)

        conflict = (max(0.0, -sim_dice_ft) + max(0.0, -sim_dice_evid) + max(0.0, -sim_ft_evid)) / 3.0

        row = {
            'epoch': epoch, 'iteration': batch_idx, 'timestamp': timestamp,
            'cosine_sim_dice_vs_ft': f'{sim_dice_ft:.6f}',
            'cosine_sim_dice_vs_evid': f'{sim_dice_evid:.6f}',
            'cosine_sim_ft_vs_evid': f'{sim_ft_evid:.6f}',
            'cosine_sim_hybrid_vs_evid': f'{sim_hybrid_evid:.6f}',
            'conflict_score': f'{conflict:.6f}',
        }
        self.batch_logs.append(row)
        return {
            'sim_dice_vs_ft': sim_dice_ft, 'sim_dice_vs_evid': sim_dice_evid,
            'sim_ft_vs_evid': sim_ft_evid, 'sim_hybrid_vs_evid': sim_hybrid_evid,
            'conflict_score': conflict,
        }

    def flush(self):
        if not self.batch_logs:
            return
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=SIMILARITY_HEADERS)
            writer.writerows(self.batch_logs)
        self.batch_logs.clear()
