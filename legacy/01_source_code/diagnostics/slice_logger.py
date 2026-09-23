"""
slice_metrics.csv writer.

Logs, per selected slice, per sample, per batch:
  - which of the 64 resampled slice indices the AdaptiveSliceSelector chose
  - that slice's raw score from the scorer network
  - that slice's learned fusion attention weight from Conv2D5Stem (the one
    quantity in the slice-selection path that IS actually trained by
    gradient descent - see PHASE_1_CODEBASE_AUDIT.md Section 6)
  - whether the always-fixed supervision index (center of the volume) was
    among the slices actually selected for this sample

This directly operationalizes the Phase 1 finding that slice identity is
computed internally but discarded, and gives a measurable answer to
"how often does the selector even include the slice we're grading it against."

Read-only: this module reads `last_top_indices` / `last_slice_scores` /
`last_fusion_alphas` attributes that final_model.py's AdaptiveSliceSelector
and Conv2D5Stem stash on themselves (an additive, side-effect-free change -
see PHASE_3_INTEGRATION.md). It does not influence the forward pass.
"""

import os
import csv
import torch
from datetime import datetime


SLICE_METRICS_HEADERS = [
    'epoch', 'iteration', 'timestamp', 'patient_id', 'case', 'timepoint',
    'sample_in_batch', 'selected_slice_index', 'slice_score',
    'fusion_attention_weight', 'supervised_center_index',
    'center_index_was_selected',
]


class SliceLogger:
    def __init__(self, log_dir, phase_name="segmentation"):
        self.log_dir = log_dir
        self.phase_name = phase_name
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "slice_metrics.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(SLICE_METRICS_HEADERS)
        self.batch_logs = []

    def log_batch(self, epoch, batch_idx, top_indices, slice_scores,
                  fusion_alphas=None, center_index=None,
                  patient_ids=None, cases=None, timepoints=None):
        """
        Args:
            top_indices: (B, k) LongTensor - selected slice indices, from
                         AdaptiveSliceSelector.last_top_indices
            slice_scores: (B, D) FloatTensor - raw score for every candidate
                          slice, from AdaptiveSliceSelector.last_slice_scores
            fusion_alphas: (B, k, 1, 1) FloatTensor or None - Conv2D5Stem's
                           learned per-slice fusion weight, from
                           Conv2D5Stem.last_fusion_alphas
            center_index: int - the fixed supervised slice index (labels.shape[2]//2)
            patient_ids / cases / timepoints: optional per-sample identity lists
                          (length B), threaded through from the batch dict
        """
        timestamp = datetime.now().isoformat()
        top_indices_cpu = top_indices.detach().cpu()
        B, k = top_indices_cpu.shape

        fusion_cpu = None
        if fusion_alphas is not None:
            fusion_cpu = fusion_alphas.detach().cpu().reshape(B, k)

        scores_cpu = slice_scores.detach().cpu() if slice_scores is not None else None

        for b in range(B):
            selected = top_indices_cpu[b].tolist()
            center_selected = (center_index in selected) if center_index is not None else ''
            for slot, slice_idx in enumerate(selected):
                score_val = float(scores_cpu[b, slice_idx].item()) if scores_cpu is not None else ''
                fusion_val = float(fusion_cpu[b, slot].item()) if fusion_cpu is not None else ''
                self.batch_logs.append({
                    'epoch': epoch,
                    'iteration': batch_idx,
                    'timestamp': timestamp,
                    'patient_id': patient_ids[b] if patient_ids else '',
                    'case': cases[b] if cases else '',
                    'timepoint': timepoints[b] if timepoints else '',
                    'sample_in_batch': b,
                    'selected_slice_index': slice_idx,
                    'slice_score': f'{score_val:.6f}' if score_val != '' else '',
                    'fusion_attention_weight': f'{fusion_val:.6f}' if fusion_val != '' else '',
                    'supervised_center_index': center_index if center_index is not None else '',
                    'center_index_was_selected': center_selected,
                })

    def flush(self):
        if not self.batch_logs:
            return
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=SLICE_METRICS_HEADERS)
            writer.writerows(self.batch_logs)
        self.batch_logs.clear()
