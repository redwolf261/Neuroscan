"""
Experiment E12: EGGO-M (Margin-only) -- first training run with the new
algorithm active. Implements exactly the design specified in:
  PHASE_E8_EGGO_V1_SIMPLIFIED.md   -- L = L_seg + lambda * U_hat * B_i * L_margin
  PHASE_E9_TRAINABLE_BOUNDARY_HEAD.md (corrected) -- dec1.detach() into boundary_head
  PHASE_E10_MARGIN_LOSS_SELECTION.md -- De Brabandere-style pairwise hinge margin
  PHASE_E11_IMPLEMENTATION_SPEC.md -- subsampling, negative cap, gradient flow
  PHASE_E11_5_READINESS_REVIEW.md  -- gradient audit, detach requirements,
    pre-registered predictions/failure criteria, boundary-head drift logging

Uses neuroscan_3d_v2.UNet3D_v2 (baseline_frozen_v2) as the base
architecture -- NOT neuroscan_3d_fixed.py directly, per the v1/v2
lineage established in PHASE_E11_5 Sec 6. v1 is never imported or
modified here except transitively (v2 subclasses it).

Total loss:
    L = L_seg + mu * L_boundary + lambda * L_margin

Modes:
  --mu 0 --lambda_margin 0  : EGGO-M fully inactive. Must reproduce
                               baseline_frozen_v1's exact training
                               dynamics (verified separately, not by
                               this script -- see verify_v2_matches_v1.py).
  --mu >0 --lambda_margin 0 : boundary head trains (via its own BCE) but
                               contributes nothing to L_margin (since
                               lambda=0 zeroes it) -- a useful
                               intermediate diagnostic mode.
  --mu >0 --lambda_margin >0: full EGGO-M active.
"""
import os
import sys
import csv
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v2 import UNet3D_v2  # noqa: E402
from neuroscan_3d_fixed import HybridLoss, FocalTverskyLoss, EvidentialBetaLoss  # noqa: E402
from Dataset.brats_dataset import create_brats_loaders  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp00b_baseline_convergence"))
from metrics import MetricAccumulator  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_c0_weight_ablation"))
from calibration import ECEAccumulator  # noqa: E402

# --- Constants derived from measurement, not picked (PHASE_E5 Sec 2.2) ---
# ORIGINAL tau_b (E12b's value, now known miscalibrated -- see
# PHASE_E12D_MECHANISM_FAILURE_INVESTIGATION.md): geometric mean of
# E1.3's measured correct/incorrect crossover distances on a STATIC,
# fully-converged offline classifier (1.456, 0.648). PHASE_E12D found
# this mismatches the LIVE boundary head's logit-magnitude growth during
# actual training (implied tau_b grows ~13x from epoch 1 to epoch 30,
# per PHASE_E12E's calibration measurement) -- causing B_i to collapse
# toward zero by epoch 15-20. Kept here only as a fallback/reference
# value, NOT used by default anymore -- see EMATauB below.
ORIGINAL_STATIC_TAU_B = float(np.sqrt(1.456 * 0.648))  # ~0.9713

# NEW delta_d (PHASE_E12E, CORRECTED after a real methodological bug):
# first calibration attempt measured a FRESH model in .eval() mode and
# got delta_d=0.0148 -- but live verification (E12e's first attempt)
# showed active_hinge_pct=0.0% throughout, not the targeted ~20%.
# Root cause: BatchNorm3d behaves completely differently in eval() vs
# train() mode, and the real training loop always runs in train() mode.
# Direct A/B check on the SAME seed/batch: dec1 std in eval mode=0.0089,
# in train mode=0.6095 -- a ~68x difference AT INITIALIZATION (this gap
# shrinks to <10% by epoch 1 of actual training and <3% by epoch 10, so
# it specifically affected this fresh-init measurement, not the
# checkpoint-based E12d/E12b.5 analyses which all used epoch>=1
# checkpoints). Recalibrated properly in train() mode: this value
# targets ~20% active hinge pairs at TRUE initialization
# (e12e_calibrate_constants.py, PART 1, corrected).
DELTA_D_CALIBRATED = 3.6659

ANCHORS_PER_VOLUME = 2000     # PHASE_E5 Sec 5.2 / PHASE_E11 Sec 2 step 4
MAX_NEGATIVES_PER_ANCHOR = 50  # PHASE_E11 Sec 2 step 6 (the added mitigation)


class EMATauB:
    """
    Live, adaptive tau_b, per PHASE_E12D's recommendation (option b):
    "make tau_b adaptive: an EMA of median|d_i| updated during training...
    so B_i's dynamic range stays meaningful throughout." PHASE_E12E
    confirmed a single FIXED tau_b cannot work (implied tau_b grows
    ~13x, 0.82->10.51, from epoch 1 to epoch 30 of a trained run) --
    this tracks an EMA of the boundary head's own |logit| median and
    derives tau_b from it each batch, so B_i = exp(-|d_i|/tau_b) stays
    centered near 0.5 for a "typical" (median) voxel throughout training,
    rather than decaying to near-zero as the boundary head's confidence
    (and thus |d_i|) grows.

    tau_b_for(median) solves exp(-median/tau_b) = 0.5 for tau_b, i.e.
    tau_b = median / ln(2) ~= median / 0.693 -- same relationship used
    in e12e_calibrate_constants.py's PART 2 measurement.
    """
    LN2 = 0.6931471805599453

    def __init__(self, decay=0.98, init_value=ORIGINAL_STATIC_TAU_B, warmup_steps=20):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.ema_median_abs_d = None
        self.step_count = 0
        self._init_value = init_value

    def update(self, abs_boundary_logit_batch):
        """abs_boundary_logit_batch: 1D tensor of |d_i| for this batch's
        sampled anchors (detached, no grad needed -- this is bookkeeping,
        not part of the loss computation graph)."""
        self.step_count += 1
        current_median = abs_boundary_logit_batch.median().item()
        if self.ema_median_abs_d is None:
            self.ema_median_abs_d = current_median
        else:
            self.ema_median_abs_d = self.decay * self.ema_median_abs_d + (1 - self.decay) * current_median

    @property
    def tau_b(self):
        # Warmup: not enough samples yet for the EMA to be meaningful --
        # same principle as ABO's delta_warmup_steps (a near-zero-sample
        # EMA is a cold-start artifact, not a real estimate).
        if self.ema_median_abs_d is None or self.step_count <= self.warmup_steps:
            return self._init_value
        return max(self.ema_median_abs_d / self.LN2, 1e-6)
EVIDENCE_P99_DEFAULT = 23.25   # measured on the 30-volume E1 extraction; re-measure on full train set before trusting beyond the smoke test


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_stratified_anchors(evidence_flat, n_sample, rng):
    """
    Stratify toward high uncertainty (low evidence), per PHASE_E11 Sec 2
    step 4 -- only ~0.14% of voxels are actually incorrect (E1 finding),
    so uniform random sampling would rarely include the voxels that
    matter. Takes the n_sample voxels with the LOWEST evidence (highest
    uncertainty) rather than a uniform random draw.

    NOTE: this is a simple, deterministic top-k strategy for the smoke
    test. A softer stratified-random approach (e.g. weighted sampling by
    1/evidence) is a reasonable future refinement but not required for
    E12a's pass/fail criteria -- flagged here, not implemented, to keep
    the smoke test's behavior easy to reason about first.
    """
    total = evidence_flat.shape[0]
    n_sample = min(n_sample, total)
    # top-k lowest evidence = highest uncertainty
    idx = torch.topk(-evidence_flat, k=n_sample).indices
    return idx


def compute_margin_loss(dec1_flat, evidence_flat, boundary_logit_flat, gt_flat,
                         anchor_idx, tau_b, evidence_p99, delta_d, max_negatives, rng, device,
                         w_hat=None, margin_mode=None, probs_flat=None):
    """
    L_margin per PHASE_E10's selected formula (De Brabandere-style
    pairwise hinge, extended to in-batch opposite-class negatives):

        L_margin = (1/|B|) sum_i  U_hat_i * B_i * (1/|N(i)|) sum_{j in N(i)}
                       [2*delta_d - ||z_i - z_j||]_+^2

    Both U_hat_i and B_i are computed with .detach()'d inputs (evidence
    and boundary_logit respectively) per PHASE_E11_5's gradient audit --
    they must never carry gradient back into the evidential head or the
    boundary head via this loss. Only dec1_flat itself (NOT detached)
    carries gradient, which is the intended mechanism.

    VECTORIZED implementation (rewritten after E12a's first smoke-test
    attempt showed a per-anchor Python loop with .tolist() device syncs
    is catastrophically slow -- ~45s/batch vs baseline's ~1s/batch. This
    version does one batched matrix-distance computation per class,
    with no Python-level loop over individual anchors).

    Returns (mean_loss, per_anchor_weight, diagnostics_dict) -- the
    diagnostics dict (added per PHASE_E12E) carries active_hinge_pct
    (live, per-batch, not just post-hoc) and the raw abs_boundary_logit
    tensor for the caller to feed into EMATauB's live tau_b tracking.

    margin_mode: explicit dispatcher between mathematically distinct
    distance functions, per PHASE_E25's design correction (replaces an
    earlier, rejected `use_signed` boolean specifically because a loose
    flag alongside w_hat permits ambiguous/undefined combinations --
    e.g. "w_hat=None, use_signed=True" has no defined meaning). Exactly
    one of:
      "euclidean"   -- w_hat MUST be None. dist = ||z_i - z_j||_2
                        (the ORIGINAL, PHASE_E10 formula, BYTE-IDENTICAL
                        to pre-E24 behavior -- verified by PHASE_E24's
                        test_baseline_preservation). delta_d=3.6659.
      "task_aligned" -- w_hat MUST be provided. dist = |w_hat . (z_i-z_j)|
                        (PHASE_E23/E24's UNSIGNED projected margin,
                        L_margin^w). delta_d=delta_d_w=0.2553.
      "sc_tam"      -- w_hat MUST be provided. SIGNED, class-conditional
                        distance, per PHASE_E25's design:
                            d^SC = (z_tumor . w_hat) - (z_bg . w_hat)
                        computed from EACH PAIR'S OWN KNOWN CLASS IDENTITY
                        (tumor_local/bg_local), NOT from loop-relative
                        zi/zj order -- CRITICAL, per PHASE_E25's own
                        design review: the existing bidirectional loop
                        (tumor-anchor/bg-negative, then bg-anchor/
                        tumor-negative) is symmetric-safe for the
                        Euclidean/unsigned-projected distances (both
                        satisfy dist(zi,zj)=dist(zj,zi)) but NOT for a
                        signed distance -- naively computing (zi-zj).w_hat
                        inside the existing loop would flip sign between
                        the two iterations and silently cancel the exact
                        directional signal this mode exists to provide.
                        This mode explicitly reconstructs which side of
                        each pair is tumor vs background, INDEPENDENT of
                        which loop iteration produced it, so the sign is
                        always "tumor side minus background side."
                        delta_d=m_ij, SC-TAM's own calibrated margin
                        (NOT delta_d_w -- a separate calibration, per
                        PHASE_E25 Section 11).
      "sc_tam_gated" -- w_hat MUST be provided, probs_flat MUST be
                        provided. Identical signed distance/hinge math to
                        "sc_tam" (this mode does NOT change dist or
                        margin_target at all), but multiplies each
                        anchor's per_anchor_loss by an ADDITIONAL,
                        ground-truth-conditioned gate:
                            gate_i = |p_i - target_i|
                        where target_i = 1.0 for a tumor-GT anchor, 0.0
                        for a background-GT anchor, and p_i is that
                        anchor's OWN predicted probability (from
                        probs_flat, DETACHED -- same discipline as
                        U_hat/B, this gate must never carry gradient back
                        into seg_head). This is C6-3's own gate, per
                        PHASE_E25_C63_GATE_RETROSPECTIVE_TEST.md's
                        retrospective validation: passes the strict
                        two-part go/no-go criterion (92.3% corrective
                        movement retained, 99.2% of TP/TN damaging
                        movement removed, measured on C6-2's own real
                        training-realized trajectory before this mode was
                        ever implemented). NOTE: ground truth is used HERE,
                        inside the training-time loss construction, where
                        it is already legitimately available (exactly as
                        gt_flat already is, for tumor_mask/bg_mask) --
                        this gate is NEVER read at inference time; a
                        trained model's forward pass (probs = seg_head(dec1))
                        is completely unaffected by this mode and produces
                        predictions with no reference to ground truth,
                        exactly as normal.

    w_hat: OPTIONAL, shape (32,), a DETACHED unit vector. Required
    (non-None) for "task_aligned", "sc_tam", and "sc_tam_gated"; must be
    None for "euclidean" (enforced by an assertion, not silently ignored).

    probs_flat: OPTIONAL, shape matching gt_flat/evidence_flat (full,
    UN-indexed voxel-flat tensor, same convention as evidence_flat/
    boundary_logit_flat -- indexed by anchor_idx internally, matching
    every other per-voxel input this function already takes this way).
    Required (non-None) ONLY for "sc_tam_gated"; ignored (must be None)
    for every other mode, enforced by assertion rather than silently
    accepted-and-unused.

    BACKWARD COMPATIBILITY: margin_mode=None (the default) auto-infers
    "euclidean" when w_hat is None, or "task_aligned" when w_hat is
    provided -- this is EXACTLY the pre-PHASE_E25 behavior (every caller
    from E14 through E24's own scripts distinguishes euclidean/task_aligned
    purely via w_hat=None vs. w_hat=<tensor>, never passing margin_mode at
    all) preserved byte-for-byte, so none of those existing call sites
    need to change. margin_mode="sc_tam"/"sc_tam_gated" must be passed
    EXPLICITLY -- there is no way to auto-infer either from w_hat alone,
    since w_hat is also non-None for "task_aligned".
    """
    if margin_mode is None:
        margin_mode = "euclidean" if w_hat is None else "task_aligned"
    assert margin_mode in ("euclidean", "task_aligned", "sc_tam", "sc_tam_gated"), f"unknown margin_mode: {margin_mode}"
    if margin_mode == "euclidean":
        assert w_hat is None, "margin_mode='euclidean' requires w_hat=None (unambiguous baseline path)"
    else:
        assert w_hat is not None, f"margin_mode='{margin_mode}' requires a real w_hat"
    if margin_mode == "sc_tam_gated":
        assert probs_flat is not None, "margin_mode='sc_tam_gated' requires probs_flat (the model's own predicted probability, for the GT-conditioned gate)"
    else:
        assert probs_flat is None, f"margin_mode='{margin_mode}' does not use probs_flat -- pass None to avoid silently ignoring it"
    anchors_z = dec1_flat[anchor_idx]                    # (n_anchor, 32), gradient-carrying
    anchors_evidence = evidence_flat[anchor_idx].detach()  # detached, per spec
    anchors_boundary = boundary_logit_flat[anchor_idx].detach()  # detached, per spec
    anchors_gt = gt_flat[anchor_idx]

    U_hat = 1.0 - torch.clamp(anchors_evidence / evidence_p99, 0.0, 1.0)
    B = torch.exp(-torch.abs(anchors_boundary) / tau_b)
    weight = U_hat * B  # (n_anchor,)

    if margin_mode == "sc_tam_gated":
        anchors_probs = probs_flat[anchor_idx].detach()  # DETACHED, same discipline as U_hat/B -- gate must never backprop into seg_head
        target = anchors_gt.detach()  # 1.0 for tumor-GT anchors, 0.0 for background-GT anchors -- gt_flat is already {0,1}-valued, used directly as target
        gate = torch.abs(anchors_probs - target)  # (n_anchor,) -- PHASE_E25_C63's retrospectively-validated gate
        weight = weight * gate

    n_anchor = anchors_z.shape[0]
    losses = torch.zeros(n_anchor, device=device)

    tumor_mask = anchors_gt > 0.5
    bg_mask = ~tumor_mask
    tumor_local = torch.where(tumor_mask)[0]
    bg_local = torch.where(bg_mask)[0]

    active_pair_count = 0
    total_pair_count = 0

    for pos_local, opp_local in ((tumor_local, bg_local), (bg_local, tumor_local)):
        if pos_local.numel() == 0 or opp_local.numel() == 0:
            continue  # no anchors of this class, or no opposite-class pool -- skip (Failure Mode 2 territory)

        n_pos = pos_local.numel()
        n_neg = min(max_negatives, opp_local.numel())

        # Vectorized negative sampling: for each of the n_pos anchors,
        # draw n_neg indices (with replacement) from opp_local -- a
        # single (n_pos, n_neg) integer tensor, no Python loop.
        rand_idx = torch.randint(0, opp_local.numel(), (n_pos, n_neg), device=device)
        neg_local = opp_local[rand_idx]  # (n_pos, n_neg), indices into anchors_z

        zi = anchors_z[pos_local].unsqueeze(1)         # (n_pos, 1, 32)
        zj = anchors_z[neg_local]                       # (n_pos, n_neg, 32)

        if margin_mode == "euclidean":
            dist = torch.norm(zi - zj, dim=2)            # (n_pos, n_neg), single batched op -- UNCHANGED baseline path
            margin_target = 2 * delta_d
        elif margin_mode == "task_aligned":
            dist = torch.abs((zi - zj) @ w_hat)          # (n_pos, n_neg) -- UNSIGNED projected path, PHASE_E23/E24
            margin_target = 2 * delta_d
        else:  # margin_mode in ("sc_tam", "sc_tam_gated") -- IDENTICAL dist/margin_target math; sc_tam_gated differs ONLY in the extra gate applied to weight, above
            # SIGNED, class-conditional distance -- per PHASE_E25's design
            # review, computed from EACH PAIR'S KNOWN CLASS IDENTITY, not
            # from loop-relative zi/zj order (which would flip sign between
            # the two loop iterations and cancel the directional signal).
            # pos_local is IS-checked against tumor_local/bg_local (the
            # exact tensor objects the outer loop's tuple was built from,
            # per Python `is` identity -- not a value heuristic) to always
            # express the result as (tumor_side_projection - bg_side_projection),
            # regardless of which local variable currently holds which class.
            zi_proj = (anchors_z[pos_local] @ w_hat).unsqueeze(1)  # (n_pos, 1) -- one projection per anchor, broadcasts over n_neg
            zj_proj = anchors_z[neg_local] @ w_hat                  # (n_pos, n_neg)
            if pos_local is tumor_local:
                dist = zi_proj - zj_proj    # pos=tumor, neg=bg: tumor - bg, correct sign
            else:
                dist = zj_proj - zi_proj    # pos=bg, neg=tumor: tumor - bg, same sign convention as above
            # dist is SIGNED here, deliberately NOT absolute-valued -- this is SC-TAM's entire point.
            margin_target = delta_d  # SC-TAM's own calibrated m_ij (Section 11/12), NOT "2*delta_d" -- that convention belongs to the symmetric unsigned formulas only

        hinge = torch.clamp(margin_target - dist, min=0.0) ** 2  # (n_pos, n_neg) -- SQUARED hinge for all three modes, per PHASE_E25's lock
        per_anchor_loss = hinge.mean(dim=1)               # (n_pos,)

        losses[pos_local] = weight[pos_local] * per_anchor_loss

        # E12e diagnostic: track active-hinge % directly during training
        # (not just post-hoc), per the user's explicit E12e criterion
        # "verify active hinge % stays nonzero" -- this must be watched
        # live, not just inferred after the fact from a completed run.
        with torch.no_grad():
            active_pair_count += (dist < margin_target).sum().item()
            total_pair_count += dist.numel()

    active_hinge_pct = active_pair_count / max(1, total_pair_count)
    diagnostics = {"active_hinge_pct": active_hinge_pct, "abs_boundary_logit": torch.abs(anchors_boundary)}

    return losses.mean(), weight, diagnostics


class EGGOMExperiment:
    def __init__(self, config_path, exp_dir, seed, mu, lambda_margin, delta_d, run_name=None, num_workers=None):
        self.seed = seed
        self.mu = mu
        self.lambda_margin = lambda_margin
        self.delta_d = delta_d
        self.num_workers_override = num_workers
        set_seed(seed)

        dir_name = run_name if run_name else f"seed{seed}_mu{mu}_lambda{lambda_margin}"
        self.exp_dir = Path(exp_dir) / dir_name
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.log_dir = self.exp_dir / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        dataset_root = self.config["dataset"]["root_dir"]
        if not Path(dataset_root).is_absolute():
            dataset_root = project_root / dataset_root
        self.config["dataset"]["root_dir"] = str(dataset_root)

        self.device = torch.device(self.config.get("hardware", {}).get("device", "cpu"))
        self.best_val_dice = 0.0
        self.epoch = 0

        self.model = UNet3D_v2(
            in_channels=self.config["model"]["in_channels"],
            out_channels=self.config["model"]["out_channels"],
        ).to(self.device)

        self.focal_fn = FocalTverskyLoss()
        self.evidential_fn = EvidentialBetaLoss(weight=0.5)
        self.focal_weight = 0.5
        self.evidential_weight = 0.5
        self.boundary_criterion = nn.BCEWithLogitsLoss()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.config["training"]["epochs"], eta_min=1e-6
        )

        self.train_loader, self.val_loader = create_brats_loaders(
            batch_size=self.config["training"]["batch_size"],
            num_workers=(self.num_workers_override if self.num_workers_override is not None
                         else self.config["training"].get("num_workers", 0)),
            root_dir=self.config["dataset"]["root_dir"],
            val_split=self.config["dataset"]["val_split"],
        )

        self.rng = np.random.RandomState(seed)
        self.tau_b_tracker = EMATauB()  # PHASE_E12E: live, adaptive tau_b, replaces the fixed ORIGINAL_STATIC_TAU_B

        print(f"[EGGO-M seed{seed} mu={mu} lambda={lambda_margin}] "
              f"Training subjects: {len(self.train_loader.dataset)}")
        print(f"[EGGO-M seed{seed}] Validation subjects: {len(self.val_loader.dataset)}")
        print(f"[EGGO-M seed{seed}] Device: {self.device}, initial tau_b={self.tau_b_tracker.tau_b:.4f} "
              f"(live/adaptive, PHASE_E12E), delta_d={delta_d}")

        self._open_logs()

    def _open_logs(self):
        self.f_metrics = open(self.exp_dir / "epoch_metrics.csv", "w", newline="")
        self.w_metrics = csv.writer(self.f_metrics)
        self.w_metrics.writerow([
            "epoch", "train_loss", "train_dice", "train_seg_loss", "train_boundary_loss",
            "train_margin_loss", "active_hinge_pct", "tau_b_end_of_epoch", "val_loss", "val_dice",
            "val_iou", "val_precision", "val_recall", "val_f1", "val_hd95", "val_ece",
            "boundary_bce", "boundary_auc_proxy", "epoch_time_sec", "peak_gpu_memory_mb",
        ])

    def close_logs(self):
        self.f_metrics.close()

    def train_epoch(self):
        self.model.train()
        acc = MetricAccumulator()
        total_seg_loss = 0.0
        total_boundary_loss = 0.0
        total_margin_loss = 0.0
        total_active_hinge_pct = 0.0  # PHASE_E12E: live monitoring, per the user's explicit criterion
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"[EGGO-M] Epoch {self.epoch+1} [Train]")
        for batch_idx, (images, masks, _) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)

            probs = outputs["probs"]
            alpha, beta = outputs["alpha"], outputs["beta"]
            boundary_logit = outputs["boundary_logit"]
            dec1 = outputs["dec1"]

            focal_loss = self.focal_fn(probs, masks)
            evidential_loss = self.evidential_fn(alpha, beta, masks)
            seg_loss = self.focal_weight * focal_loss + self.evidential_weight * evidential_loss

            if not torch.isfinite(seg_loss):
                raise RuntimeError(f"[EGGO-M] NaN/Inf seg_loss at epoch {self.epoch} batch {batch_idx}")

            boundary_loss = self.boundary_criterion(boundary_logit, masks)
            if not torch.isfinite(boundary_loss):
                raise RuntimeError(f"[EGGO-M] NaN/Inf boundary_loss at epoch {self.epoch} batch {batch_idx}")

            margin_loss = torch.tensor(0.0, device=self.device)
            active_hinge_pct = 0.0
            current_tau_b = self.tau_b_tracker.tau_b
            if self.lambda_margin > 0:
                # Compute evidence (detached scalar) for stratified sampling + U_hat
                with torch.no_grad():
                    evidence_full = (alpha + beta - 2.0)  # (B,1,D,H,W)

                B, C, D, H, W = dec1.shape
                dec1_perm = dec1.permute(0, 2, 3, 4, 1).reshape(-1, C)  # (B*D*H*W, 32)
                evidence_flat = evidence_full.reshape(-1)
                boundary_flat = boundary_logit.reshape(-1)
                gt_flat = masks.reshape(-1)

                # Per-volume stratified anchor sampling, then combine
                voxels_per_vol = D * H * W
                anchor_idx_list = []
                for b in range(B):
                    vol_evidence = evidence_flat[b * voxels_per_vol:(b + 1) * voxels_per_vol]
                    local_idx = sample_stratified_anchors(vol_evidence, ANCHORS_PER_VOLUME, self.rng)
                    anchor_idx_list.append(local_idx + b * voxels_per_vol)
                anchor_idx = torch.cat(anchor_idx_list)

                margin_loss, mw, margin_diag = compute_margin_loss(
                    dec1_perm, evidence_flat, boundary_flat, gt_flat,
                    anchor_idx, current_tau_b, EVIDENCE_P99_DEFAULT, self.delta_d,
                    MAX_NEGATIVES_PER_ANCHOR, self.rng, self.device,
                )
                if not torch.isfinite(margin_loss):
                    raise RuntimeError(f"[EGGO-M] NaN/Inf margin_loss at epoch {self.epoch} batch {batch_idx}")

                active_hinge_pct = margin_diag["active_hinge_pct"]
                # PHASE_E12E: update the live tau_b EMA from this batch's
                # |boundary_logit| values -- keeps B_i's dynamic range
                # meaningful as the boundary head's own confidence grows,
                # instead of decaying toward zero against a fixed tau_b.
                self.tau_b_tracker.update(margin_diag["abs_boundary_logit"])

            total_loss = seg_loss + self.mu * boundary_loss + self.lambda_margin * margin_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            acc.update(seg_loss.item(), probs, masks, compute_hd95=False)
            total_seg_loss += seg_loss.item()
            total_boundary_loss += boundary_loss.item()
            total_margin_loss += margin_loss.item() if isinstance(margin_loss, torch.Tensor) else margin_loss
            total_active_hinge_pct += active_hinge_pct
            n_batches += 1

            pbar.set_postfix({
                "seg": seg_loss.item(), "bnd": boundary_loss.item(),
                "margin": margin_loss.item() if isinstance(margin_loss, torch.Tensor) else margin_loss,
                "active%": f"{active_hinge_pct*100:.1f}", "tau_b": f"{current_tau_b:.3f}",
            })

        summary = acc.summary()
        return {
            "loss": summary["loss"], "dice": summary["dice"],
            "seg_loss": total_seg_loss / max(1, n_batches),
            "boundary_loss": total_boundary_loss / max(1, n_batches),
            "margin_loss": total_margin_loss / max(1, n_batches),
            "active_hinge_pct": total_active_hinge_pct / max(1, n_batches),
            "tau_b_end_of_epoch": self.tau_b_tracker.tau_b,
        }

    def validate(self):
        self.model.eval()
        acc = MetricAccumulator()
        ece_acc = ECEAccumulator(n_bins=15)
        boundary_bce_total = 0.0
        boundary_correct = 0
        boundary_total = 0
        n_batches = 0

        pbar = tqdm(self.val_loader, desc=f"[EGGO-M] Epoch {self.epoch+1} [Val]")
        with torch.no_grad():
            for images, masks, _ in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                outputs = self.model(images)

                probs, alpha, beta = outputs["probs"], outputs["alpha"], outputs["beta"]
                boundary_logit = outputs["boundary_logit"]

                focal_loss = self.focal_fn(probs, masks)
                evidential_loss = self.evidential_fn(alpha, beta, masks)
                total = self.focal_weight * focal_loss + self.evidential_weight * evidential_loss

                acc.update(total.item(), probs, masks, compute_hd95=True)
                ece_acc.update(alpha, beta, masks)

                b_bce = self.boundary_criterion(boundary_logit, masks)
                boundary_bce_total += b_bce.item()
                boundary_pred = (torch.sigmoid(boundary_logit) >= 0.5).float()
                boundary_correct += (boundary_pred == masks).sum().item()
                boundary_total += masks.numel()
                n_batches += 1

                pbar.set_postfix({"loss": total.item()})

        summary = acc.summary()
        ece, _ = ece_acc.compute()
        summary["ece"] = ece
        summary["boundary_bce"] = boundary_bce_total / max(1, n_batches)
        # accuracy as a cheap proxy for AUC in the per-epoch log; full AUC
        # computed separately in analyze_boundary_head.py against E1.3's benchmark
        summary["boundary_accuracy_proxy"] = boundary_correct / max(1, boundary_total)
        return summary

    def save_checkpoint(self, is_best=False, is_periodic=True):
        checkpoint = {
            "epoch": self.epoch, "seed": self.seed, "mu": self.mu,
            "lambda_margin": self.lambda_margin,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_dice": self.best_val_dice, "config": self.config,
        }
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")
        # Only save the full per-epoch checkpoint on periodic epochs
        # (E12b.5 diagnostic points) to avoid filling disk with a
        # full model state_dict every single epoch over a 20-30 epoch run.
        if is_periodic:
            torch.save(checkpoint, self.checkpoint_dir / f"epoch_{self.epoch+1}.pth")

    def train(self, epochs, checkpoint_every=5):
        print("\n" + "=" * 70)
        print(f"EXPERIMENT E12: EGGO-M [seed={self.seed}, mu={self.mu}, lambda={self.lambda_margin}]")
        print("=" * 70 + "\n")

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for epoch in range(epochs):
            self.epoch = epoch
            t0 = time.time()

            train_metrics = self.train_epoch()
            val = self.validate()
            self.scheduler.step()
            epoch_time = time.time() - t0

            peak_mem_mb = (torch.cuda.max_memory_allocated() / 1e6) if self.device.type == "cuda" else 0.0

            print(
                f"[EGGO-M] Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s, peak_mem={peak_mem_mb:.0f}MB) | "
                f"Train: loss={train_metrics['loss']:.4f} dice={train_metrics['dice']:.4f} "
                f"seg={train_metrics['seg_loss']:.4f} bnd={train_metrics['boundary_loss']:.4f} "
                f"margin={train_metrics['margin_loss']:.6f} "
                f"active%={train_metrics['active_hinge_pct']*100:.2f} tau_b={train_metrics['tau_b_end_of_epoch']:.3f} | "
                f"Val: dice={val['dice']:.4f} precision={val['precision']:.4f} recall={val['recall']:.4f} "
                f"hd95={val['hd95']:.2f} ece={val['ece']:.4f} "
                f"boundary_bce={val['boundary_bce']:.4f} boundary_acc={val['boundary_accuracy_proxy']:.4f}"
            )

            self.w_metrics.writerow([
                epoch, train_metrics["loss"], train_metrics["dice"], train_metrics["seg_loss"],
                train_metrics["boundary_loss"], train_metrics["margin_loss"],
                train_metrics["active_hinge_pct"], train_metrics["tau_b_end_of_epoch"],
                val["loss"], val["dice"], val["iou"], val["precision"], val["recall"], val["f1"],
                val["hd95"], val["ece"], val["boundary_bce"], val["boundary_accuracy_proxy"],
                epoch_time, peak_mem_mb,
            ])
            self.f_metrics.flush()

            is_best = val["dice"] > self.best_val_dice
            if is_best:
                self.best_val_dice = val["dice"]

            # E12b.5 geometry diagnostic checkpoints: every `checkpoint_every`
            # epochs, plus always epoch 1, always the best-so-far, and
            # always the final epoch -- matches E7's checkpoint pattern so
            # analyze_checkpoints.py can be reused directly on these runs.
            epoch_1indexed = epoch + 1
            is_periodic = (epoch_1indexed % checkpoint_every == 0) or epoch_1indexed == 1
            is_final = epoch_1indexed == epochs
            self.save_checkpoint(is_best=is_best, is_periodic=(is_periodic or is_final))

        self.close_logs()
        print(f"\n[EGGO-M seed{self.seed}] Done. Best Val Dice: {self.best_val_dice:.4f}")
        return self.best_val_dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment E12: EGGO-M (margin-only)")
    parser.add_argument("--config", default="../../configs/brats.yaml")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mu", type=float, default=0.1, help="Boundary head loss weight")
    parser.add_argument("--lambda_margin", type=float, default=0.1, help="Margin loss weight")
    parser.add_argument("--delta_d", type=float, default=DELTA_D_CALIBRATED,
                         help="Margin hinge distance (default: PHASE_E12E-calibrated value targeting "
                              "~20%% active hinge pairs at initialization; the E12b pilot's delta_d=1.0 "
                              "was ~67x too large, see PHASE_E12D)")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--num_workers", type=int, default=None,
                         help="Override config's num_workers (e.g. 0 for background-launched runs -- "
                              "see windows_training_env_gotchas memory: num_workers>0 can hang silently "
                              "when this script is launched via a background process runner)")
    args = parser.parse_args()

    exp_dir = Path(__file__).parent
    trainer = EGGOMExperiment(
        args.config, exp_dir, seed=args.seed, mu=args.mu,
        lambda_margin=args.lambda_margin, delta_d=args.delta_d, run_name=args.run_name,
        num_workers=args.num_workers,
    )
    trainer.train(args.epochs)
