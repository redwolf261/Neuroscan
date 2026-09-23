"""
Phase E20: dec1 Effective-Update Decomposition -- purely diagnostic, no
training, no algorithm/hyperparameter changes to the actual model. Direct
follow-up to E19: E19 established that dec1 receives the large majority
(35-54%) of the decoder-side margin gradient norm, but that this raw
gradient becomes weak relative to L_seg (rho ~ 0.1-0.3) from epoch 5
onward, and that E19's own |delta_theta_l| (real AdamW update) could NOT
answer whether the OPTIMIZER specifically under-allocates movement to the
margin objective -- because the checkpoint's saved exp_avg/exp_avg_sq are
accumulated from the COMBINED loss, and Adam's per-parameter m/v
normalization is nonlinear, so there is no way to split real optimizer
state back into a "margin's share" after the fact. This phase closes that
gap with a clearly-labeled, bounded approximation.

Scope decision (confirmed with user before writing this script):
  - dec1 ONLY -- E19 established this is where the margin objective's
    raw gradient concentrates; other decoder blocks are a natural
    follow-up, not run here.
  - Epochs 5/10/15/20/25/30 ONLY -- epoch 1 is the pathological early-
    reorganization window E17 already explains (severe representation
    rotation, moving BN statistics, moving margin target); per the
    project's own prior framing, designing around the epoch-1 regime
    specifically would risk optimizing for a transient, not the
    steady-state problem. Excluded here, not silently forgotten.
  - Method: SHORT LOCAL REPLAY per checkpoint, not a full parallel-shadow
    retraining run. At each checkpoint, three FRESH Adam moment-accumulator
    pairs (m_seg/v_seg, m_margin/v_margin, m_total/v_total) are initialized
    at step=0 from that checkpoint's own dec1 parameters, then stepped
    together over the SAME 15 real batches (one gradient computation per
    batch, three logical accumulators updated from it), using the
    checkpoint's own real lr/beta1/beta2/eps (read from its saved
    optimizer_state, same as E19). This is DELIBERATELY an approximation,
    not a reconstruction of ground truth (unlike E19's |delta_theta_l|,
    which read real saved state) -- explicitly confirmed with the user
    before writing this script, chosen over a full 30-epoch shadow-training
    run to keep this a diagnostic, not a multi-hour commitment.

  IMPORTANT LIMITATION, stated explicitly because it changes what these
  numbers can and cannot support:
  1. Fresh accumulators start at step=0, while real training's step count
     at these checkpoints is in the hundreds-to-thousands (E19 measured
     step=705 at epoch 5, step=4230 at epoch 30). Adam's bias correction
     (1 - beta2^step) is still small and volatile over the first ~15 steps
     of ANY fresh accumulator (beta2=0.999 => 1-0.999^15 ~= 0.0149), so
     the ABSOLUTE magnitude of delta_theta_seg/delta_theta_margin from this
     replay is NOT directly comparable to E19's real |delta_theta_l|
     (which reflects a mature, thousands-of-steps-old accumulator). Only
     the RATIO between delta_theta_margin and delta_theta_seg within the
     SAME replay (both accumulators equally fresh, equally biased) is a
     fair comparison -- and that ratio is exactly what this phase's
     questions need.
  2. delta_theta_total (combined-loss accumulator, replayed the same way)
     is NOT expected to equal delta_theta_seg + delta_theta_margin exactly
     -- Adam's m/sqrt(v) normalization is nonlinear, so the combined
     accumulator's own trajectory is its own thing, not a sum of the
     other two. It is included as a THIRD, independently-replayed
     reference point (does the short local replay's combined update at
     least roughly track the real training trajectory's direction),
     not as an algebraic identity to be verified.
  3. 15 steps only sample a narrow local neighborhood of parameter space
     around each checkpoint -- this measures "if training continued from
     exactly this point using only this loss term, which direction would
     it initially go and how far," not "what actually accumulated over
     the whole 30-epoch run" (that would require the full parallel-shadow
     alternative, explicitly not chosen for this phase).

Method:
  For each of epochs {5,10,15,20,25,30}:
    1. Load the real checkpoint (model weights + real optimizer_state for
       lr/beta1/beta2/eps at that point in the schedule).
    2. Initialize THREE fresh (m,v) pairs, one per accumulator
       (seg/margin/total), each shaped like dec1's parameters, zeroed.
    3. For 15 real training-batches (same loader, same stratified anchor
       sampling as E14/E19/train_eggo_m.py):
         a. Forward pass through the CURRENT dec1 weights (a single shared
            set of weights this replay does NOT actually update in place
            for the model as a whole -- see note below).
         b. Compute seg_loss, margin_loss (unweighted, exact same formulas
            as E14/E19), and total_loss = seg_loss + lambda_margin *
            margin_loss (lambda_margin=0.1, the real calibrated E12f/E13
            value, so "total" reflects the actual applied weighting, unlike
            E14/E19's deliberately-unweighted margin_loss used for raw
            conflict/attribution analysis).
         c. torch.autograd.grad each of the three losses w.r.t. dec1's
            parameters (three independent calls on the same forward pass,
            same pattern as E14/E19).
         d. Adam-update each of the three (m,v) pairs with its own
            gradient (standard Adam moment update: m = beta1*m +
            (1-beta1)*g; v = beta2*v + (1-beta2)*g^2), and accumulate that
            step's bias-corrected delta_theta = -lr * m_hat/(sqrt(v_hat)+eps)
            into a running SUM of update vectors (not applied to the
            weights -- see note below).
    4. After 15 steps, report:
         ||sum of delta_theta_margin||, ||sum of delta_theta_seg||,
         ||sum of delta_theta_total||,
         cos(sum_delta_theta_margin, sum_delta_theta_total),
         ||projection of sum_delta_theta_margin onto sum_delta_theta_total|| / ||sum_delta_theta_total||

  NOTE on not applying updates to the model in place: dec1's actual weights
  are held FIXED at the checkpoint's values throughout all 15 steps (only
  the three shadow (m,v) accumulators evolve) -- this keeps all 15 steps'
  gradients evaluated at (approximately) the same point, which is what
  makes summing their bias-corrected deltas a meaningful "which direction
  and how far would this objective alone try to move dec1 from here"
  measurement, rather than a moving target chasing itself. Real training
  of course does apply updates every step; this replay deliberately
  trades that fidelity for a clean, comparable, apples-to-apples read at
  a fixed point, consistent with this being a local, not cumulative,
  diagnostic (see Limitation 3 above).

Deliverables: PHASE_E20_DEC1_UPDATE_DECOMPOSITION.md (written separately
from this script's output), plot, and raw per-checkpoint / per-step
statistics saved to JSON.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v2 import UNet3D_v2  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss  # noqa: E402
from Dataset.brats_dataset import create_brats_loaders  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m"))
from train_eggo_m import (  # noqa: E402
    sample_stratified_anchors, compute_margin_loss,
    DELTA_D_CALIBRATED, ANCHORS_PER_VOLUME, MAX_NEGATIVES_PER_ANCHOR,
    EVIDENCE_P99_DEFAULT, EMATauB,
)

CHECKPOINT_EPOCHS = [5, 10, 15, 20, 25, 30]  # epoch 1 deliberately excluded, see module docstring
SEED_DIR = "e12f_pilot_calibrated_seed0"
N_REPLAY_STEPS = 15
FOCAL_WEIGHT = 0.5
EVIDENTIAL_WEIGHT = 0.5
LAMBDA_MARGIN_CALIBRATED = 0.1  # real E12f/E13 applied weight, used for "total" here (unlike E14/E19's unweighted margin_loss)


class ShadowAdam:
    """Minimal, from-scratch Adam moment-accumulator for a fixed list of
    parameter tensors -- deliberately NOT torch.optim.Adam (which would
    require a full param-group/optimizer object per shadow accumulator and
    obscures the exact bias-correction arithmetic this phase needs to be
    transparent about). Matches PyTorch AdamW's default decoupled-weight-
    decay-EXCLUDED update rule exactly (same choice E19 made: weight_decay
    is omitted from delta_theta since it acts directly on theta, not
    through the gradient, and this phase cares about gradient-driven
    movement specifically)."""

    def __init__(self, params, lr, beta1, beta2, eps):
        self.params = params  # list of tensors (shapes only matter for zeros_like)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.step_count = 0
        self.delta_sum = [torch.zeros_like(p) for p in params]

    def step(self, grads):
        self.step_count += 1
        bias_c1 = 1.0 - self.beta1 ** self.step_count
        bias_c2 = 1.0 - self.beta2 ** self.step_count
        for i, g in enumerate(grads):
            if g is None:
                continue
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[i] / bias_c1
            v_hat = self.v[i] / bias_c2
            delta = -self.lr * m_hat / (v_hat.sqrt() + self.eps)
            self.delta_sum[i] = self.delta_sum[i] + delta

    def total_delta_norm(self):
        sq_sum = sum(float((d ** 2).sum()) for d in self.delta_sum)
        return sq_sum ** 0.5

    def flat_delta(self):
        return torch.cat([d.reshape(-1) for d in self.delta_sum])


def analyze_checkpoint(epoch, ckpt_dir, loader, device, n_steps, focal_fn, evidential_fn):
    ckpt_path = ckpt_dir / f"epoch_{epoch}.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.train()  # matches E14/E19: eval() mode would give a different dec1 than training-time BN saw

    dec1_params = list(model.dec1.parameters())

    pg = ckpt["optimizer_state"]["param_groups"][0]
    lr, (beta1, beta2), eps = pg["lr"], pg["betas"], pg["eps"]
    print(f"  [epoch {epoch}] real optimizer state at this checkpoint: lr={lr:.6e} beta1={beta1} beta2={beta2} eps={eps}")

    shadow_seg = ShadowAdam(dec1_params, lr, beta1, beta2, eps)
    shadow_margin = ShadowAdam(dec1_params, lr, beta1, beta2, eps)
    shadow_total = ShadowAdam(dec1_params, lr, beta1, beta2, eps)

    rng = np.random.RandomState(epoch)  # reproducible, matches E14/E19 convention
    tau_b_tracker = EMATauB()

    it = iter(loader)
    step_records = []
    n_valid = 0

    for step_idx in range(n_steps):
        try:
            images, masks, _ = next(it)
        except StopIteration:
            it = iter(loader)  # replay needs more batches than E14/E19's 8-per-checkpoint; wrap if exhausted
            images, masks, _ = next(it)
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass through dec1's CURRENT (checkpoint-fixed) weights --
        # weights are never updated in place across steps, see module
        # docstring's "NOTE on not applying updates to the model in place".
        outputs = model(images)
        probs = outputs["probs"]
        alpha, beta = outputs["alpha"], outputs["beta"]
        boundary_logit = outputs["boundary_logit"]
        dec1 = outputs["dec1"]

        B, C, D, H, W = dec1.shape
        with torch.no_grad():
            evidence_full = alpha + beta - 2.0
        dec1_perm = dec1.permute(0, 2, 3, 4, 1).reshape(-1, C)
        evidence_flat = evidence_full.reshape(-1)
        boundary_flat = boundary_logit.reshape(-1)
        gt_flat = masks.reshape(-1)

        focal_loss = focal_fn(probs, masks)
        evidential_loss = evidential_fn(alpha, beta, masks)
        seg_loss = FOCAL_WEIGHT * focal_loss + EVIDENTIAL_WEIGHT * evidential_loss

        voxels_per_vol = D * H * W
        anchor_idx_list = []
        for b in range(B):
            vol_evidence = evidence_flat[b * voxels_per_vol:(b + 1) * voxels_per_vol]
            local_idx = sample_stratified_anchors(vol_evidence, ANCHORS_PER_VOLUME, rng)
            anchor_idx_list.append(local_idx + b * voxels_per_vol)
        anchor_idx = torch.cat(anchor_idx_list)

        current_tau_b = tau_b_tracker.tau_b
        margin_loss, _, margin_diag = compute_margin_loss(
            dec1_perm, evidence_flat, boundary_flat, gt_flat,
            anchor_idx, current_tau_b, EVIDENCE_P99_DEFAULT, DELTA_D_CALIBRATED,
            MAX_NEGATIVES_PER_ANCHOR, rng, device,
        )
        tau_b_tracker.update(margin_diag["abs_boundary_logit"])

        if not (torch.isfinite(margin_loss) and torch.isfinite(seg_loss)):
            print(f"  WARNING epoch {epoch} step {step_idx}: non-finite loss, skipping")
            continue

        total_loss = seg_loss + LAMBDA_MARGIN_CALIBRATED * margin_loss

        g_seg = torch.autograd.grad(seg_loss, dec1_params, retain_graph=True, allow_unused=True)
        g_margin = torch.autograd.grad(margin_loss, dec1_params, retain_graph=True, allow_unused=True)
        g_total = torch.autograd.grad(total_loss, dec1_params, retain_graph=False, allow_unused=True)

        g_seg = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_seg, dec1_params)]
        g_margin = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_margin, dec1_params)]
        g_total = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_total, dec1_params)]

        shadow_seg.step(g_seg)
        shadow_margin.step(g_margin)
        shadow_total.step(g_total)
        n_valid += 1

        step_records.append({
            "step": step_idx,
            "G_seg": flat_grad_norm(g_seg),
            "G_margin": flat_grad_norm(g_margin),
        })

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    flat_margin = shadow_margin.flat_delta()
    flat_seg = shadow_seg.flat_delta()
    flat_total = shadow_total.flat_delta()

    def cos(a, b):
        na, nb = a.norm().item(), b.norm().item()
        if na < 1e-20 or nb < 1e-20:
            return None
        return float((a @ b).item() / (na * nb))

    result = {
        "epoch": epoch,
        "n_valid_steps": n_valid,
        "delta_margin_norm": shadow_margin.total_delta_norm(),
        "delta_seg_norm": shadow_seg.total_delta_norm(),
        "delta_total_norm": shadow_total.total_delta_norm(),
        "cos_margin_total": cos(flat_margin, flat_total),
        "cos_margin_seg": cos(flat_margin, flat_seg),
        "step_records": step_records,
    }
    # Signed scalar projection coefficient of delta_margin onto delta_total's
    # direction: (delta_margin . delta_total) / ||delta_total||^2 is the
    # coefficient c such that c * delta_total is the vector projection of
    # delta_margin onto delta_total -- e.g. c=0.5 means delta_margin's
    # component along the real combined-update direction is half of that
    # update's own length; c near 0 means margin's shadow update is nearly
    # orthogonal to where training actually moved.
    n_total_sq = float((flat_total @ flat_total).item())
    if n_total_sq > 1e-20:
        result["margin_projection_coeff_on_total"] = float((flat_margin @ flat_total).item() / n_total_sq)
    else:
        result["margin_projection_coeff_on_total"] = None

    return result


def flat_grad_norm(grads):
    sq_sum = 0.0
    for g in grads:
        if g is None:
            continue
        sq_sum += float((g.detach() ** 2).sum())
    return sq_sum ** 0.5


def main():
    exp_dir = Path(__file__).parent
    ckpt_dir = exp_dir / SEED_DIR / "checkpoints"
    out_dir = exp_dir / "e20_dec1_update_decomposition_results"
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _ = create_brats_loaders(
        batch_size=2, num_workers=0,
        root_dir=str(project_root / "Dataset" / "Training"),
        val_split=0.1,
    )

    focal_fn = FocalTverskyLoss()
    evidential_fn = EvidentialBetaLoss(weight=0.5)

    all_results = []
    for epoch in CHECKPOINT_EPOCHS:
        print(f"Analyzing epoch {epoch}...")
        result = analyze_checkpoint(epoch, ckpt_dir, train_loader, device, N_REPLAY_STEPS, focal_fn, evidential_fn)
        all_results.append(result)
        ratio = result["delta_margin_norm"] / result["delta_seg_norm"] if result["delta_seg_norm"] > 1e-20 else float("nan")
        print(f"  n_valid_steps={result['n_valid_steps']}/{N_REPLAY_STEPS}")
        print(f"  ||delta_margin||={result['delta_margin_norm']:.4e} ||delta_seg||={result['delta_seg_norm']:.4e} "
              f"||delta_total||={result['delta_total_norm']:.4e} ratio(margin/seg)={ratio:.4f}")
        print(f"  cos(delta_margin, delta_total)={result['cos_margin_total']} "
              f"cos(delta_margin, delta_seg)={result['cos_margin_seg']} "
              f"margin_projection_coeff_on_total={result['margin_projection_coeff_on_total']}")

    # --- Save raw data ---
    serializable = [
        {k: v for k, v in r.items()} for r in all_results
    ]
    with open(out_dir / "e20_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    # --- Plot ---
    epochs_arr = [r["epoch"] for r in all_results]
    margin_norms = [r["delta_margin_norm"] for r in all_results]
    seg_norms = [r["delta_seg_norm"] for r in all_results]
    total_norms = [r["delta_total_norm"] for r in all_results]
    ratios = [r["delta_margin_norm"] / r["delta_seg_norm"] if r["delta_seg_norm"] > 1e-20 else np.nan for r in all_results]
    cos_margin_total = [r["cos_margin_total"] for r in all_results]
    proj_coeff = [r["margin_projection_coeff_on_total"] for r in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs_arr, margin_norms, marker="o", label="||sum delta_theta_margin||")
    axes[0, 0].plot(epochs_arr, seg_norms, marker="s", label="||sum delta_theta_seg||")
    axes[0, 0].plot(epochs_arr, total_norms, marker="^", label="||sum delta_theta_total||")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Epoch (checkpoint replay origin)")
    axes[0, 0].set_ylabel("Summed effective update norm over 15 replay steps (log)")
    axes[0, 0].set_title("dec1 shadow-Adam update magnitude, per objective")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs_arr, ratios, marker="o", color="crimson")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("||delta_margin|| / ||delta_seg||")
    axes[0, 1].set_title("Relative effective-update allocation (dec1)")

    axes[1, 0].plot(epochs_arr, cos_margin_total, marker="o", color="steelblue")
    axes[1, 0].axhline(0, color="gray", linewidth=0.8)
    axes[1, 0].set_ylim(-1, 1)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("cos(delta_margin, delta_total)")
    axes[1, 0].set_title("Does margin's shadow update align with the real combined-loss direction?")

    axes[1, 1].plot(epochs_arr, proj_coeff, marker="o", color="darkgreen")
    axes[1, 1].axhline(0, color="gray", linewidth=0.8)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("scalar projection coeff of delta_margin onto delta_total")
    axes[1, 1].set_title("Margin's contribution along the real update direction")

    fig.suptitle("Phase E20: dec1 Effective-Update Decomposition (short local replay, 15 steps/checkpoint)")
    fig.tight_layout()
    fig.savefig(out_dir / "e20_plots.png", dpi=150)
    plt.close(fig)

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
