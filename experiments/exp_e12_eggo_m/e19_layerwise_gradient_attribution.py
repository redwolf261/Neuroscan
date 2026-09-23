"""
Phase E19: Layer-wise Parameter-Gradient Attribution -- purely diagnostic,
no training, no algorithm/hyperparameter changes. Directly follows the
Pre-E19 Review's recommendation (a): E17 showed the representation rotates
early then settles into a permanently-offset orientation; E18 showed that
freezing the decoder (the strongest rotation-reduction lever found) does
NOT restore margin-Dice coupling -- rotation is a correlate, not the
bottleneck. E14/E16 measured gradients w.r.t. the ACTIVATION dec1 only.
This phase asks the next, more specific question: which trainable
PARAMETERS does the margin objective actually try to move, how does that
compare to where the segmentation objective tries to move, and does
AdamW's adaptive normalization let those movements actually happen.

Scope decision (confirmed with user before writing this script):
  - Baseline checkpoints ONLY (e12f_pilot_calibrated_seed0 / equivalently
    e18_none_seed0), epochs 1/5/10/15/20/25/30. Matches E14/E16's own
    scope. freeze_decoder is explicitly OUT of scope for this phase --
    a natural follow-up, not run here.
  - Effective AdamW update |delta_theta_l| is computed from each
    checkpoint's OWN SAVED optimizer_state (exact exp_avg/exp_avg_sq/
    step/lr/betas/eps, verified present in checkpoint dicts before this
    script was written), reconstructing the TRUE update PyTorch actually
    applied at that training step -- not a re-simulation.
  - IMPORTANT LIMITATION, stated explicitly because it changes what the
    numbers can and cannot support: the saved exp_avg/exp_avg_sq are
    accumulated from the COMBINED loss (seg + mu*boundary + lambda*margin
    all mixed via a single backward() call in train_eggo_m.py), so there
    is NO clean way to split the resulting |delta_theta_l| back into a
    "margin's share" vs "seg's share" -- Adam's per-parameter
    normalization is nonlinear over a shared m/v state. This script does
    NOT claim to decompose the effective update by loss term. It reports
    |delta_theta_l| (real, combined-loss update) ALONGSIDE the separately
    and cleanly decomposable RAW gradient quantities (G_l^seg, G_l^margin,
    R_l^margin, rho_l, each from independent autograd.grad calls, exactly
    as E14 did for the activation-level analysis) and asks whether layers/
    epochs where rho_l is high (margin-gradient-dominant in raw gradient
    space) also show large real |delta_theta_l| -- an indirect but honest
    version of the "is there an optimizer-conditioning bottleneck" test,
    not a literal per-term Adam decomposition.

Method:
  For each of E12f seed 0's saved checkpoints (epochs 1,5,10,15,20,25,30),
  run several real training-set batches through the model (single forward
  pass per batch, same batch_size/num_workers=0 pattern as E14), then for
  EACH trainable module block l in
    {enc1, enc2, enc3, bottleneck,           <- encoder, comparison group
     upconv3, dec3, upconv2, dec2, upconv1, dec1,  <- decoder, main interest
     seg_head}
  compute, via two SEPARATE torch.autograd.grad(..., retain_graph=True)
  calls on that block's OWN parameters (not dec1's activation):

    G_l^seg    = || d L_seg    / d theta_l ||_2   (all params in block l, flattened)
    G_l^margin = || d L_margin / d theta_l ||_2

  L_seg and L_margin are the EXACT same formulas, weights, and anchor-
  sampling calls as train_eggo_m.py / E14 (FOCAL_WEIGHT=0.5,
  EVIDENTIAL_WEIGHT=0.5, unweighted-by-lambda margin_loss, same
  sample_stratified_anchors call) -- this measures exactly the objective
  the model was actually trained on.

  Then, separately, |delta_theta_l| is computed ONCE per checkpoint (not
  per batch -- it's a property of the checkpoint's saved optimizer state,
  not of any particular forward pass) by reconstructing AdamW's own update
  rule from the checkpoint's saved exp_avg/exp_avg_sq/step/lr/betas/eps:

    m_hat = exp_avg / (1 - beta1^step)
    v_hat = exp_avg_sq / (1 - beta2^step)
    delta_theta = -lr * m_hat / (sqrt(v_hat) + eps)   (weight_decay term
                                                         omitted -- decoupled
                                                         AdamW WD acts on
                                                         theta directly, not
                                                         part of the
                                                         "gradient-driven"
                                                         update this phase
                                                         cares about)

Deliverables: PHASE_E19_LAYERWISE_GRADIENT_ATTRIBUTION.md (written
separately from this script's output), plots, and raw per-checkpoint /
per-block / per-batch statistics saved to JSON.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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

CHECKPOINT_EPOCHS = [1, 5, 10, 15, 20, 25, 30]
SEED_DIR = "e12f_pilot_calibrated_seed0"
N_BATCHES_PER_CHECKPOINT = 8  # matches E14, for a per-checkpoint distribution not a single point
FOCAL_WEIGHT = 0.5      # matches EGGOMExperiment.focal_weight
EVIDENTIAL_WEIGHT = 0.5  # matches EGGOMExperiment.evidential_weight

# Block definitions -- same module names/grouping as E18's DECODER_MODULES/
# ENCODER_MODULES, extended with seg_head as a third, smaller comparison
# group. evidential_head/boundary_head deliberately excluded: L_margin has
# no direct autograd path to their parameters that isn't already captured
# via dec1 (boundary_head reads dec1.detach() -- gradient is structurally
# blocked from reaching it at all, so G_l^margin there is trivially exactly
# zero by construction, not an interesting measurement).
ENCODER_BLOCKS = ["enc1", "enc2", "enc3", "bottleneck"]
DECODER_BLOCKS = ["upconv3", "dec3", "upconv2", "dec2", "upconv1", "dec1"]
HEAD_BLOCKS = ["seg_head"]
ALL_BLOCKS = ENCODER_BLOCKS + DECODER_BLOCKS + HEAD_BLOCKS


def flat_grad_norm(grads):
    """L2 norm over ALL elements of a list of gradient tensors (one block's
    full parameter set), pooled as if concatenated into one vector -- the
    standard "block gradient magnitude" definition."""
    sq_sum = 0.0
    for g in grads:
        if g is None:
            continue
        sq_sum += float((g.detach() ** 2).sum())
    return sq_sum ** 0.5


def compute_effective_updates(model, optimizer_state):
    """Reconstructs AdamW's true per-parameter update from the checkpoint's
    OWN saved exp_avg/exp_avg_sq/step/lr/betas/eps -- the real update
    PyTorch applied at that training step, not a re-simulation. Returns
    dict: block_name -> ||delta_theta||_2 pooled over all params in block.
    """
    param_groups = optimizer_state["param_groups"]
    assert len(param_groups) == 1, f"expected single param group, got {len(param_groups)}"
    pg = param_groups[0]
    lr, (beta1, beta2), eps = pg["lr"], pg["betas"], pg["eps"]
    opt_param_ids = pg["params"]  # list of integer ids (positions), matches optimizer_state["state"] keys
    state = optimizer_state["state"]

    # Map each named parameter to its position in the optimizer's flat
    # params list, IN THE SAME ORDER model.parameters() was passed to
    # AdamW at construction time in train_eggo_m.py (standard
    # nn.Module.parameters() order, deterministic and consistent with how
    # the checkpoint was originally created).
    named_params = list(model.named_parameters())
    assert len(named_params) == len(opt_param_ids), (
        f"param count mismatch: model has {len(named_params)}, "
        f"optimizer_state has {len(opt_param_ids)} -- cannot safely align, aborting"
    )

    block_updates = {b: [] for b in ALL_BLOCKS}
    n_matched = 0
    for (name, p), pid in zip(named_params, opt_param_ids):
        if pid not in state:
            continue  # this parameter had no optimizer step yet (shouldn't happen post-epoch-1, but guard anyway)
        block = name.split(".", 1)[0]
        if block not in block_updates:
            continue  # evidential_head / boundary_head / anything else not in our block list
        s = state[pid]
        step = float(s["step"])
        exp_avg = s["exp_avg"]
        exp_avg_sq = s["exp_avg_sq"]
        bias_c1 = 1.0 - beta1 ** step
        bias_c2 = 1.0 - beta2 ** step
        m_hat = exp_avg / bias_c1
        v_hat = exp_avg_sq / bias_c2
        delta = -lr * m_hat / (v_hat.sqrt() + eps)
        block_updates[block].append(delta)
        n_matched += 1

    result = {}
    for block, deltas in block_updates.items():
        if not deltas:
            result[block] = None
            continue
        sq_sum = sum(float((d ** 2).sum()) for d in deltas)
        result[block] = sq_sum ** 0.5
    n_in_scope = sum(1 for name, _ in named_params if name.split(".", 1)[0] in block_updates)
    return result, n_matched, n_in_scope, len(named_params)


def analyze_checkpoint(epoch, ckpt_dir, loader, device, n_batches, focal_fn, evidential_fn):
    ckpt_path = ckpt_dir / f"epoch_{epoch}.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    # train() mode, matching training-time BatchNorm behavior -- same E12e
    # lesson E14 already applied: eval() mode on a mid-training checkpoint
    # gives a meaningfully different forward pass than what the optimizer
    # actually saw.
    model.train()

    # Block -> list of (name, param) for that block, used to slice
    # torch.autograd.grad's output back into per-block groups.
    block_params = {b: [] for b in ALL_BLOCKS}
    for name, p in model.named_parameters():
        top = name.split(".", 1)[0]
        if top in block_params:
            block_params[top].append(p)

    # --- Effective AdamW update magnitude, once per checkpoint (property
    # of saved optimizer state, not of any forward pass) ---
    effective_updates, n_matched, n_in_scope, n_total = compute_effective_updates(model, ckpt["optimizer_state"])
    n_excluded = n_total - n_in_scope
    print(f"  [effective updates] matched {n_matched}/{n_in_scope} in-scope params to optimizer state "
          f"({n_excluded} evidential_head/boundary_head params intentionally excluded, "
          f"{n_total} total named params)")
    if n_matched != n_in_scope:
        print(f"  WARNING: {n_in_scope - n_matched} in-scope params had NO optimizer state -- "
              f"unexpected, investigate before trusting this checkpoint's |delta_theta|")

    rng = np.random.RandomState(epoch)  # per-checkpoint but reproducible across runs, matches E14
    tau_b_tracker = EMATauB()

    batch_results = []
    it = iter(loader)

    for b_idx in range(n_batches):
        try:
            images, masks, _ = next(it)
        except StopIteration:
            break
        images = images.to(device)
        masks = masks.to(device)

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

        # --- L_seg: EXACT same formula/weights as train_eggo_m.py / E14 ---
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

        # --- L_margin: EXACT same formula as train_eggo_m.py / E14
        # (unweighted by lambda -- measuring the raw loss terms' own
        # parameter-space geometry, not the applied/weighted contribution) ---
        current_tau_b = tau_b_tracker.tau_b
        margin_loss, _, margin_diag = compute_margin_loss(
            dec1_perm, evidence_flat, boundary_flat, gt_flat,
            anchor_idx, current_tau_b, EVIDENCE_P99_DEFAULT, DELTA_D_CALIBRATED,
            MAX_NEGATIVES_PER_ANCHOR, rng, device,
        )
        tau_b_tracker.update(margin_diag["abs_boundary_logit"])

        if not (torch.isfinite(margin_loss) and torch.isfinite(seg_loss)):
            print(f"  WARNING epoch {epoch} batch {b_idx}: non-finite loss, skipping")
            continue

        # --- Independent PARAMETER-SPACE gradients per block, from the
        # SAME forward pass. allow_unused=True because seg_loss has no
        # path to boundary_head-only params (none in our block list
        # anyway) and margin_loss has no path to seg_head params (seg_head
        # only feeds probs, not dec1/margin) -- both are legitimate zero
        # gradients, not bugs, so they must not raise. ---
        all_params_ordered = []
        block_slice = {}
        cursor = 0
        for b in ALL_BLOCKS:
            plist = block_params[b]
            block_slice[b] = (cursor, cursor + len(plist))
            all_params_ordered.extend(plist)
            cursor += len(plist)

        g_seg_all = torch.autograd.grad(
            seg_loss, all_params_ordered, retain_graph=True, allow_unused=True
        )
        g_margin_all = torch.autograd.grad(
            margin_loss, all_params_ordered, retain_graph=False, allow_unused=True
        )

        block_G_seg = {}
        block_G_margin = {}
        for b in ALL_BLOCKS:
            lo, hi = block_slice[b]
            block_G_seg[b] = flat_grad_norm(g_seg_all[lo:hi])
            block_G_margin[b] = flat_grad_norm(g_margin_all[lo:hi])

        batch_results.append({
            "epoch": epoch, "batch": b_idx,
            "G_seg": block_G_seg, "G_margin": block_G_margin,
        })

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return batch_results, effective_updates


def main():
    exp_dir = Path(__file__).parent
    ckpt_dir = exp_dir / SEED_DIR / "checkpoints"
    out_dir = exp_dir / "e19_layerwise_gradient_results"
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _ = create_brats_loaders(
        batch_size=2, num_workers=0,
        root_dir=str(project_root / "Dataset" / "Training"),
        val_split=0.1,
    )

    focal_fn = FocalTverskyLoss()
    evidential_fn = EvidentialBetaLoss(weight=0.5)

    all_results = {}
    all_effective_updates = {}
    for epoch in CHECKPOINT_EPOCHS:
        print(f"Analyzing epoch {epoch}...")
        batch_results, effective_updates = analyze_checkpoint(
            epoch, ckpt_dir, train_loader, device, N_BATCHES_PER_CHECKPOINT, focal_fn, evidential_fn
        )
        all_results[epoch] = batch_results
        all_effective_updates[epoch] = effective_updates
        if not batch_results:
            print(f"  WARNING: no valid batches for epoch {epoch}")
            continue
        mean_G_margin = {b: float(np.mean([r["G_margin"][b] for r in batch_results])) for b in ALL_BLOCKS}
        mean_G_seg = {b: float(np.mean([r["G_seg"][b] for r in batch_results])) for b in ALL_BLOCKS}
        total_margin = sum(mean_G_margin.values())
        print(f"  {len(batch_results)} batches")
        for b in ALL_BLOCKS:
            rho = mean_G_margin[b] / mean_G_seg[b] if mean_G_seg[b] > 1e-12 else float("nan")
            R = mean_G_margin[b] / total_margin if total_margin > 1e-12 else float("nan")
            du = effective_updates[b]
            du_str = f"{du:.6e}" if du is not None else "N/A"
            print(f"    {b:12s} G_seg={mean_G_seg[b]:.4e} G_margin={mean_G_margin[b]:.4e} "
                  f"rho={rho:.4f} R_margin={R:.4f} |dtheta|={du_str}")

    # --- Save raw data ---
    serializable = {}
    for epoch, batch_results in all_results.items():
        serializable[str(epoch)] = [
            {"epoch": r["epoch"], "batch": r["batch"], "G_seg": r["G_seg"], "G_margin": r["G_margin"]}
            for r in batch_results
        ]
    with open(out_dir / "raw_layerwise_gradient_data.json", "w") as f:
        json.dump(serializable, f, indent=2)

    eff_serializable = {str(e): u for e, u in all_effective_updates.items()}
    with open(out_dir / "effective_updates.json", "w") as f:
        json.dump(eff_serializable, f, indent=2)

    # --- Aggregate per-epoch, per-block summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY: epoch x block -> G_seg, G_margin, rho, R_margin, |delta_theta|")
    print("=" * 70)
    epoch_summary = []
    for epoch in CHECKPOINT_EPOCHS:
        batch_results = all_results[epoch]
        if not batch_results:
            continue
        row = {"epoch": epoch, "blocks": {}}
        mean_G_margin = {b: float(np.mean([r["G_margin"][b] for r in batch_results])) for b in ALL_BLOCKS}
        std_G_margin = {b: float(np.std([r["G_margin"][b] for r in batch_results])) for b in ALL_BLOCKS}
        mean_G_seg = {b: float(np.mean([r["G_seg"][b] for r in batch_results])) for b in ALL_BLOCKS}
        std_G_seg = {b: float(np.std([r["G_seg"][b] for r in batch_results])) for b in ALL_BLOCKS}
        total_margin = sum(mean_G_margin.values())
        total_seg = sum(mean_G_seg.values())
        for b in ALL_BLOCKS:
            rho = mean_G_margin[b] / mean_G_seg[b] if mean_G_seg[b] > 1e-12 else None
            R_margin = mean_G_margin[b] / total_margin if total_margin > 1e-12 else None
            R_seg = mean_G_seg[b] / total_seg if total_seg > 1e-12 else None
            row["blocks"][b] = {
                "mean_G_seg": mean_G_seg[b], "std_G_seg": std_G_seg[b],
                "mean_G_margin": mean_G_margin[b], "std_G_margin": std_G_margin[b],
                "rho": rho, "R_margin": R_margin, "R_seg": R_seg,
                "effective_update_norm": all_effective_updates[epoch][b],
            }
        epoch_summary.append(row)

    with open(out_dir / "epoch_block_summary.json", "w") as f:
        json.dump(epoch_summary, f, indent=2)

    # --- Plots ---
    epochs_arr = [r["epoch"] for r in epoch_summary]

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    # (0,0) R_margin per decoder block over epoch -- "where does margin
    # gradient concentrate"
    for b in DECODER_BLOCKS:
        vals = [r["blocks"][b]["R_margin"] for r in epoch_summary]
        axes[0, 0].plot(epochs_arr, vals, marker="o", label=b)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("R_margin (share of total margin grad norm)")
    axes[0, 0].set_title("Margin gradient concentration across decoder blocks")
    axes[0, 0].legend(fontsize=8)

    # (0,1) rho per decoder block over epoch, log scale
    for b in DECODER_BLOCKS:
        vals = [r["blocks"][b]["rho"] for r in epoch_summary]
        axes[0, 1].plot(epochs_arr, vals, marker="o", label=b)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("rho_l = ||g_margin|| / ||g_seg|| (log scale)")
    axes[0, 1].set_title("Margin/seg gradient ratio per decoder block")
    axes[0, 1].legend(fontsize=8)

    # (0,2) encoder vs decoder vs head, pooled rho
    for group_name, group in [("encoder", ENCODER_BLOCKS), ("decoder", DECODER_BLOCKS), ("seg_head", HEAD_BLOCKS)]:
        vals = []
        for r in epoch_summary:
            g_m = sum(r["blocks"][b]["mean_G_margin"] for b in group)
            g_s = sum(r["blocks"][b]["mean_G_seg"] for b in group)
            vals.append(g_m / g_s if g_s > 1e-12 else np.nan)
        axes[0, 2].plot(epochs_arr, vals, marker="o", label=group_name)
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("pooled rho (log scale)")
    axes[0, 2].set_title("Margin/seg gradient ratio: encoder vs decoder vs seg_head")
    axes[0, 2].legend()

    # (1,0) effective |delta_theta| per decoder block over epoch, log scale
    for b in DECODER_BLOCKS:
        vals = [r["blocks"][b]["effective_update_norm"] for r in epoch_summary]
        axes[1, 0].plot(epochs_arr, vals, marker="o", label=b)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("||delta_theta_l|| (real AdamW update, log scale)")
    axes[1, 0].set_title("Effective (combined-loss) AdamW update per decoder block")
    axes[1, 0].legend(fontsize=8)

    # (1,1) scatter: rho vs effective update norm, colored by epoch, decoder blocks only
    xs, ys, cs, labels = [], [], [], []
    for r in epoch_summary:
        for b in DECODER_BLOCKS:
            rho = r["blocks"][b]["rho"]
            du = r["blocks"][b]["effective_update_norm"]
            if rho is not None and du is not None:
                xs.append(rho)
                ys.append(du)
                cs.append(r["epoch"])
                labels.append(b)
    sc = axes[1, 1].scatter(xs, ys, c=cs, cmap="viridis")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("rho_l (margin/seg gradient ratio, log)")
    axes[1, 1].set_ylabel("||delta_theta_l|| (real update, log)")
    axes[1, 1].set_title("Does margin-gradient dominance predict real movement? (decoder blocks)")
    fig.colorbar(sc, ax=axes[1, 1], label="epoch")

    # (1,2) G_margin per block, epoch 1 vs epoch 30 bar comparison
    width = 0.35
    x = np.arange(len(DECODER_BLOCKS))
    e1_row = next((r for r in epoch_summary if r["epoch"] == 1), None)
    e30_row = next((r for r in epoch_summary if r["epoch"] == 30), None)
    if e1_row and e30_row:
        e1_vals = [e1_row["blocks"][b]["mean_G_margin"] for b in DECODER_BLOCKS]
        e30_vals = [e30_row["blocks"][b]["mean_G_margin"] for b in DECODER_BLOCKS]
        axes[1, 2].bar(x - width / 2, e1_vals, width, label="epoch 1")
        axes[1, 2].bar(x + width / 2, e30_vals, width, label="epoch 30")
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(DECODER_BLOCKS, rotation=30, ha="right")
        axes[1, 2].set_yscale("log")
        axes[1, 2].set_ylabel("mean G_margin (log)")
        axes[1, 2].set_title("Margin gradient magnitude per decoder block: epoch 1 vs 30")
        axes[1, 2].legend()

    fig.suptitle("Phase E19: Layer-wise Parameter-Gradient Attribution (EGGO-M seed 0, E12f baseline checkpoints)")
    fig.tight_layout()
    fig.savefig(out_dir / "layerwise_gradient_plots.png", dpi=150)
    plt.close(fig)

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
