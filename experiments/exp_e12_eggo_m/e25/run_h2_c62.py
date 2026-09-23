"""
Phase E25, C6-2, H2: does the ACTUAL parameter-induced representation
displacement (not just the loss-level gradient) align with SC-TAM's
own signed, class-conditional axis?

Directly adapted from experiments/exp_e12_eggo_m/e24/run_h2_all_conditions.py's
mechanism (E21's exact perturb-and-forward: ShadowAdam's 15-replay-step
accumulated Delta_theta_margin on real training batches, applied
out-of-place to a deep-copied model, re-forward-passed on the SAME batch
to get the realized Delta_z) -- UNCHANGED logic, only:
  1. objective=ObjectiveConfig(mode="sc_tam") threaded through instead of
     task_aligned/random_projection.
  2. CLASS-CONDITIONAL, SIGNED reporting -- the ONE substantive adaptation
     this design doc's H2 section requires (per PHASE_E25_CANDIDATE6_
     SC_TAM_DESIGN.md's H2 adaptation): B/E's H2 only ever asked "does
     Delta_z align with the axis" (a single, class-agnostic question,
     since B/E's margin loss has no signed, per-class directional claim).
     SC-TAM's defining mechanistic claim IS class-conditional and signed
     (tumor voxels should move toward -w_hat, background voxels toward
     +w_hat -- see test_sc_tam_sign_correctness and Gate 5's criterion 9).
     So this script computes cos(Delta_z_voxel, w_hat) SEPARATELY for
     tumor-labeled and background-labeled voxels (using that batch's own
     ground-truth mask, downsampled to dec1's spatial resolution the same
     way compute_margin_loss's own anchor sampling does), and reports
     whether the SIGN of the mean cosine matches the design's directional
     prediction per class (tumor: expected NEGATIVE; background: expected
     POSITIVE) -- not just a single pooled magnitude, which would average
     away exactly the asymmetry this whole candidate exists to fix.

SCOPE: H2 computed for C6-2 only (this script), kept in a separate file
from B/E's own h2_results_{B,E}.json -- no merging, matching H1's
discipline and the project's standing "never cross-compare raw
magnitudes across different objectives" rule. Same 6 checkpoints, same
N_TRANSPORT_BATCHES=4, same N_REPLAY_STEPS=15 as B/E's H2.
"""
import sys
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v2 import UNet3D_v2  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss  # noqa: E402
from Dataset.brats_dataset import create_brats_loaders  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m"))
from train_eggo_m import (  # noqa: E402
    sample_stratified_anchors, compute_margin_loss,
    ANCHORS_PER_VOLUME, MAX_NEGATIVES_PER_ANCHOR, EVIDENCE_P99_DEFAULT, EMATauB,
)

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e24"))
from run_counterfactual import ObjectiveConfig  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m"))
from e20_dec1_update_decomposition import ShadowAdam  # noqa: E402 -- SAME class E21/H2 itself imports and uses, unchanged

CHECKPOINT_EPOCHS = [5, 10, 15, 20, 25, 30]  # frozen, matches E20/E21/H1/H2(B,E)
N_REPLAY_STEPS = 15  # frozen, matches E20/E21/H2(B,E) exactly
N_TRANSPORT_BATCHES = 4  # frozen, matches H2(B,E)'s own scope
FOCAL_WEIGHT = 0.5
EVIDENTIAL_WEIGHT = 0.5

C62_CHECKPOINT_DIR = (project_root / "experiments" / "exp_e12_eggo_m" / "e25" /
                       "c62_runs" / "C62_sc_tam_seed0" / "checkpoints")
OUT_DIR = Path(__file__).parent / "h2_results"


def compute_delta_theta_margin_for_checkpoint(epoch, ckpt_dir, loader_iter, device, n_steps, focal_fn, evidential_fn, rng, tau_b_tracker, objective):
    """Byte-identical structure to run_h2_all_conditions.py's own function
    of the same name -- ONLY difference: objective is ObjectiveConfig(mode
    ="sc_tam"), which get_w_hat()/delta_d/margin_mode already handle
    correctly via the E24-regression-verified extension."""
    ckpt_path = ckpt_dir / f"epoch_{epoch}.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.train()  # matches E14/E19/E20/E21/H2(B,E): training-time BN behavior

    dec1_params = list(model.dec1.parameters())
    pg = ckpt["optimizer_state"]["param_groups"][0]
    lr, (beta1, beta2), eps = pg["lr"], pg["betas"], pg["eps"]

    shadow_margin = ShadowAdam(dec1_params, lr, beta1, beta2, eps)

    for step_idx in range(n_steps):
        images, masks, _ = next(loader_iter)
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        alpha, beta_t = outputs["alpha"], outputs["beta"]
        boundary_logit = outputs["boundary_logit"]
        dec1 = outputs["dec1"]

        B, C, D, H, W = dec1.shape
        with torch.no_grad():
            evidence_full = alpha + beta_t - 2.0
        dec1_perm = dec1.permute(0, 2, 3, 4, 1).reshape(-1, C)
        evidence_flat = evidence_full.reshape(-1)
        boundary_flat = boundary_logit.reshape(-1)
        gt_flat = masks.reshape(-1)

        voxels_per_vol = D * H * W
        anchor_idx_list = []
        for b in range(B):
            vol_evidence = evidence_flat[b * voxels_per_vol:(b + 1) * voxels_per_vol]
            local_idx = sample_stratified_anchors(vol_evidence, ANCHORS_PER_VOLUME, rng)
            anchor_idx_list.append(local_idx + b * voxels_per_vol)
        anchor_idx = torch.cat(anchor_idx_list)

        current_tau_b = tau_b_tracker.tau_b
        w_hat = objective.get_w_hat(model, device)
        margin_loss, _, margin_diag = compute_margin_loss(
            dec1_perm, evidence_flat, boundary_flat, gt_flat,
            anchor_idx, current_tau_b, EVIDENCE_P99_DEFAULT, objective.delta_d,
            MAX_NEGATIVES_PER_ANCHOR, rng, device,
            w_hat=w_hat, margin_mode=objective.margin_mode,
        )
        tau_b_tracker.update(margin_diag["abs_boundary_logit"])

        if not torch.isfinite(margin_loss):
            continue

        g_margin = torch.autograd.grad(margin_loss, dec1_params, retain_graph=False, allow_unused=True)
        g_margin = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_margin, dec1_params)]
        shadow_margin.step(g_margin)

    return model, dec1_params, shadow_margin.delta_sum


def apply_delta_and_forward(model, delta_list, images):
    """Verbatim copy from run_h2_all_conditions.py / E21's original -- UNCHANGED."""
    model_copy = copy.deepcopy(model)
    model_copy.train()
    with torch.no_grad():
        for p, d in zip(model_copy.dec1.parameters(), delta_list):
            p.add_(d)
    with torch.no_grad():
        outputs = model_copy(images)
    del model_copy
    return outputs["dec1"]


def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _ = create_brats_loaders(
        batch_size=2, num_workers=0,
        root_dir=str(project_root / "Dataset" / "Training"),
        val_split=0.1,
    )

    focal_fn = FocalTverskyLoss()
    evidential_fn = EvidentialBetaLoss(weight=0.5)

    print("\n" + "#" * 70)
    print("# H2: condition C6-2 (objective=sc_tam), CLASS-CONDITIONAL SIGNED REPORTING")
    print("#" * 70)

    objective = ObjectiveConfig(mode="sc_tam")
    all_records = []

    for epoch in CHECKPOINT_EPOCHS:
        print(f"\n=== Checkpoint epoch {epoch} ===")
        rng = np.random.RandomState(epoch)
        tau_b_tracker = EMATauB()
        loader_iter = iter(train_loader)

        model, dec1_params, delta_theta_margin = compute_delta_theta_margin_for_checkpoint(
            epoch, C62_CHECKPOINT_DIR, loader_iter, device, N_REPLAY_STEPS, focal_fn, evidential_fn, rng, tau_b_tracker, objective
        )
        theta_norm = sum(float((d ** 2).sum()) for d in delta_theta_margin) ** 0.5
        print(f"  ||delta_theta_margin|| = {theta_norm:.4e}")

        # Axis: this checkpoint's OWN live seg_head direction (SAME
        # construction as B's task_aligned axis -- SC-TAM uses the
        # identical w_hat, only the LOSS differs).
        axis = objective.get_w_hat(model, device)
        axis_unit = axis / axis.norm().clamp_min(1e-8)

        for batch_idx in range(N_TRANSPORT_BATCHES):
            try:
                images, masks, _ = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                images, masks, _ = next(loader_iter)
            images = images.to(device)
            masks = masks.to(device)

            with torch.no_grad():
                outputs_orig = model(images)
                dec1_orig = outputs_orig["dec1"]

            dec1_perturbed = apply_delta_and_forward(model, delta_theta_margin, images)

            B, C, D, H, W = dec1_orig.shape
            dz = (dec1_perturbed - dec1_orig).permute(0, 2, 3, 4, 1).reshape(-1, C)  # (N_voxels, 32)
            gt_flat = masks.reshape(-1)  # (N_voxels,) -- SAME spatial resolution as dec1 (masks are pre-downsampled to match dec1's 64^3, per train_eggo_m.py's own convention -- gt_flat used directly against dec1-space indices exactly as compute_margin_loss itself does)
            dz_norm_per_voxel = dz.norm(dim=1)
            valid = dz_norm_per_voxel > 1e-12

            if valid.sum() == 0:
                print(f"  batch {batch_idx}: no valid (nonzero) displacement voxels, skipping")
                continue

            cos_per_voxel = F.cosine_similarity(dz[valid], axis_unit.unsqueeze(0).expand(valid.sum(), -1), dim=1)
            gt_valid = gt_flat[valid]
            tumor_mask = gt_valid > 0.5
            bg_mask = ~tumor_mask

            def summarize(cos_subset, expected_sign):
                if cos_subset.numel() == 0:
                    return None
                mean_cos = float(cos_subset.mean().item())
                return {
                    "n_voxels": int(cos_subset.numel()),
                    "cos_mean": mean_cos,
                    "cos_median": float(cos_subset.median().item()),
                    "cos_std": float(cos_subset.std().item()) if cos_subset.numel() > 1 else 0.0,
                    "frac_correct_sign": float(((cos_subset * expected_sign) > 0).float().mean().item()),
                    "mean_sign_correct": (mean_cos * expected_sign) > 0,
                }

            tumor_summary = summarize(cos_per_voxel[tumor_mask], expected_sign=-1.0)   # tumor SHOULD move toward -w_hat
            bg_summary = summarize(cos_per_voxel[bg_mask], expected_sign=+1.0)          # background SHOULD move toward +w_hat
            pooled_summary = {
                "n_voxels": int(cos_per_voxel.numel()),
                "cos_mean": float(cos_per_voxel.mean().item()),
                "cos_median": float(cos_per_voxel.median().item()),
                "cos_std": float(cos_per_voxel.std().item()),
            }

            record = {
                "condition": "C6-2",
                "epoch": epoch,
                "batch_idx": batch_idx,
                "n_valid_voxels": int(valid.sum().item()),
                "delta_theta_margin_norm": theta_norm,
                "pooled": pooled_summary,
                "tumor": tumor_summary,
                "background": bg_summary,
            }
            all_records.append(record)

            tumor_str = (f"cos_mean={tumor_summary['cos_mean']:+.4f} frac_correct_sign={tumor_summary['frac_correct_sign']:.3f} "
                         f"(n={tumor_summary['n_voxels']})") if tumor_summary else "n/a"
            bg_str = (f"cos_mean={bg_summary['cos_mean']:+.4f} frac_correct_sign={bg_summary['frac_correct_sign']:.3f} "
                      f"(n={bg_summary['n_voxels']})") if bg_summary else "n/a"
            print(f"  batch {batch_idx}: pooled_cos_mean={pooled_summary['cos_mean']:+.4f} | "
                  f"tumor(expect<0): {tumor_str} | background(expect>0): {bg_str}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    json_path = OUT_DIR / "h2_results_C62.json"
    with open(json_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nSaved {len(all_records)} records to {json_path}")

    print("\n" + "=" * 70)
    print("H2: C6-2 COMPLETE")
    print("=" * 70)
    print("NOTE: no epsilon dimension in this data (same as B/E's H2) -- ONE real accumulated")
    print("delta_theta_margin per checkpoint (15 replay steps). Results kept at per-checkpoint,")
    print("per-batch granularity, split by voxel class (tumor/background), not pooled --")
    print("pooling would average away the class-conditional sign asymmetry this design targets.")


if __name__ == "__main__":
    main()
