"""
Phase E24, Gate 6, H2: does the ACTUAL parameter-induced representation
displacement (not just the loss-level gradient) align with the axis each
objective's margin term was constrained to?

Reuses E21's EXACT perturb-and-forward mechanism (ShadowAdam's 15-replay-
step accumulated Delta_theta_margin, computed on real training batches,
then applied out-of-place to a deep-copied model's dec1 parameters,
re-forward-passed on the SAME batch to get the realized Delta_z) --
UNCHANGED logic, only objective_config threaded through so B/E's own
compute_margin_loss calls use their own w_hat/delta_d (never
cross-evaluated, matching H1's discipline).

SCOPE, confirmed with user before writing this script:
  - H2 computed for B and E only. A's margin gradient (baseline Euclidean)
    is never constrained to a single fixed axis -- there is no natural
    "w_hat" for A to project onto in the same sense H2 asks about for B/E.
  - NO epsilon sweep. E21's mechanism produces ONE real accumulated
    Delta_theta_margin per checkpoint per batch (not a magnitude-scaled
    family like E22's counterfactual grid) -- reported per-checkpoint,
    per-subject/batch granularity, which ARE real dimensions in E21's
    mechanism, with an explicit note that no epsilon dimension exists
    here (not silently omitted, not invented).
  - Same 6 checkpoints as E20/E21/H1 (epochs 5,10,15,20,25,30).
  - Same N_TRANSPORT_BATCHES=4 real training batches per checkpoint as
    E21's own original scope.
  - Axis per condition: B's axis = w_hat (that checkpoint's OWN live
    seg_head direction, recomputed fresh per checkpoint, matching how
    B's real training computed it). E's axis = r (the FIXED frozen
    random vector from that condition's own checkpoint metadata --
    loaded directly from the checkpoint's saved r_vector field, not
    regenerated, per Gate 6's own "do not regenerate r" discipline).

Output preserved at full granularity (not collapsed to one mean
immediately, per explicit instruction): per-checkpoint, per-batch
cos(Delta_z_actual, axis) values, plus summary statistics (mean, median,
std, n) computed AFTER the raw values are saved.
"""
import sys
import copy
import json
from pathlib import Path

import numpy as np
import torch

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
from e20_dec1_update_decomposition import ShadowAdam  # noqa: E402 -- SAME class E21 itself imports and uses, unchanged

CHECKPOINT_EPOCHS = [5, 10, 15, 20, 25, 30]  # frozen, matches E20/E21/H1
N_REPLAY_STEPS = 15  # frozen, matches E20/E21 exactly
N_TRANSPORT_BATCHES = 4  # frozen, matches E21's own original scope
FOCAL_WEIGHT = 0.5
EVIDENTIAL_WEIGHT = 0.5

GATE6_RUNS_DIR = project_root / "experiments" / "exp_e12_eggo_m" / "e24" / "gate6_runs"
OUT_DIR = Path(__file__).parent / "h2_results"

CONDITIONS = [
    ("B", "task_aligned", GATE6_RUNS_DIR / "B_task_aligned_seed0" / "checkpoints"),
    ("E", "random_projection", GATE6_RUNS_DIR / "E_random_projection_seed0" / "checkpoints"),
]


def compute_delta_theta_margin_for_checkpoint(epoch, ckpt_dir, loader_iter, device, n_steps, focal_fn, evidential_fn, rng, tau_b_tracker, objective):
    """Same structure as E21's compute_delta_theta_for_checkpoint, but
    ONLY computes delta_theta_margin (not seg/total -- H2 only needs the
    margin update's realized effect), and threads objective through the
    compute_margin_loss call (the ONLY change from E21's own logic)."""
    ckpt_path = ckpt_dir / f"epoch_{epoch}.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.train()  # matches E14/E19/E20/E21: training-time BN behavior

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
        w_hat = objective.get_w_hat(model, device)  # ONLY CHANGE from E21: objective-aware w_hat/delta_d
        margin_loss, _, margin_diag = compute_margin_loss(
            dec1_perm, evidence_flat, boundary_flat, gt_flat,
            anchor_idx, current_tau_b, EVIDENCE_P99_DEFAULT, objective.delta_d,
            MAX_NEGATIVES_PER_ANCHOR, rng, device,
            w_hat=w_hat,
        )
        tau_b_tracker.update(margin_diag["abs_boundary_logit"])

        if not torch.isfinite(margin_loss):
            continue

        g_margin = torch.autograd.grad(margin_loss, dec1_params, retain_graph=False, allow_unused=True)
        g_margin = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_margin, dec1_params)]
        shadow_margin.step(g_margin)

    return model, dec1_params, shadow_margin.delta_sum


def apply_delta_and_forward(model, delta_list, images):
    """Verbatim copy from e21_parameter_to_representation_transport.py -- UNCHANGED."""
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

    for condition_label, mode, checkpoint_dir in CONDITIONS:
        print("\n" + "#" * 70)
        print(f"# H2: condition {condition_label} (objective={mode})")
        print("#" * 70)

        objective = ObjectiveConfig(mode=mode)

        # For E: the axis is the FIXED r loaded from the checkpoint's own
        # saved metadata (not regenerated) -- verified to match
        # ObjectiveConfig's own construction (same seed=999001) as an
        # extra consistency check before trusting it.
        if mode == "random_projection":
            sample_ckpt = torch.load(checkpoint_dir / f"epoch_{CHECKPOINT_EPOCHS[0]}.pth", map_location=device, weights_only=False)
            r_from_checkpoint = torch.tensor(sample_ckpt["r_vector"], device=device)
            r_from_objective = objective._frozen_random_w_hat.to(device)
            assert torch.allclose(r_from_checkpoint, r_from_objective, atol=1e-6), (
                "FATAL: r loaded from checkpoint does not match ObjectiveConfig's own construction -- "
                "H2's axis for condition E would not be the axis E actually trained under"
            )
            print(f"Verified: checkpoint's saved r_vector matches ObjectiveConfig(mode='random_projection') exactly")

        all_records = []

        for epoch in CHECKPOINT_EPOCHS:
            print(f"\n=== Checkpoint epoch {epoch} ===")
            rng = np.random.RandomState(epoch)
            tau_b_tracker = EMATauB()
            loader_iter = iter(train_loader)

            model, dec1_params, delta_theta_margin = compute_delta_theta_margin_for_checkpoint(
                epoch, checkpoint_dir, loader_iter, device, N_REPLAY_STEPS, focal_fn, evidential_fn, rng, tau_b_tracker, objective
            )
            theta_norm = sum(float((d ** 2).sum()) for d in delta_theta_margin) ** 0.5
            print(f"  ||delta_theta_margin|| = {theta_norm:.4e}")

            # Get this checkpoint's axis (recomputed fresh for B, since
            # w_hat depends on the model's CURRENT seg_head weight; fixed
            # for E, loaded from the checkpoint / ObjectiveConfig, same
            # object for every checkpoint by construction).
            axis = objective.get_w_hat(model, device)

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
                dz_norm_per_voxel = dz.norm(dim=1)
                valid = dz_norm_per_voxel > 1e-12

                if valid.sum() == 0:
                    print(f"  batch {batch_idx}: no valid (nonzero) displacement voxels, skipping")
                    continue

                axis_unit = axis / axis.norm().clamp_min(1e-8)
                cos_per_voxel = torch.nn.functional.cosine_similarity(dz[valid], axis_unit.unsqueeze(0).expand(valid.sum(), -1), dim=1)

                record = {
                    "condition": condition_label,
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "n_valid_voxels": int(valid.sum().item()),
                    "delta_theta_margin_norm": theta_norm,
                    "cos_mean": float(cos_per_voxel.mean().item()),
                    "cos_median": float(cos_per_voxel.median().item()),
                    "cos_std": float(cos_per_voxel.std().item()),
                    "cos_min": float(cos_per_voxel.min().item()),
                    "cos_max": float(cos_per_voxel.max().item()),
                    # ADDED per follow-up analysis request: voxel-level fraction
                    # statistics, computed directly from cos_per_voxel (not
                    # approximated from a normal-distribution assumption --
                    # mean/median already show visible skew, so a Gaussian
                    # approximation would be a real methodological shortcut here).
                    "frac_cos_gt_0": float((cos_per_voxel > 0).float().mean().item()),
                    "frac_cos_gt_0.5": float((cos_per_voxel > 0.5).float().mean().item()),
                    "frac_cos_lt_0": float((cos_per_voxel < 0).float().mean().item()),
                }
                all_records.append(record)
                print(f"  batch {batch_idx}: n_valid={record['n_valid_voxels']} "
                      f"cos_mean={record['cos_mean']:+.4f} cos_median={record['cos_median']:+.4f} "
                      f"cos_std={record['cos_std']:.4f}")

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        json_path = OUT_DIR / f"h2_results_{condition_label}.json"
        with open(json_path, "w") as f:
            json.dump(all_records, f, indent=2)
        print(f"\nSaved {len(all_records)} records to {json_path}")

    print("\n" + "=" * 70)
    print("H2: ALL CONDITIONS (B, E) COMPLETE")
    print("=" * 70)
    print("NOTE: no epsilon dimension in this data -- E21's mechanism produces ONE real")
    print("accumulated delta_theta_margin per checkpoint (15 replay steps), not a magnitude sweep.")
    print("Results kept at per-checkpoint, per-batch granularity, not collapsed to one mean.")


if __name__ == "__main__":
    main()
