"""
Phase E25, C6-2.5: parameter-group localization experiment. Per the
user's explicit instruction, run BEFORE any C6-3 design: given that
PHASE_E25_C62_ISOLATION_CHECKS.md ruled out sampling bias, cross-anchor
aggregation, and AdamW's moment accumulation as the source of FN's
wrong-signed realized movement -- leaving "the network's shared
parameterization/Jacobian" as the remaining suspect -- this experiment
asks WHICH parameter group introduces the inversion.

Method: for a fixed set of real FN/FP/TP/TN anchor voxels (isolated,
single-anchor SC-TAM loss rows, same construction as the isolation
checks' Check 1/2 -- real negatives drawn from the full anchor pool,
loss/gradient attributed to ONE anchor only), compute the FULL parameter
gradient (now spanning encoder+decoder+dec1, verified by direct
inspection immediately before writing this script: SC-TAM's gradient
reaches every encoder/decoder parameter via the standard backward pass,
NOT confined to dec1's own local parameters -- only the three output
heads, seg_head/evidential_head/boundary_head, correctly receive zero
gradient, per the E11.5 detach audit). Then, for each of 5 parameter-
group configurations, mask the gradient to ONLY that group, apply a
tiny controlled perturbation (pure gradient descent, no AdamW -- matches
Check 2's own isolated-probe convention), and measure the resulting
Delta_z at the SAME target voxel via a real forward pass.

Parameter groups tested (per the user's specification):
  - dec1 only:            model.dec1's own parameters (the FINAL decoder
                           block, whose output IS z -- not the whole
                           decoder)
  - decoder:               upconv3+dec3+upconv2+dec2+upconv1+dec1 (every
                           decoder-side parameter, including dec1 itself)
  - encoder:               enc1+pool1+enc2+pool2+enc3+pool3+bottleneck
                           (pool layers have no parameters but are listed
                           for completeness/clarity)
  - encoder+decoder:        both of the above combined (equivalent to
                           "all trainable parameters that receive a
                           nonzero SC-TAM gradient" -- verified this
                           EXCLUDES only the 3 output heads, which is
                           consistent with "all" below)
  - all:                   every model parameter (matches C6-2's own
                           real training update scope, i.e. what
                           Check 2's realized-trajectory measurement
                           already used) -- the existing baseline/
                           reference condition, not a new measurement,
                           included here for direct side-by-side
                           comparison at IDENTICAL epsilon/voxel/anchor
                           construction.

  seg_head ALONE is included as an explicit NULL CONTROL (not one of the
  5 "real" conditions in the table above) -- verified analytically before
  writing this script that SC-TAM's loss has NO gradient path into
  seg_head/evidential_head/boundary_head at all (all three are outside
  its backward graph, by the E11.5 detach-audit design), so this
  condition is expected, by construction, to produce EXACTLY zero
  Delta_z everywhere -- included purely as a sanity check that the
  masking/perturbation harness itself works, not as a real localization
  data point.

Finite-difference control (per the user's explicit requirement, run and
verified BEFORE trusting the localization sweep -- see module-level
smoke test in PHASE_E25 session log): confirmed |Delta_z|/epsilon
converges to a stable value as epsilon shrinks (0.1->0.01->0.001 gives
86.3->174.6->186.8, converging not diverging), and the measured sign
matched the correctly-predicted sign for a real background-class control
voxel -- confirms Delta_z behaves as a genuine first-order linear
response to the parameter perturbation, not a ShadowAdam or measurement
artifact. This experiment reuses that SAME validated mechanism
(apply_delta_and_forward + a small, fixed epsilon), just with the
gradient masked to different parameter subsets before each perturbation.
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
from Dataset.brats_dataset import create_brats_loaders  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m"))
from train_eggo_m import sample_stratified_anchors, ANCHORS_PER_VOLUME, EMATauB  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e24"))
from run_counterfactual import ObjectiveConfig  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e25"))
from run_c62_isolation_checks import _sc_tam_per_anchor_losses, classify_voxels, CAT_NAMES  # noqa: E402

CHECKPOINT_EPOCHS = [5, 10, 15, 20, 25, 30]
N_PER_CAT = 3  # individual voxels probed per category per checkpoint (5 groups x 4 real categories x 3 voxels x 6 checkpoints = 360 forward/backward passes -- kept modest since this is a targeted localization probe, not a full statistical run)
EPS_PROBE = 0.01  # SAME epsilon as the isolation checks' Check 2 probe -- keeps this experiment's numbers on a comparable scale to already-reported results

C62_CHECKPOINT_DIR = (project_root / "experiments" / "exp_e12_eggo_m" / "e25" /
                       "c62_runs" / "C62_sc_tam_seed0" / "checkpoints")
OUT_DIR = Path(__file__).parent / "jacobian_localization_results"

# Named submodules, per direct inspection of neuroscan_3d_fixed.py's
# UNet3D.__init__ (base class) + neuroscan_3d_v2.py's UNet3D_v2 (adds
# boundary_head only) -- verified against the real source immediately
# before writing this script, not assumed from memory.
ENCODER_MODULES = ["enc1", "enc2", "enc3", "bottleneck"]  # pool1/pool2/pool3 have no parameters (nn.MaxPool3d)
DECODER_MODULES = ["upconv3", "dec3", "upconv2", "dec2", "upconv1", "dec1"]  # dec1 IS the final decoder block, included here AND tested alone below
HEAD_MODULES = ["seg_head", "evidential_head", "boundary_head"]

PARAM_GROUPS = {
    "dec1_only": ["dec1"],
    "decoder": DECODER_MODULES,
    "encoder": ENCODER_MODULES,
    "encoder_and_decoder": ENCODER_MODULES + DECODER_MODULES,
    "all": None,  # None = every model parameter, no masking
    "seg_head_only_NULL_CONTROL": ["seg_head"],  # expected to be a hard zero, by construction
}


def get_param_group_mask(model, module_names):
    """Returns a list of bool masks, one per model.parameters() entry (in
    the SAME order autograd.grad returns gradients), True where that
    parameter belongs to one of the named top-level submodules. If
    module_names is None, returns all-True (every parameter, i.e. no
    masking -- the 'all' condition)."""
    if module_names is None:
        return [True for _ in model.parameters()]
    mask = []
    for name, _ in model.named_parameters():
        top = name.split(".")[0]
        mask.append(top in module_names)
    return mask


def apply_delta_and_forward(model_in, delta_list, images_in):
    """Verbatim mechanism from H2/E21/the isolation checks -- UNCHANGED,
    the SAME validated apply-and-reforward used throughout this phase,
    now applied to a MASKED delta_list (zeros outside the target
    parameter group) rather than a full-model delta."""
    model_copy = copy.deepcopy(model_in)
    model_copy.train()
    with torch.no_grad():
        for p, d in zip(model_copy.parameters(), delta_list):
            p.add_(d)
    with torch.no_grad():
        out = model_copy(images_in)
    del model_copy
    return out["dec1"]


def run_checkpoint(epoch, device, train_loader):
    ckpt = torch.load(C62_CHECKPOINT_DIR / f"epoch_{epoch}.pth", map_location=device, weights_only=False)
    model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.train()

    objective = ObjectiveConfig(mode="sc_tam")
    w_hat = objective.get_w_hat(model, device)
    tau_b_tracker = EMATauB()
    rng = np.random.RandomState(epoch)

    images, masks, _ = next(iter(train_loader))
    images = images.to(device)
    masks = masks.to(device)

    with torch.enable_grad():
        outputs = model(images)
        dec1 = outputs["dec1"]
        alpha, beta = outputs["alpha"], outputs["beta"]
        with torch.no_grad():
            evidence_full = (alpha + beta - 2.0)
            probs = outputs["probs"]
            pred_binary = (probs >= 0.5).float()
        boundary_logit = outputs["boundary_logit"]

        B, C, D, H, W = dec1.shape
        dec1_perm = dec1.permute(0, 2, 3, 4, 1).reshape(-1, C)
        evidence_flat = evidence_full.reshape(-1)
        boundary_flat = boundary_logit.reshape(-1)
        gt_flat = masks.reshape(-1)
        pred_flat = pred_binary.reshape(-1)
        cat_flat = classify_voxels(gt_flat, pred_flat)

        voxels_per_vol = D * H * W
        anchor_idx_list = []
        for b in range(B):
            vol_evidence = evidence_flat[b * voxels_per_vol:(b + 1) * voxels_per_vol]
            local_idx = sample_stratified_anchors(vol_evidence, ANCHORS_PER_VOLUME, rng)
            anchor_idx_list.append(local_idx + b * voxels_per_vol)
        anchor_idx = torch.cat(anchor_idx_list)
        anchors_cat = cat_flat[anchor_idx]

        torch.manual_seed(0)
        per_anchor_loss = _sc_tam_per_anchor_losses(
            dec1_perm, evidence_flat, boundary_flat, gt_flat,
            anchor_idx, tau_b_tracker.tau_b, w_hat, objective.delta_d, device,
        )

    all_params = list(model.parameters())
    named_params = list(model.named_parameters())
    axis_unit = w_hat / w_hat.norm().clamp_min(1e-8)

    with torch.no_grad():
        dec1_orig = model(images)["dec1"]

    # Pick up to N_PER_CAT ACTIVE (nonzero-loss) anchors per category
    records = []
    for cat_id in (0, 1, 2, 3):  # TN, TP, FP, FN
        cat_local_idx = torch.where(anchors_cat == cat_id)[0]
        active_local = cat_local_idx[per_anchor_loss[cat_local_idx] > 0]
        if active_local.numel() == 0:
            continue
        chosen = active_local[torch.randperm(active_local.numel())[:N_PER_CAT]]

        for local_i in chosen.tolist():
            global_voxel_idx = int(anchor_idx[local_i].item())
            g_full = torch.autograd.grad(per_anchor_loss[local_i], all_params, retain_graph=True, allow_unused=True)
            g_full = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_full, all_params)]
            g_norm_all = sum(float((g ** 2).sum()) for g in g_full) ** 0.5
            if g_norm_all < 1e-12:
                continue

            voxel_result = {
                "epoch": epoch, "category": CAT_NAMES[cat_id], "voxel_idx": global_voxel_idx,
                "g_norm_full_model": g_norm_all,
                "pathways": {},
            }

            for group_name, module_names in PARAM_GROUPS.items():
                mask = get_param_group_mask(model, module_names)
                g_masked = [g if m else torch.zeros_like(g) for g, m in zip(g_full, mask)]
                g_masked_norm = sum(float((g ** 2).sum()) for g in g_masked) ** 0.5

                if g_masked_norm < 1e-12:
                    # Expected for seg_head_only_NULL_CONTROL -- confirms
                    # the masking harness correctly finds zero gradient
                    # exactly where the detach audit predicts it should.
                    voxel_result["pathways"][group_name] = {
                        "g_masked_norm": g_masked_norm,
                        "delta_z_norm": 0.0,
                        "cos_with_w_hat": None,
                        "note": "zero gradient in this parameter group (expected for the null control)",
                    }
                    continue

                delta = [-EPS_PROBE * g for g in g_masked]
                dec1_pert = apply_delta_and_forward(model, delta, images)
                with torch.no_grad():
                    dz_full = (dec1_pert - dec1_orig).permute(0, 2, 3, 4, 1).reshape(-1, C)
                    dz_target = dz_full[global_voxel_idx]
                    dz_norm = float(dz_target.norm().item())
                    if dz_norm < 1e-12:
                        cos_w = None
                    else:
                        cos_w = float(F.cosine_similarity(dz_target.unsqueeze(0), axis_unit.unsqueeze(0)).item())

                voxel_result["pathways"][group_name] = {
                    "g_masked_norm": g_masked_norm,
                    "delta_z_norm": dz_norm,
                    "cos_with_w_hat": cos_w,
                }

            records.append(voxel_result)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records


def expected_sign(category):
    """TP/FN (GT=tumor) expect Delta_z toward -w_hat (cos<0); TN/FP
    (GT=background) expect +w_hat (cos>0) -- SC-TAM's own ground-truth-
    conditioned prediction, same convention as every prior phase in E25."""
    return -1.0 if category in ("TP", "FN") else 1.0


def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _ = create_brats_loaders(
        batch_size=2, num_workers=0,
        root_dir=str(project_root / "Dataset" / "Training"),
        val_split=0.1,
    )

    all_records = []
    for epoch in CHECKPOINT_EPOCHS:
        print(f"\n=== Checkpoint epoch {epoch} ===")
        records = run_checkpoint(epoch, device, train_loader)
        all_records.extend(records)
        for r in records:
            exp_sign = expected_sign(r["category"])
            line = f"  {r['category']} voxel={r['voxel_idx']}: "
            for group_name in PARAM_GROUPS:
                pw = r["pathways"][group_name]
                if pw["cos_with_w_hat"] is None:
                    line += f"{group_name}=n/a "
                else:
                    correct = (pw["cos_with_w_hat"] * exp_sign) > 0
                    mark = "+" if correct else "-"
                    line += f"{group_name}={pw['cos_with_w_hat']:+.3f}({mark}) "
            print(line)

    json_path = OUT_DIR / "jacobian_localization_C62.json"
    with open(json_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nSaved {len(all_records)} voxel records to {json_path}")


if __name__ == "__main__":
    main()
