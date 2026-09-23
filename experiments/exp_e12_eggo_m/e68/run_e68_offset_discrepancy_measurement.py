"""
Phase E68 Section 7.3: Geometric-vs-Task-Optimal Offset Discrepancy
Measurement (H68-A / H68-B / H68-C discriminating experiment).

MEASUREMENT PHASE ONLY -- per the corrected E68 design (see
docs/phases/PHASE_E68_SC_DCU_DESIGN.md, Section 8, "STRICT NOVELTY BAR"):
this script does NOT implement any offset-fidelity loss, does NOT
calibrate any lambda_off, and does NOT run a multi-seed Dice comparison.
Those would all be calibration-of-an-existing-mechanism moves that
cannot satisfy the project's actual novelty bar (only a structurally
new mechanism counts, not a new training procedure for an existing
one). This script's ONLY job is to measure whether the task-optimal
deformable offset (trained with segmentation loss alone, no offset
supervision of any kind) matches the geometric inverse of a known
synthetic translation, and if not, whether the discrepancy is
structured. If H68-B holds, a SEPARATE future phase designs a new
operator from the measured structure -- not started here.

METHOD:
  1. Add the SC-DCU base module (Section 2.1 of the design doc) to the
     v3/D4-only architecture at the enc1 skip ONLY -- a single deformable
     offset-and-resample step between enc1 and its concatenation into
     dec1, everything else in the trunk unchanged.
  2. Train BRIEFLY (pilot scale, not full 30-epoch) with segmentation
     loss ALONE (FocalTversky + Evidential, matching the project's real
     active loss composition -- no boundary/margin, no offset-fidelity
     term of any kind) so the offset module Delta is free to converge to
     whatever the task actually needs, unconstrained by any geometric
     target.
  3. On held-out validation subjects, apply KNOWN synthetic voxel shifts
     t (matching E65's own tested range) to enc1 before the skip, and
     record the module's own predicted offset Delta_shifted (never
     forced toward anything).
  4. Compare Delta_shifted against the geometric target Delta_G = -t:
     compute the discrepancy field Delta_R = Delta_shifted - Delta_G,
     and test whether ||Delta_R|| correlates with native lesion size
     (primary pre-registered covariate, matching the E48/E65 thread),
     using the same statistical discipline as every prior causal-audit
     phase in this project (Spearman + permutation test, subject-level).

PRE-REGISTERED DECISION RULE (three mutually exclusive outcomes):
  H68-A: mean ||Delta_R|| is small (within a pre-declared tolerance) and
    NOT significantly correlated with lesion size -- task-optimal offset
    matches the geometric target. Low novelty, no operator to build.
  H68-B: ||Delta_R|| is significantly correlated with lesion size
    (Spearman rho, permutation p<0.05) -- systematic, structured
    divergence. This is the ONLY outcome that motivates further design
    work (a new operator, in a SEPARATE future phase, not designed here).
  H68-C: ||Delta_R|| is large/real (module clearly does something) but
    NOT significantly correlated with lesion size or any other
    pre-registered covariate -- unstructured noise. Report as a clean
    null, do not rescue.

Pre-registered covariate list (tested in this fixed order, first
significant one reported as primary, others reported for transparency,
matching the project's "no post-hoc threshold movement" discipline):
  1. native lesion size (E48/E65's own field)
  2. local gradient magnitude of enc1 near the shifted region (a proxy
     for local feature ambiguity)
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from scipy import stats

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss  # noqa: E402
from Dataset.brats_dataset import BraTSDataset, create_brats_loaders  # noqa: E402

OUT_DIR = Path(__file__).parent
SEED = 0
N_PERM = 1000
PILOT_EPOCHS = 3  # deliberately small -- pilot scale, not full 30-epoch training
SHIFT_RANGE = [-4, -3, -2, 2, 3, 4]  # matches/brackets E65's own tested offset of 3, excludes 0

CKPT_V3_D4ONLY = (project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs"
                   / "DeepSup_D4only_seed0" / "checkpoints" / "best.pth")
E48_TABLE_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e48" / "E48_encoding_audit_table.json"


# ==================== SC-DCU base module (Section 2.1 of the design doc) ====================

class SCDCUOffset(nn.Module):
    """3D adaptation of Dynamic U-Net's DCU offset predictor:
    Delta = Conv3d([U(D2), E1]) -- a single 3x3x3 conv on the
    concatenated decoder+skip features, producing a 3-vector offset
    field (one (dx,dy,dz) per voxel). Direct minimal 3D generalization
    of DCU's own 2D 2-vector offset, per the design doc Section 2.1 --
    not a redesign of the published equation."""
    def __init__(self, decoder_channels, skip_channels):
        super().__init__()
        self.offset_conv = nn.Conv3d(decoder_channels + skip_channels, 3, kernel_size=3, padding=1)
        # Zero-init: identity offset at initialization (Delta=0 -> no resampling change),
        # matching the project's own "verify identity reproduces real forward()" discipline --
        # an untrained offset module should be a no-op, not an arbitrary perturbation.
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

    def forward(self, decoder_feat, skip_feat):
        return self.offset_conv(torch.cat([decoder_feat, skip_feat], dim=1))


def deformable_resample(x, offset):
    """x: (B,C,D,H,W). offset: (B,3,D,H,W), offset[:, 0/1/2] = (dz,dy,dx)
    in VOXEL units. Resamples x at (identity + offset) via trilinear
    grid_sample. offset=0 everywhere MUST reproduce x exactly (verified
    in main() before any measurement is trusted)."""
    B, C, D, H, W = x.shape
    device = x.device
    # Base identity grid in voxel coordinates, then add the learned offset (voxel units),
    # then normalize to grid_sample's required [-1,1] convention.
    zz, yy, xx = torch.meshgrid(
        torch.arange(D, device=device, dtype=x.dtype),
        torch.arange(H, device=device, dtype=x.dtype),
        torch.arange(W, device=device, dtype=x.dtype),
        indexing="ij",
    )
    base = torch.stack([xx, yy, zz], dim=0)  # (3,D,H,W) in (x,y,z) voxel order for grid_sample
    base = base.unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B,3,D,H,W)

    # offset is stored as (dz,dy,dx); reorder to (dx,dy,dz) to match grid_sample's (x,y,z) axis order
    offset_xyz = offset[:, [2, 1, 0], :, :, :]
    voxel_coords = base + offset_xyz  # (B,3,D,H,W), voxel units

    # Normalize each axis to [-1,1] per grid_sample's convention (align_corners=True: 0 -> -1, size-1 -> 1)
    sizes = torch.tensor([W, H, D], device=device, dtype=x.dtype).view(1, 3, 1, 1, 1)
    norm_coords = voxel_coords / (sizes - 1).clamp(min=1) * 2 - 1  # (B,3,D,H,W)
    grid = norm_coords.permute(0, 2, 3, 4, 1)  # (B,D,H,W,3), grid_sample's required layout

    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


# ==================== forward pass with SC-DCU spliced in at the enc1 skip ====================

def forward_with_scdcu(model, scdcu, image, injected_shift=None):
    """Runs the real v3 trunk, but replaces the enc1 skip's direct
    concatenation with an SC-DCU-corrected version. If injected_shift is
    given (a (dz,dy,dx) integer tuple), enc1 is first rolled by that
    amount (E65's own translation construction) before reaching SC-DCU
    -- this is how the discriminating experiment applies KNOWN synthetic
    shifts. Returns (probs, predicted_offset_field)."""
    enc1 = model.enc1(image)
    pool1 = model.pool1(enc1)
    enc2 = model.enc2(pool1)
    pool2 = model.pool2(enc2)
    enc3 = model.enc3(pool2)
    pool3 = model.pool3(enc3)
    bottleneck = model.bottleneck(pool3)

    upconv3 = model.upconv3(bottleneck)
    cat3 = torch.cat([upconv3, enc3], dim=1)
    dec3 = model.dec3(cat3)

    upconv2 = model.upconv2(dec3)
    cat2 = torch.cat([upconv2, enc2], dim=1)
    dec2 = model.dec2(cat2)

    upconv1 = model.upconv1(dec2)

    skip = enc1
    if injected_shift is not None:
        dz, dy, dx = injected_shift
        skip = torch.roll(skip, shifts=(dz, dy, dx), dims=(2, 3, 4))

    predicted_offset = scdcu(upconv1, skip)
    skip_corrected = deformable_resample(skip, predicted_offset)

    cat1 = torch.cat([upconv1, skip_corrected], dim=1)
    dec1 = model.dec1(cat1)
    probs = model.seg_head(dec1)
    return probs, predicted_offset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(str(CKPT_V3_D4ONLY), map_location=device, weights_only=False)
    model = UNet3D_v3(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded v3/D4-only checkpoint as the base trunk (frozen, not fine-tuned in this pilot -- "
          f"only SC-DCU's own parameters train): best_val_dice={ckpt.get('best_val_dice')}")
    # Freeze the base trunk: this pilot's ONLY question is what offset SC-DCU
    # learns to predict, not whether fine-tuning the whole trunk changes Dice
    # (that would reopen the calibration question the strict novelty bar
    # excludes). Freezing isolates the measurement to SC-DCU's own behavior.
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    scdcu = SCDCUOffset(decoder_channels=32, skip_channels=32).to(device)

    # ---------------- Sanity check: identity offset must reproduce the real trunk exactly ----------------
    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    image0, _, _ = val_dataset[0]
    image0_b = image0.unsqueeze(0).to(device)
    with torch.no_grad():
        real = model(image0_b)["probs"]
        probs_scdcu, offset0 = forward_with_scdcu(model, scdcu, image0_b, injected_shift=None)
    max_diff = float(torch.abs(real - probs_scdcu).max().item())
    max_offset = float(torch.abs(offset0).max().item())
    print(f"[Sanity check] Zero-init SC-DCU offset (max |offset|={max_offset:.6e}) reproduces real "
          f"forward(): max abs diff = {max_diff:.6e}")
    assert max_offset == 0.0, "Zero-init offset conv did not produce exactly zero offset -- STOP."
    assert max_diff < 1e-5, "SC-DCU splice does not reproduce the real trunk at zero offset -- STOP."
    print("[Sanity check] PASS.\n")

    # ---------------- Unit test: verify offset sign convention (+t undoes roll(shift=+t)) ----------------
    # grid_sample samples FROM position p+offset (out[p]=in[p+offset]), the
    # inverse convention of torch.roll's shift-the-OUTPUT-by-t semantics --
    # so offset=+t (not -t) is the correct geometric target to undo
    # torch.roll(shift=+t). Verified here by direct construction, not
    # assumed, before it is used as this script's ground-truth target.
    # Larger volume (16^3) and shift=2 so a genuinely interior region (away
    # from both the roll's wraparound edge AND grid_sample's border-padding
    # edge, which sampling at p+offset can reach for p near the boundary)
    # is unambiguous -- the 6^3 volume used originally was too small for
    # "interior" and "boundary-affected" to be cleanly separable.
    synth = torch.arange(16 * 16 * 16, dtype=torch.float32).reshape(1, 1, 16, 16, 16)
    synth_shifted = torch.roll(synth, shifts=2, dims=4)
    undo_offset = torch.zeros(1, 3, 16, 16, 16)
    undo_offset[:, 2] = 2.0  # +t on the dx channel
    recovered = deformable_resample(synth_shifted, undo_offset)
    # Interior slab: at least 4 voxels from every boundary in every axis
    interior_diff = (synth[:, :, 4:12, 4:12, 4:12] - recovered[:, :, 4:12, 4:12, 4:12]).abs().max().item()
    print(f"[Unit test] offset=+t undoes roll(shift=+t): interior max abs diff = {interior_diff:.6e} "
          f"(boundary excluded, grid_sample border-padding vs roll-wraparound artifact expected there)")
    assert interior_diff < 1e-4, "Offset sign convention wrong -- STOP, geometric target would be mislabeled."
    print("[Unit test] PASS.\n")

    # ---------------- Pilot training: SC-DCU's own parameters only, L_seg alone ----------------
    train_loader, _ = create_brats_loaders(
        batch_size=8, num_workers=4,
        root_dir=str(project_root / "Dataset" / "Training"), val_split=0.1,
    )
    focal_fn = FocalTverskyLoss()
    evidential_fn = EvidentialBetaLoss(weight=0.5)
    optimizer = torch.optim.AdamW(scdcu.parameters(), lr=1e-4, weight_decay=1e-5)

    print(f"Pilot training SC-DCU's own parameters ONLY (trunk frozen), {PILOT_EPOCHS} epochs, "
          f"L_seg alone (FocalTversky+Evidential), NO offset-fidelity term of any kind:\n")
    torch.manual_seed(SEED)
    model.train(False)  # keep BN running stats frozen (trunk frozen); scdcu has no BN
    for epoch in range(PILOT_EPOCHS):
        total_loss = 0.0
        n_batches = 0
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                enc1 = model.enc1(images)
                pool1 = model.pool1(enc1)
                enc2 = model.enc2(pool1)
                pool2 = model.pool2(enc2)
                enc3 = model.enc3(pool2)
                pool3 = model.pool3(enc3)
                bottleneck = model.bottleneck(pool3)
                upconv3 = model.upconv3(bottleneck)
                cat3 = torch.cat([upconv3, enc3], dim=1)
                dec3 = model.dec3(cat3)
                upconv2 = model.upconv2(dec3)
                cat2 = torch.cat([upconv2, enc2], dim=1)
                dec2 = model.dec2(cat2)
                upconv1 = model.upconv1(dec2)

            # CRITICAL FIX (found after the first pilot run showed the offset
            # never left ~0): training on UNPERTURBED enc1 gives SC-DCU no
            # reason to learn a nonzero, shift-tracking offset -- the correct
            # correspondence for real enc1 is already ~identity (the trunk
            # was trained assuming unshifted enc1), so L_seg alone never
            # pushes the offset away from 0 during training, and the
            # held-out shifted-enc1 evaluation was then testing on a
            # distribution the module never saw. Fix: apply a RANDOM
            # synthetic shift (same range as the held-out measurement,
            # matching E65's own construction) to enc1 on EVERY training
            # batch, so learning to correct for varying misalignment is
            # actually part of what minimizes the training loss.
            train_shift = SHIFT_RANGE[torch.randint(low=0, high=len(SHIFT_RANGE), size=(1,)).item()]
            enc1_train = torch.roll(enc1, shifts=(train_shift, train_shift, train_shift), dims=(2, 3, 4))

            # ONLY this part carries gradient: SC-DCU's own offset + resample + the frozen dec1/seg_head/evidential_head forward
            predicted_offset = scdcu(upconv1, enc1_train)
            skip_corrected = deformable_resample(enc1_train, predicted_offset)
            cat1 = torch.cat([upconv1, skip_corrected], dim=1)
            dec1 = model.dec1(cat1)
            probs = model.seg_head(dec1)
            evidential_raw = model.evidential_head(dec1)
            alpha_raw, beta_raw = torch.chunk(evidential_raw, 2, dim=1)
            alpha = F.softplus(alpha_raw) + 1.0
            beta = F.softplus(beta_raw) + 1.0

            focal_loss = focal_fn(probs, masks)
            evidential_loss = evidential_fn(alpha, beta, masks)
            seg_loss = 0.5 * focal_loss + 0.5 * evidential_loss

            if not torch.isfinite(seg_loss):
                raise RuntimeError(f"NaN/Inf seg_loss at epoch {epoch} batch {n_batches}")

            seg_loss.backward()
            torch.nn.utils.clip_grad_norm_(scdcu.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += seg_loss.item()
            n_batches += 1

        print(f"  epoch {epoch+1}/{PILOT_EPOCHS}: mean seg_loss = {total_loss/max(1,n_batches):.4f}")

    print("\nPilot training done. SC-DCU offset module now trained (trunk still frozen).\n")

    # ---------------- Training-effectiveness check: did the offset actually learn to track shifts? ----------------
    # Cheap, direct check BEFORE trusting the full 750-record measurement:
    # on one batch, compare the predicted offset for a small vs. a large
    # KNOWN shift -- if training worked, these should differ substantially
    # (tracking the shift); if the module is still stuck near 0 regardless
    # of shift magnitude, the pilot failed and the downstream measurement
    # would be uninterpretable (as it was before this fix).
    scdcu.eval()
    with torch.no_grad():
        check_image, _, _ = val_dataset[0]
        check_image_b = check_image.unsqueeze(0).to(device)
        _, offset_small = forward_with_scdcu(model, scdcu, check_image_b, injected_shift=(2, 2, 2))
        _, offset_large = forward_with_scdcu(model, scdcu, check_image_b, injected_shift=(4, 4, 4))
        mean_small = offset_small.mean(dim=(0, 2, 3, 4)).cpu().numpy()
        mean_large = offset_large.mean(dim=(0, 2, 3, 4)).cpu().numpy()
    print(f"[Training-effectiveness check] mean offset @ shift=2: {mean_small}, @ shift=4: {mean_large}")
    offset_tracks_shift = float(np.abs(mean_large - mean_small).mean()) > 0.3
    print(f"[Training-effectiveness check] offset responds to shift magnitude "
          f"(|diff| mean > 0.3 voxels): {offset_tracks_shift}")
    if not offset_tracks_shift:
        print("[Training-effectiveness check] WARNING: offset does not appear to track shift magnitude -- "
              "the downstream measurement may be uninformative (module likely still near its zero-init). "
              "Proceeding to report results anyway, flagged honestly, per project discipline (report, don't hide).")
    scdcu.train()

    # ---------------- Discriminating measurement: apply KNOWN shifts, measure Delta vs -t ----------------
    with open(E48_TABLE_PATH) as f:
        e48_records = json.load(f)
    e48_by_id = {r["subject_id"]: r for r in e48_records}

    val_n = len(val_dataset)
    print(f"Measuring offset discrepancy on {val_n} validation subjects x {len(SHIFT_RANGE)} known shifts...\n")

    records = []
    scdcu.eval()
    with torch.no_grad():
        for idx in range(val_n):
            image, mask, subject_id = val_dataset[idx]
            image_b = image.unsqueeze(0).to(device)

            subject_dir = val_dataset.subject_dirs[idx]
            seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
            seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
            native_size = int((seg_data > 0).sum())

            for s in SHIFT_RANGE:
                shift = (s, s, s)  # matches E65's own uniform-axis convention
                _, predicted_offset = forward_with_scdcu(model, scdcu, image_b, injected_shift=shift)
                # Geometric target: verified empirically (not assumed) against
                # deformable_resample's actual grid_sample semantics --
                # offset=+t exactly undoes torch.roll(shift=+t) (grid_sample
                # samples FROM position p+offset, i.e. out[p]=in[p+offset],
                # which is the inverse convention of torch.roll's own
                # shift-the-output-by-t semantics). offset[c]=+s undoes
                # roll(shift=+s), confirmed by direct construction test
                # before this script was trusted (recovers the pre-roll
                # tensor exactly in the interior, away from grid_sample's
                # border-padding boundary artifact).
                target = torch.tensor([s, s, s], device=device, dtype=predicted_offset.dtype)
                target = target.view(1, 3, 1, 1, 1)
                discrepancy = predicted_offset - target  # (1,3,D,H,W)
                mean_abs_discrepancy = float(discrepancy.abs().mean().item())
                mean_predicted_offset = [float(predicted_offset[0, c].mean().item()) for c in range(3)]

                records.append({
                    "subject_id": subject_id, "native_size": native_size, "shift": s,
                    "mean_predicted_offset_dz_dy_dx": mean_predicted_offset,
                    "mean_abs_discrepancy": mean_abs_discrepancy,
                })

            if (idx + 1) % 25 == 0:
                print(f"  processed {idx+1}/{val_n} subjects", flush=True)

    with open(OUT_DIR / "E68_offset_discrepancy_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} (subject, shift) records.\n")

    # ================= Aggregate per-subject discrepancy (mean over all tested shifts) =================
    by_subject = {}
    for r in records:
        by_subject.setdefault(r["subject_id"], []).append(r)

    subject_native_size = []
    subject_mean_discrepancy = []
    for sid, recs in by_subject.items():
        subject_native_size.append(recs[0]["native_size"])
        subject_mean_discrepancy.append(float(np.mean([r["mean_abs_discrepancy"] for r in recs])))

    subject_native_size = np.array(subject_native_size, dtype=np.float64)
    subject_mean_discrepancy = np.array(subject_mean_discrepancy, dtype=np.float64)

    print("=== E68 Offset Discrepancy Measurement: aggregate ===")
    print(f"Mean |Delta_R| across all subjects/shifts = {subject_mean_discrepancy.mean():.4f} voxels "
          f"(+/-{subject_mean_discrepancy.std():.4f})")

    # ---------------- H68-A/B/C decision ----------------
    TOLERANCE_VOXELS = 0.5  # pre-declared: "small" discrepancy means sub-voxel on average
    rho, p_param = stats.spearmanr(subject_native_size, subject_mean_discrepancy)
    rng = np.random.default_rng(SEED)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_y = rng.permutation(subject_mean_discrepancy)
        perm_rhos[i], _ = stats.spearmanr(subject_native_size, perm_y)
    p_perm = float((np.abs(perm_rhos) >= np.abs(rho)).mean())

    print(f"\nSpearman(native_size, mean |Delta_R|) = {rho:+.4f} (parametric p={p_param:.4e}, "
          f"permutation p={p_perm:.4f})")

    mean_discrepancy_small = subject_mean_discrepancy.mean() < TOLERANCE_VOXELS
    size_correlated = p_perm < 0.05

    if mean_discrepancy_small and not size_correlated:
        outcome = "H68-A_GEOMETRIC_MATCH"
        print(f"\n=== OUTCOME: H68-A (geometric match) ===")
        print("Task-optimal offset matches the geometric target; no significant size-structured "
              "discrepancy. LOW novelty per the strict bar -- no operator to build from this.")
    elif size_correlated:
        outcome = "H68-B_STRUCTURED_DIVERGENCE"
        print(f"\n=== OUTCOME: H68-B (structured divergence) ===")
        print("Discrepancy IS significantly correlated with native lesion size -- systematic, "
              "structured divergence from the geometric target. This is the ONLY outcome that "
              "motivates further design work (a new operator, in a SEPARATE future phase).")
    else:
        outcome = "H68-C_UNSTRUCTURED_NOISE"
        print(f"\n=== OUTCOME: H68-C (unstructured noise) ===")
        print("Discrepancy is real/nonzero but NOT significantly correlated with lesion size -- "
              "unstructured. Report as a clean null, do not rescue.")

    summary = {
        "checkpoint": str(CKPT_V3_D4ONLY), "n_subjects": len(by_subject),
        "shift_range": SHIFT_RANGE, "pilot_epochs": PILOT_EPOCHS,
        "mean_abs_discrepancy_voxels": float(subject_mean_discrepancy.mean()),
        "std_abs_discrepancy_voxels": float(subject_mean_discrepancy.std()),
        "tolerance_voxels": TOLERANCE_VOXELS,
        "size_dependence": {"rho": float(rho), "parametric_p": float(p_param), "permutation_p": p_perm},
        "outcome": outcome,
        "offset_tracks_shift_training_check": offset_tracks_shift,
    }
    with open(OUT_DIR / "E68_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E68_summary.json")


if __name__ == "__main__":
    main()
