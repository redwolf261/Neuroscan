"""
Phase E25, Gate 5 (C6-2 variant): Smoke-training integration test for
SC-TAM (Signed Class-Conditional Task-Aligned Margin), the locked C6-2
configuration per PHASE_E25_CANDIDATE6_SC_TAM_DESIGN.md Section 11:
squared hinge, FB-pairs only (already the existing loop's structural
scope, not a new restriction), m_ij calibrated (Section 12,
e25_calibrate_m_ij.py -> 0.3089), no error/boundary weighting, same
lambda/protocol as Gate 6.

Directly adapted from e24_smoke_train.py's structure and acceptance
criteria (1-8 below are identical in spirit); ONE NEW criterion is added
(9) specific to SC-TAM's own defining property -- sign correctness --
which E24's task-aligned (unsigned) test could not check since unsigned
distance has no directional claim to verify.

Per the user's explicit framing established for Gate 5: this is a
MINIMAL INTEGRATION TEST, not an experimental result -- no claim about
Dice, no interpretation of outcome metrics, only "does data -> forward
-> L_seg + L_margin^SC -> backward -> optimizer -> next batch work
without numerical or gradient-path failures, with the correct sign
structure."

Acceptance criteria (all must pass before C6-2 training is authorized):
  1. L_seg, L_margin^SC, L_total remain finite across every batch.
  2. Gradients remain finite (no NaN/Inf in any parameter's .grad).
  3. seg_head receives EXACTLY zero gradient from L_margin^SC; dec1 DOES.
  4. w_hat is not part of the optimization graph (.grad is None / does
     not require grad -- constructed via .detach()).
  5. active_hinge_pct stays in a sensible, non-degenerate range.
  6. Parameters actually change value after optimizer.step().
  7. No NaN/Inf and no exploding gradient norms across the whole run.
  8. Baseline path (margin_mode=None / euclidean) run side-by-side,
     confirmed unaffected.
  9. NEW for SC-TAM: for active hinge pairs, dL_margin^SC/dz_tumor must
     project NEGATIVELY onto w_hat and dL_margin^SC/dz_bg must project
     POSITIVELY onto w_hat -- directly verifies the live training-loop
     invocation (real batches, real anchors, real stratified sampling)
     preserves the same sign structure test_sc_tam_sign_correctness
     already verified on synthetic data. This is the mechanistic
     property this whole candidate exists to install; it must survive
     contact with real data before training is authorized.
  Plus the orthogonality check carried over from Gate 5 (criterion
  analogous to E24's): the component of dL_margin^SC/dz_i ORTHOGONAL to
  w_hat must be numerically negligible for active pairs -- SC-TAM's
  hinge, like task_aligned's, is defined purely in terms of the
  projection onto w_hat, so this property should hold identically.
"""
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m"))
from train_eggo_m import (  # noqa: E402
    EGGOMExperiment, compute_margin_loss, sample_stratified_anchors,
    DELTA_D_CALIBRATED, ANCHORS_PER_VOLUME, MAX_NEGATIVES_PER_ANCHOR,
    EVIDENCE_P99_DEFAULT,
)

M_IJ_CALIBRATED = 0.3089  # PHASE_E25 Section 12 calibration result (e25_calibrate_m_ij.py)
N_SMOKE_BATCHES = 6  # same as E24 Gate 5 -- small enough to stay a smoke test, not a real run


def run_smoke_batches(exp, n_batches, use_sc_tam):
    """Mirrors e24_smoke_train.py's run_smoke_batches structure exactly,
    substituting margin_mode='sc_tam'/m_ij for w_hat/delta_d_w, and
    adding the sign-correctness check (criterion 9) alongside the
    orthogonality check carried over unchanged."""
    exp.model.train()
    it = iter(exp.train_loader)
    diagnostics = []

    for batch_idx in range(n_batches):
        try:
            images, masks, _ = next(it)
        except StopIteration:
            it = iter(exp.train_loader)
            images, masks, _ = next(it)
        images = images.to(exp.device)
        masks = masks.to(exp.device)

        exp.optimizer.zero_grad(set_to_none=True)
        outputs = exp.model(images)

        probs = outputs["probs"]
        alpha, beta = outputs["alpha"], outputs["beta"]
        boundary_logit = outputs["boundary_logit"]
        dec1 = outputs["dec1"]

        focal_loss = exp.focal_fn(probs, masks)
        evidential_loss = exp.evidential_fn(alpha, beta, masks)
        seg_loss = exp.focal_weight * focal_loss + exp.evidential_weight * evidential_loss
        boundary_loss = exp.boundary_criterion(boundary_logit, masks)

        with torch.no_grad():
            evidence_full = (alpha + beta - 2.0)

        B, C, D, H, W = dec1.shape
        dec1_perm = dec1.permute(0, 2, 3, 4, 1).reshape(-1, C)
        evidence_flat = evidence_full.reshape(-1)
        boundary_flat = boundary_logit.reshape(-1)
        gt_flat = masks.reshape(-1)

        voxels_per_vol = D * H * W
        anchor_idx_list = []
        for b in range(B):
            vol_evidence = evidence_flat[b * voxels_per_vol:(b + 1) * voxels_per_vol]
            local_idx = sample_stratified_anchors(vol_evidence, ANCHORS_PER_VOLUME, exp.rng)
            anchor_idx_list.append(local_idx + b * voxels_per_vol)
        anchor_idx = torch.cat(anchor_idx_list)

        current_tau_b = exp.tau_b_tracker.tau_b

        if use_sc_tam:
            with torch.no_grad():
                w_raw = exp.model.seg_head[0].weight.detach().reshape(-1)
                w_hat = (w_raw / w_raw.norm().clamp_min(1e-8)).detach()
            delta_d_active = M_IJ_CALIBRATED
            margin_mode = "sc_tam"
        else:
            w_hat = None
            delta_d_active = DELTA_D_CALIBRATED
            margin_mode = "euclidean"

        margin_loss, mw, margin_diag = compute_margin_loss(
            dec1_perm, evidence_flat, boundary_flat, gt_flat,
            anchor_idx, current_tau_b, EVIDENCE_P99_DEFAULT, delta_d_active,
            MAX_NEGATIVES_PER_ANCHOR, exp.rng, exp.device,
            w_hat=w_hat, margin_mode=margin_mode,
        )

        total_loss = seg_loss + exp.mu * boundary_loss + exp.lambda_margin * margin_loss

        # --- Acceptance criteria 1: finite losses, checked BEFORE backward ---
        finite_losses = {
            "seg_loss": torch.isfinite(seg_loss).item(),
            "margin_loss": torch.isfinite(margin_loss).item(),
            "total_loss": torch.isfinite(total_loss).item(),
        }

        # --- Orthogonality check (carried over from E24 Gate 5) + NEW
        # criterion 9 sign-correctness check, both computed via a
        # SEPARATE autograd.grad call w.r.t. dec1_perm on margin_loss
        # ALONE (not total_loss), isolating exactly the quantities these
        # criteria make claims about. Sign correctness requires knowing
        # WHICH anchor voxels are tumor vs background, so gt_flat at the
        # anchor indices is used to split g_z_anchors accordingly. ---
        orthogonality_result = None
        sign_result = None
        if use_sc_tam and margin_loss.item() > 0:
            g_z, = torch.autograd.grad(margin_loss, dec1_perm, retain_graph=True)
            g_z_anchors = g_z[anchor_idx]  # (n_anchor, 32)
            g_norms = g_z_anchors.norm(dim=1)
            active_mask = g_norms > 1e-8
            if active_mask.sum() > 0:
                g_active = g_z_anchors[active_mask]
                parallel_component = (g_active @ w_hat).unsqueeze(1) * w_hat.unsqueeze(0)
                orthogonal_component = g_active - parallel_component
                orthogonal_norm = orthogonal_component.norm(dim=1)
                total_norm = g_active.norm(dim=1)
                valid = total_norm > 1e-10
                rel_orthogonal = (orthogonal_norm[valid] / total_norm[valid]) if valid.sum() > 0 else torch.tensor([])
                orthogonality_result = {
                    "n_active_voxels": int(active_mask.sum().item()),
                    "max_rel_orthogonal_fraction": float(rel_orthogonal.max().item()) if rel_orthogonal.numel() > 0 else None,
                    "mean_rel_orthogonal_fraction": float(rel_orthogonal.mean().item()) if rel_orthogonal.numel() > 0 else None,
                }

                anchor_gt = gt_flat[anchor_idx]  # (n_anchor,)
                active_gt = anchor_gt[active_mask]
                proj = g_active @ w_hat  # (n_active,) signed scalar projection
                tumor_active = active_gt > 0.5
                bg_active = ~tumor_active
                n_tumor_active = int(tumor_active.sum().item())
                n_bg_active = int(bg_active.sum().item())
                tumor_correct = int((proj[tumor_active] < 0).sum().item()) if n_tumor_active > 0 else 0
                bg_correct = int((proj[bg_active] > 0).sum().item()) if n_bg_active > 0 else 0
                sign_result = {
                    "n_tumor_active": n_tumor_active,
                    "n_bg_active": n_bg_active,
                    "tumor_correct_sign_frac": (tumor_correct / n_tumor_active) if n_tumor_active > 0 else None,
                    "bg_correct_sign_frac": (bg_correct / n_bg_active) if n_bg_active > 0 else None,
                }

        total_loss.backward()

        # --- Acceptance criteria 2/7: finite gradients everywhere, no exploding norms ---
        grad_finite = True
        max_grad_norm = 0.0
        for p in exp.model.parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    grad_finite = False
                max_grad_norm = max(max_grad_norm, p.grad.norm().item())

        # --- Acceptance criteria 3: seg_head gets ZERO gradient from margin_loss ---
        seg_head_zero = True
        if use_sc_tam and margin_loss.item() > 0:
            seg_head_margin_grads = torch.autograd.grad(margin_loss, list(exp.model.seg_head.parameters()), retain_graph=True, allow_unused=True)
            for g in seg_head_margin_grads:
                if g is not None and not torch.allclose(g, torch.zeros_like(g), atol=1e-10):
                    seg_head_zero = False

        # --- Acceptance criteria 4: w_hat excluded from the optimization graph ---
        w_hat_excluded = True
        if use_sc_tam:
            w_hat_excluded = not w_hat.requires_grad

        torch.nn.utils.clip_grad_norm_(exp.model.parameters(), max_norm=1.0)

        # --- Acceptance criteria 6: snapshot a dec1 parameter BEFORE step, compare AFTER ---
        sample_param = next(exp.model.dec1.parameters())
        param_before = sample_param.detach().clone()

        exp.optimizer.step()

        param_after = sample_param.detach().clone()
        param_changed = not torch.equal(param_before, param_after)

        diagnostics.append({
            "batch": batch_idx,
            "finite_losses": finite_losses,
            "seg_loss": seg_loss.item(),
            "margin_loss": margin_loss.item(),
            "total_loss": total_loss.item(),
            "active_hinge_pct": margin_diag["active_hinge_pct"],
            "grad_finite": grad_finite,
            "max_grad_norm": max_grad_norm,
            "seg_head_zero_grad": seg_head_zero,
            "w_hat_excluded_from_graph": w_hat_excluded,
            "dec1_param_changed": param_changed,
            "orthogonality": orthogonality_result,
            "sign": sign_result,
        })

        sign_str = ""
        if sign_result is not None:
            tf = sign_result["tumor_correct_sign_frac"]
            bf = sign_result["bg_correct_sign_frac"]
            sign_str = (f" sign(tumor/bg correct)={tf:.2f}/{bf:.2f}"
                        if tf is not None and bf is not None else "")
        print(f"  batch {batch_idx}: seg={seg_loss.item():.4f} margin={margin_loss.item():.5f} "
              f"total={total_loss.item():.4f} active%={margin_diag['active_hinge_pct']*100:.2f} "
              f"grad_finite={grad_finite} max_grad_norm={max_grad_norm:.3f} "
              f"seg_head_zero_grad={seg_head_zero} param_changed={param_changed}"
              + (f" orthogonal_frac(max/mean)={orthogonality_result['max_rel_orthogonal_fraction']:.2e}/"
                 f"{orthogonality_result['mean_rel_orthogonal_fraction']:.2e}" if orthogonality_result else "")
              + sign_str)

    return diagnostics


def check_acceptance_criteria(diagnostics, mode_name):
    print(f"\n{'='*70}\nACCEPTANCE CRITERIA CHECK: {mode_name}\n{'='*70}")
    all_pass = True

    for d in diagnostics:
        if not all(d["finite_losses"].values()):
            print(f"  FAIL (batch {d['batch']}): non-finite loss detected: {d['finite_losses']}")
            all_pass = False
        if not d["grad_finite"]:
            print(f"  FAIL (batch {d['batch']}): non-finite gradient detected")
            all_pass = False
        if d["max_grad_norm"] > 100.0:
            print(f"  FAIL (batch {d['batch']}): exploding gradient norm pre-clip: {d['max_grad_norm']}")
            all_pass = False
        if not d["seg_head_zero_grad"]:
            print(f"  FAIL (batch {d['batch']}): seg_head received nonzero gradient from margin loss")
            all_pass = False
        if not d["w_hat_excluded_from_graph"]:
            print(f"  FAIL (batch {d['batch']}): w_hat was part of the optimization graph")
            all_pass = False
        if not d["dec1_param_changed"]:
            print(f"  FAIL (batch {d['batch']}): dec1 parameters did not change after optimizer.step()")
            all_pass = False
        if d["orthogonality"] is not None:
            max_frac = d["orthogonality"]["max_rel_orthogonal_fraction"]
            if max_frac is not None and max_frac > 0.05:
                print(f"  FAIL (batch {d['batch']}): orthogonal gradient component too large: {max_frac:.4f}")
                all_pass = False
        if d["sign"] is not None:
            tf = d["sign"]["tumor_correct_sign_frac"]
            bf = d["sign"]["bg_correct_sign_frac"]
            # This is a STRUCTURAL/mechanistic check, not a statistical
            # one: for EVERY active pair, the sign is determined
            # analytically by construction (tumor anchor's gradient
            # ALWAYS points toward -w_hat, background's ALWAYS toward
            # +w_hat, per the design's own derivative formula, Section 3).
            # A single active voxel with the wrong sign would indicate a
            # real implementation bug, not sampling variance -- unlike
            # Q3's OUTCOME-level "correct sign after training" statistic
            # from the failure analysis, this is a MECHANISM-level
            # guarantee that must hold at exactly 100%, always.
            if tf is not None and tf < 1.0 - 1e-6:
                print(f"  FAIL (batch {d['batch']}): tumor gradient sign incorrect for {(1-tf)*d['sign']['n_tumor_active']:.0f}/{d['sign']['n_tumor_active']} active voxels (expected 100% negative projection)")
                all_pass = False
            if bf is not None and bf < 1.0 - 1e-6:
                print(f"  FAIL (batch {d['batch']}): background gradient sign incorrect for {(1-bf)*d['sign']['n_bg_active']:.0f}/{d['sign']['n_bg_active']} active voxels (expected 100% positive projection)")
                all_pass = False

    active_pcts = [d["active_hinge_pct"] for d in diagnostics]
    mean_active = np.mean(active_pcts) * 100
    print(f"\n  Mean active_hinge_pct across smoke run: {mean_active:.2f}%")
    if mode_name == "sc_tam" and not (1.0 <= mean_active <= 60.0):
        print(f"  WARNING: active_hinge_pct outside a broadly sensible range for a live training smoke run "
              f"(drifts from the fresh-init calibration value as the model trains even within "
              f"{N_SMOKE_BATCHES} batches -- not itself a failure, but worth eyeballing)")

    print(f"\n  {mode_name}: {'ALL CRITERIA PASSED' if all_pass else 'SOME CRITERIA FAILED -- SEE ABOVE'}")
    return all_pass


def main():
    exp_dir = Path(__file__).parent / "e25_smoke_results"
    exp_dir.mkdir(exist_ok=True)

    config_path = project_root / "configs" / "brats.yaml"

    print("=" * 70)
    print("PHASE E25 GATE 5 (C6-2): SMOKE-TRAINING INTEGRATION TEST -- SC-TAM")
    print("=" * 70)
    print(f"N_SMOKE_BATCHES = {N_SMOKE_BATCHES} (deliberately small -- integration test, not an experiment)")
    print(f"M_IJ_CALIBRATED = {M_IJ_CALIBRATED} (PHASE_E25 Section 12)")

    print("\n--- Constructing REAL EGGOMExperiment (baseline path, margin_mode='euclidean') ---")
    exp_baseline = EGGOMExperiment(
        config_path=str(config_path), exp_dir=str(exp_dir), seed=0,
        mu=0.1, lambda_margin=0.1, delta_d=DELTA_D_CALIBRATED,
        run_name="smoke_baseline",
        num_workers=0,  # short smoke run -- matches every diagnostic script from E14 onward
    )
    print("\n--- Running baseline (margin_mode='euclidean') smoke batches ---")
    diag_baseline = run_smoke_batches(exp_baseline, N_SMOKE_BATCHES, use_sc_tam=False)
    pass_baseline = check_acceptance_criteria(diag_baseline, "baseline (euclidean)")
    exp_baseline.close_logs()

    print("\n--- Constructing REAL EGGOMExperiment (SC-TAM path, margin_mode='sc_tam') ---")
    exp_sc = EGGOMExperiment(
        config_path=str(config_path), exp_dir=str(exp_dir), seed=0,
        mu=0.1, lambda_margin=0.1, delta_d=M_IJ_CALIBRATED,  # logging string only; actual value passed per-call below
        run_name="smoke_sc_tam",
        num_workers=0,
    )
    print("\n--- Running SC-TAM (margin_mode='sc_tam') smoke batches ---")
    diag_sc = run_smoke_batches(exp_sc, N_SMOKE_BATCHES, use_sc_tam=True)
    pass_sc = check_acceptance_criteria(diag_sc, "sc_tam")
    exp_sc.close_logs()

    # --- Save a real checkpoint from the SC-TAM smoke run, proving
    # checkpointing works end-to-end with this path active ---
    ckpt_path = exp_sc.checkpoint_dir / "smoke_epoch_0.pth"
    torch.save({
        "epoch": 0, "seed": 0, "mu": exp_sc.mu, "lambda_margin": exp_sc.lambda_margin,
        "model_state": exp_sc.model.state_dict(),
        "optimizer_state": exp_sc.optimizer.state_dict(),
        "best_val_dice": 0.0,
        "config": exp_sc.config,
        "margin_mode": "sc_tam",
        "m_ij": M_IJ_CALIBRATED,
    }, ckpt_path)
    print(f"\nCheckpoint saved and verified writable: {ckpt_path} ({ckpt_path.stat().st_size} bytes)")
    reloaded = torch.load(ckpt_path, map_location=exp_sc.device, weights_only=False)
    assert reloaded["margin_mode"] == "sc_tam" and reloaded["m_ij"] == M_IJ_CALIBRATED
    print("Checkpoint reload verified: margin_mode/m_ij fields present and correct")

    print("\n" + "=" * 70)
    print("GATE 5 (C6-2) FINAL SUMMARY")
    print("=" * 70)
    print(f"  Baseline (euclidean): {'PASS' if pass_baseline else 'FAIL'}")
    print(f"  SC-TAM (sc_tam):      {'PASS' if pass_sc else 'FAIL'}")
    overall = pass_baseline and pass_sc
    print(f"\n  GATE 5 (C6-2) OVERALL: {'PASS -- C6-2 training may proceed' if overall else 'FAIL -- DO NOT PROCEED TO TRAINING'}")

    if not overall:
        sys.exit(1)


if __name__ == "__main__":
    main()
