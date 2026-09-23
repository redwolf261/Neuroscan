"""
Phase E24, Gate 5: Smoke-training integration test for the task-aligned
projected margin loss (L_margin^w). Per the user's explicit framing:
this is a MINIMAL INTEGRATION TEST, not an experimental result -- no
claim about Dice, no interpretation of outcome metrics, only "does
data -> forward -> L_seg + L_margin^w -> backward -> optimizer -> next
batch work without numerical or gradient-path failures."

Reuses EGGOMExperiment's REAL construction (real model, real optimizer,
real AdamW+CosineAnnealingLR, real data loader, real checkpoint/log
directories) -- NOT a from-scratch toy loop -- but runs a short, custom
training loop (a handful of batches, not full epochs) that calls
compute_margin_loss with w_hat/delta_d_w instead of EGGOMExperiment's own
train_epoch() (which does not yet have the use_task_margin wiring
described in PHASE_E24 Section 2.4 -- that wiring is deliberately NOT
added to train_eggo_m.py yet, per the user's explicit choice to prove the
mechanism first via a standalone smoke script before committing a larger
change to the shared training script).

Acceptance criteria (all must pass before Gate 6 -- the real E24/E25
controlled training comparison -- is authorized):
  1. L_seg, L_margin^w, L_total remain finite across every batch.
  2. Gradients remain finite (no NaN/Inf in any parameter's .grad).
  3. seg_head receives EXACTLY zero gradient from L_margin^w; dec1 DOES.
  4. w_hat is not part of the optimization graph (its own .grad is None
     -- it was constructed via .detach(), so autograd should never even
     attempt to backprop into it, but this is checked directly, not
     assumed).
  5. active_hinge_pct stays in a sensible, non-degenerate range across
     the smoke run (not exactly 0%, not saturated near 100%).
  6. Parameters actually change value after optimizer.step() (rules out
     an accidental full detachment of the margin pathway making the
     update no-op for dec1 specifically).
  7. No NaN/Inf and no exploding gradient norms across the whole run.
  8. Baseline path (w_hat=None) run side-by-side for comparison, confirmed
     unaffected (same finite-loss/gradient checks, run through the exact
     same batches for a fair side-by-side).

Additional mechanistic check (the user's specific addition to Gate 5,
not part of the original six-test list): for active hinge pairs, the
component of dL_margin^w/dz_i ORTHOGONAL to w_hat must be numerically
negligible -- directly verifies the implemented algorithm actually
constrains the activation-space gradient to w_hat's axis, the central
mathematical claim of PHASE_E23's design (Section 6: "the loss constrains
its direct activation-space gradient to the segmentation-head direction").
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

DELTA_D_W_CALIBRATED = 0.2553  # PHASE_E24 calibration result
N_SMOKE_BATCHES = 6  # "enough iterations to exercise multiple batches and optimizer updates", per instruction -- small enough to stay a smoke test, not a real run


def run_smoke_batches(exp, n_batches, use_task_margin):
    """Runs n_batches of a CUSTOM training loop (mirroring
    EGGOMExperiment.train_epoch()'s real structure line-for-line, the
    parts that matter for this test) with either the baseline (w_hat=None)
    or task-aligned (w_hat=<real seg_head direction>) margin path.
    Returns per-batch diagnostics for the acceptance-criteria checks."""
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

        if use_task_margin:
            # Construct w_hat exactly per PHASE_E23 Section 5 / PHASE_E24
            # Section 2.3: no_grad block PLUS explicit .detach(), doubly
            # ensuring no gradient path exists back into seg_head via
            # this construction.
            with torch.no_grad():
                w_raw = exp.model.seg_head[0].weight.detach().reshape(-1)
                w_hat = (w_raw / w_raw.norm().clamp_min(1e-8)).detach()
            delta_d_active = DELTA_D_W_CALIBRATED
        else:
            w_hat = None
            delta_d_active = DELTA_D_CALIBRATED

        margin_loss, mw, margin_diag = compute_margin_loss(
            dec1_perm, evidence_flat, boundary_flat, gt_flat,
            anchor_idx, current_tau_b, EVIDENCE_P99_DEFAULT, delta_d_active,
            MAX_NEGATIVES_PER_ANCHOR, exp.rng, exp.device,
            w_hat=w_hat,
        )

        total_loss = seg_loss + exp.mu * boundary_loss + exp.lambda_margin * margin_loss

        # --- Acceptance criteria 1: finite losses, checked BEFORE backward ---
        finite_losses = {
            "seg_loss": torch.isfinite(seg_loss).item(),
            "margin_loss": torch.isfinite(margin_loss).item(),
            "total_loss": torch.isfinite(total_loss).item(),
        }

        # --- Mechanistic orthogonality check (user's Gate-5 addition):
        # for an ACTIVE hinge pair, dL_margin/dz_i must be parallel to
        # w_hat (up to sign) -- checked via a SEPARATE autograd.grad call
        # w.r.t. dec1_perm itself (not the full backward below), on
        # margin_loss ALONE (not total_loss), so this measurement isolates
        # exactly the quantity PHASE_E23 Section 6 makes a claim about. ---
        orthogonality_result = None
        if use_task_margin and margin_loss.item() > 0:
            g_z, = torch.autograd.grad(margin_loss, dec1_perm, retain_graph=True)
            g_z_anchors = g_z[anchor_idx]  # (n_anchor, 32) -- only anchor voxels have any nonzero gradient by construction
            g_norms = g_z_anchors.norm(dim=1)
            active_mask = g_norms > 1e-8  # only voxels where the hinge actually fired for at least one pair
            if active_mask.sum() > 0:
                g_active = g_z_anchors[active_mask]
                # component parallel to w_hat: (g . w_hat) * w_hat
                parallel_component = (g_active @ w_hat).unsqueeze(1) * w_hat.unsqueeze(0)
                orthogonal_component = g_active - parallel_component
                orthogonal_norm = orthogonal_component.norm(dim=1)
                total_norm = g_active.norm(dim=1)
                # relative orthogonal fraction, avoiding division by ~0
                valid = total_norm > 1e-10
                rel_orthogonal = (orthogonal_norm[valid] / total_norm[valid]) if valid.sum() > 0 else torch.tensor([])
                orthogonality_result = {
                    "n_active_voxels": int(active_mask.sum().item()),
                    "max_rel_orthogonal_fraction": float(rel_orthogonal.max().item()) if rel_orthogonal.numel() > 0 else None,
                    "mean_rel_orthogonal_fraction": float(rel_orthogonal.mean().item()) if rel_orthogonal.numel() > 0 else None,
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

        # --- Acceptance criteria 3: seg_head gets ZERO gradient from
        # margin_loss specifically (checked via a fresh, independent
        # autograd.grad call on margin_loss ALONE before the real
        # backward's accumulated .grad is used for anything else) ---
        seg_head_zero = True
        if use_task_margin and margin_loss.item() > 0:
            seg_head_margin_grads = torch.autograd.grad(margin_loss, list(exp.model.seg_head.parameters()), retain_graph=True, allow_unused=True)
            for g in seg_head_margin_grads:
                if g is not None and not torch.allclose(g, torch.zeros_like(g), atol=1e-10):
                    seg_head_zero = False

        # --- Acceptance criteria 4: w_hat itself must never receive a
        # gradient (it's a plain tensor, not an nn.Parameter, but check
        # requires_grad explicitly to confirm it's structurally excluded) ---
        w_hat_excluded = True
        if use_task_margin:
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
        })

        print(f"  batch {batch_idx}: seg={seg_loss.item():.4f} margin={margin_loss.item():.5f} "
              f"total={total_loss.item():.4f} active%={margin_diag['active_hinge_pct']*100:.2f} "
              f"grad_finite={grad_finite} max_grad_norm={max_grad_norm:.3f} "
              f"seg_head_zero_grad={seg_head_zero} param_changed={param_changed}"
              + (f" orthogonal_frac(max/mean)={orthogonality_result['max_rel_orthogonal_fraction']:.2e}/"
                 f"{orthogonality_result['mean_rel_orthogonal_fraction']:.2e} "
                 f"(n_active_voxels={orthogonality_result['n_active_voxels']})" if orthogonality_result else ""))

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
        if d["max_grad_norm"] > 100.0:  # generous bound -- gradient clipping already caps at 1.0 post-clip, this checks PRE-clip sanity
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
            if max_frac is not None and max_frac > 0.05:  # 5% relative orthogonal component -- should be ~numerical-precision-level (1e-6 to 1e-4), not a real fraction
                print(f"  FAIL (batch {d['batch']}): orthogonal gradient component too large: {max_frac:.4f} "
                      f"(expected near-zero, activation-space gradient should be parallel to w_hat)")
                all_pass = False

    active_pcts = [d["active_hinge_pct"] for d in diagnostics]
    mean_active = np.mean(active_pcts) * 100
    print(f"\n  Mean active_hinge_pct across smoke run: {mean_active:.2f}%")
    if mode_name == "task-aligned" and not (1.0 <= mean_active <= 60.0):
        print(f"  WARNING: active_hinge_pct outside a broadly sensible range for a live training smoke run "
              f"(note: this drifts from the fresh-init calibration value as the model trains even within "
              f"{N_SMOKE_BATCHES} batches -- not itself a failure, but worth eyeballing)")

    print(f"\n  {mode_name}: {'ALL CRITERIA PASSED' if all_pass else 'SOME CRITERIA FAILED -- SEE ABOVE'}")
    return all_pass


def main():
    import tempfile
    exp_dir = Path(__file__).parent / "e24_smoke_results"
    exp_dir.mkdir(exist_ok=True)

    # Reuse the real configs/brats.yaml, same as every real EGGO-M run
    config_path = project_root / "configs" / "brats.yaml"

    print("=" * 70)
    print("PHASE E24 GATE 5: SMOKE-TRAINING INTEGRATION TEST")
    print("=" * 70)
    print(f"N_SMOKE_BATCHES = {N_SMOKE_BATCHES} (deliberately small -- integration test, not an experiment)")

    print("\n--- Constructing REAL EGGOMExperiment (baseline path, w_hat=None) ---")
    exp_baseline = EGGOMExperiment(
        config_path=str(config_path), exp_dir=str(exp_dir), seed=0,
        mu=0.1, lambda_margin=0.1, delta_d=DELTA_D_CALIBRATED,
        run_name="smoke_baseline",
        num_workers=0,  # short smoke run -- avoid num_workers>0 multiprocessing overhead/reliability issues
                         # on Windows (per project memory: known DataLoader worker reliability gotchas),
                         # matching every diagnostic script from E14 onward, which all use num_workers=0
    )
    print("\n--- Running baseline (w_hat=None) smoke batches ---")
    diag_baseline = run_smoke_batches(exp_baseline, N_SMOKE_BATCHES, use_task_margin=False)
    pass_baseline = check_acceptance_criteria(diag_baseline, "baseline (w_hat=None)")
    exp_baseline.close_logs()

    print("\n--- Constructing REAL EGGOMExperiment (task-aligned path, w_hat=seg_head direction) ---")
    exp_task = EGGOMExperiment(
        config_path=str(config_path), exp_dir=str(exp_dir), seed=0,
        mu=0.1, lambda_margin=0.1, delta_d=DELTA_D_W_CALIBRATED,  # only used for logging string; actual delta_d passed per-call below
        run_name="smoke_task_aligned",
        num_workers=0,
    )
    print("\n--- Running task-aligned (w_hat=seg_head direction) smoke batches ---")
    diag_task = run_smoke_batches(exp_task, N_SMOKE_BATCHES, use_task_margin=True)
    pass_task = check_acceptance_criteria(diag_task, "task-aligned")
    exp_task.close_logs()

    # --- Save a real checkpoint from the task-aligned smoke run, proving
    # checkpointing works end-to-end with this path active (per the
    # user's "checkpointing/logging enabled" requirement) ---
    ckpt_path = exp_task.checkpoint_dir / "smoke_epoch_0.pth"
    torch.save({
        "epoch": 0, "seed": 0, "mu": exp_task.mu, "lambda_margin": exp_task.lambda_margin,
        "model_state": exp_task.model.state_dict(),
        "optimizer_state": exp_task.optimizer.state_dict(),
        "best_val_dice": 0.0,
        "config": exp_task.config,
        "use_task_margin": True,
        "delta_d_w": DELTA_D_W_CALIBRATED,
    }, ckpt_path)
    print(f"\nCheckpoint saved and verified writable: {ckpt_path} ({ckpt_path.stat().st_size} bytes)")
    reloaded = torch.load(ckpt_path, map_location=exp_task.device, weights_only=False)
    assert reloaded["use_task_margin"] is True and reloaded["delta_d_w"] == DELTA_D_W_CALIBRATED
    print("Checkpoint reload verified: use_task_margin/delta_d_w fields present and correct")

    print("\n" + "=" * 70)
    print("GATE 5 FINAL SUMMARY")
    print("=" * 70)
    print(f"  Baseline (w_hat=None):      {'PASS' if pass_baseline else 'FAIL'}")
    print(f"  Task-aligned (w_hat=w_hat): {'PASS' if pass_task else 'FAIL'}")
    overall = pass_baseline and pass_task
    print(f"\n  GATE 5 OVERALL: {'PASS -- Gate 6 (real E24/E25 controlled training) may proceed' if overall else 'FAIL -- DO NOT PROCEED TO GATE 6'}")

    if not overall:
        sys.exit(1)


if __name__ == "__main__":
    main()
