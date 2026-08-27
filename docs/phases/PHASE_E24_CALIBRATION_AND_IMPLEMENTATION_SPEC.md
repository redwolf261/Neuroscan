# Phase E24: Calibration + Formal Implementation Specification

**Status**: 🟡 In progress — calibration complete and verified; implementation specification complete; **no training-script code has been modified yet** (per this phase's own scope: spec + calibration + unit tests, not a training run). This document is the artifact E25 (controlled training + ablations) will implement against.

**Date**: 2026-08-09

---

## 1. Calibration: `delta_d_w`

### Method

Direct replication of `e12e_calibrate_constants.py`'s own `delta_d` methodology (same target criterion, same percentile-based derivation, same fresh-init + `model.train()` discipline E12e's own module docstring documents was **critical** — a first E12e attempt used `model.eval()` and got a `delta_d` that produced 0.0% active hinge in real training, traced to a ~68x BatchNorm3d train/eval discrepancy at fresh initialization). `experiments/exp_e12_eggo_m/e24/e24_calibrate_delta_d_w.py` reuses this exact lesson: `model.train()` throughout, never `.eval()`, on a freshly-initialized model — verified in the script's own docstring and re-confirmed by direct comparison against E12e's original run (see Verification below).

**Scope decision, confirmed with user before running this**: calibrate on a fresh, randomly-initialized model (matching E12e's own convention and real training's actual starting regime), **not** after a warmup period. This means `w_hat` (from `seg_head`'s own random init) is itself not yet a meaningful decision direction at calibration time — an acknowledged, unresolved property of this calibration point, not something calibration timing can fix. Calibrating later was considered and rejected: it would repeat exactly the mistake E12d already diagnosed and E12e already fixed for `τ_b` (a value calibrated at one snapshot going stale as training progresses) — `delta_d_w`'s job is only to set the hinge's *initial* active-pair rate correctly, mirroring `delta_d`'s existing role exactly, not to encode any claim about `w_hat`'s own eventual meaningfulness (that is the separate, already-named "`w_hat` stability" risk from `PHASE_E23_ALGORITHMIC_REDESIGN.md` Section 8, to be checked empirically in E25/E26, not resolved by calibration timing).

### Result

```
Euclidean distance distribution at init (cross-check): p20 = 7.3317, mean = 9.6044, std = 2.8808
  -> implied delta_d = 3.6659 (matches E12e's own reported value to 4 decimal places, confirming
     this run's fresh-init geometry reproduces E12e's exactly under the same seed/procedure)

Projected (w_hat) distance distribution at init: p20 = 0.5105, mean = 1.3300, std = 0.8899

delta_d_w = 0.2553   (2*delta_d_w = 0.5105, targeting the same 20th-percentile / 10-30%-active-hinge
                       criterion E12e used for delta_d)

Verification: 20.04% of pairs violate the hinge at delta_d_w=0.2553 (target: 10-30%) -- PASSED

Sanity check: mean(proj_dist)/mean(euclid_dist) = 0.1385 <= 1.0 -- PASSED
  (a projection onto a unit vector can never exceed the full-space distance; this check would have
   caught a sign/normalization bug in the projection math had one existed)
```

Full output: `experiments/exp_e12_eggo_m/e24/e24_results/delta_d_w_calibration.json`, raw log: `experiments/exp_e12_eggo_m/e24/e24_calibration_log.txt`.

**`delta_d_w = 0.2553` is the calibrated constant E25's training run will use.**

---

## 2. Implementation Specification

### 2.1 Scope decision: extend `compute_margin_loss`, don't fork it

Confirmed with user: add an **optional** `w_hat` parameter to the existing `compute_margin_loss` function (`experiments/exp_e12_eggo_m/train_eggo_m.py:169`) rather than writing a separate `compute_margin_loss_w`. When `w_hat=None` (the default), behavior must be **byte-identical** to today's function — baseline runs are provably unaffected by this change, verifiable by unit test (Section 3). When `w_hat` is provided, only the distance computation switches; anchor sampling, `U_hat`/`B` evidence weighting, negative sampling, `active_hinge_pct` tracking, and the diagnostics dict all remain shared, unduplicated, and already-audited (E21.5) code.

### 2.2 Exact diff

Current (`train_eggo_m.py:169-248`), the only line that changes is 230:

```python
dist = torch.norm(zi - zj, dim=2)                # (n_pos, n_neg), single batched op
```

**Proposed** (full function signature and the one changed computation; all other lines unchanged from the current file):

```python
def compute_margin_loss(dec1_flat, evidence_flat, boundary_logit_flat, gt_flat,
                         anchor_idx, tau_b, evidence_p99, delta_d, max_negatives, rng, device,
                         w_hat=None):
    """
    L_margin per PHASE_E10's selected formula (unchanged docstring content
    above this point -- see current file). ...

    w_hat: OPTIONAL, shape (32,), a detached unit vector. When None
    (default), behavior is BYTE-IDENTICAL to the original Euclidean
    formula -- verified by unit test (see PHASE_E24 Section 3). When
    provided, the pairwise distance is computed as the projection
    |w_hat . (z_i - z_j)| instead of ||z_i - z_j||_2, per PHASE_E23's
    task-aligned projected margin loss (L_margin^w). `delta_d` must be
    the CORRESPONDINGLY CALIBRATED constant for whichever distance
    function is active (delta_d=3.6659 for w_hat=None, delta_d_w=0.2553
    for w_hat provided, per PHASE_E24's calibration) -- the caller is
    responsible for passing the right one; this function does not
    validate that the scale is appropriate for the mode selected.
    """
    ...  # unchanged: anchors_z, anchors_evidence, anchors_boundary, anchors_gt,
         # U_hat, B, weight, tumor_mask/bg_mask, tumor_local/bg_local -- all identical

    for pos_local, opp_local in ((tumor_local, bg_local), (bg_local, tumor_local)):
        ...  # unchanged: n_pos, n_neg, rand_idx, neg_local

        zi = anchors_z[pos_local].unsqueeze(1)         # (n_pos, 1, 32)
        zj = anchors_z[neg_local]                       # (n_pos, n_neg, 32)

        if w_hat is None:
            dist = torch.norm(zi - zj, dim=2)            # (n_pos, n_neg) -- UNCHANGED baseline path
        else:
            dist = torch.abs((zi - zj) @ w_hat)          # (n_pos, n_neg) -- NEW projected path

        hinge = torch.clamp(2 * delta_d - dist, min=0.0) ** 2  # (n_pos, n_neg) -- unchanged formula, dist now mode-dependent
        ...  # unchanged: per_anchor_loss, losses[pos_local], active_pair_count/total_pair_count tracking
```

**What did NOT change**: anchor sampling (`sample_stratified_anchors`, untouched), `U_hat`/`B` weighting (untouched, still computed from detached `evidence`/`boundary_logit`), negative sampling (`torch.randint`, untouched — including its own already-known, E21.5-flagged, unrelated cosmetic quirk of not using the passed-in `rng`, not touched or fixed by this change, out of scope here), the hinge formula's outer structure (`relu(2δ−d)²`, unchanged), the outer mean (`losses.mean()`, unchanged), `active_hinge_pct` tracking (unchanged — now correctly reports the active rate under whichever distance function is active, since it reads the same `dist` variable both modes share).

### 2.3 `w_hat` construction and detachment (call-site change)

At the call site (`train_eggo_m.py:387-391`), when the task-aligned mode is active, `w_hat` must be constructed and detached **before** the call, exactly as `PHASE_E23_ALGORITHMIC_REDESIGN.md` Section 5 specifies:

```python
if self.use_task_margin:  # new config flag, see Section 2.4
    with torch.no_grad():
        w_raw = self.model.seg_head[0].weight.detach().reshape(-1)  # (32,), matches E15's own extraction pattern exactly
        w_hat = w_raw / w_raw.norm().clamp_min(1e-8)
else:
    w_hat = None

margin_loss, mw, margin_diag = compute_margin_loss(
    dec1_perm, evidence_flat, boundary_flat, gt_flat,
    anchor_idx, current_tau_b, EVIDENCE_P99_DEFAULT,
    self.delta_d_w if self.use_task_margin else self.delta_d,  # mode-appropriate calibrated constant
    MAX_NEGATIVES_PER_ANCHOR, self.rng, self.device,
    w_hat=w_hat,
)
```

**Gradient audit for this construction** (per E19/E21.5's established discipline of verifying detach points directly, not assuming them): `seg_head[0].weight` is read inside a `torch.no_grad()` block and additionally `.detach()`'d — doubly ensured no gradient path exists from `w_hat`'s construction back into `seg_head`'s parameters via this loss term. This preserves the E19-verified structural fact `∂L_margin/∂θ_seg_head = 0`, now also true of `L_margin^w` (stated in `PHASE_E23` Section 5, re-affirmed here at the implementation level).

### 2.4 New configuration surface

Two new fields, both required, no silent defaults that could make a run ambiguous about which mode it used:

- `self.use_task_margin: bool` — selects `L_margin` (False, baseline) vs. `L_margin^w` (True, proposed). Threaded through the same `EGGOMExperiment.__init__` pattern as `self.mu`/`self.lambda_margin`.
- `self.delta_d_w: float` — the calibrated constant from Section 1 (`0.2553`), passed in analogously to how `self.delta_d` already is. **Not hardcoded inside `compute_margin_loss`** — passed explicitly from the calibration result, keeping the calibration↔training coupling visible in the run's own saved config (same principle `self.delta_d` already follows).

Checkpoint dicts (already saving `mu`/`lambda_margin`/`delta_d`/`config` per E19/E20's confirmed structure) must additionally save `use_task_margin` and `delta_d_w` — required for any later diagnostic script (an E25/E26-era rerun of E14/E19/E20/E21/E22's own machinery) to know, from the checkpoint alone, which loss variant produced it, without relying on the run's directory name or external memory.

---

## 3. Unit Tests (required before any training run, per E22's own smoke-test discipline)

All tests below are to be written as a new, standalone `experiments/exp_e12_eggo_m/e24/test_margin_loss_w.py`, using small synthetic tensors (not real checkpoints/data) so each test isolates exactly one property.

| # | Test | What it proves | Why it's necessary (not just nice-to-have) |
|---|---|---|---|
| 1 | `w_hat=None` reproduces the EXACT original `compute_margin_loss` output (same loss value, same `active_hinge_pct`, to floating-point tolerance) on a fixed synthetic input, compared against a saved reference computed from the *current, unmodified* function before this diff is applied. | The refactor did not silently change baseline behavior. | This is the single most important test — an accidental change to the shared baseline path would invalidate every one of E12–E22's existing results if this diff were applied carelessly. |
| 2 | For a hand-constructed pair `z_i, z_j` and a hand-constructed `w_hat`, verify `d_ij^w` computed by the function exactly equals `abs(w_hat @ (z_i - z_j))` computed independently in the test itself (not reusing any of the function's own code). | The projection formula is implemented correctly, not just "runs without error." | Matches the discipline E21.5 used for verifying E19/E20/E21's own formulas against independent reconstructions. |
| 3 | Sanity bound: for random `z_i, z_j, w_hat` (unit norm), assert `d_ij^w <= ||z_i - z_j||_2` always (a projection onto a unit vector can never exceed the full vector's norm). | Catches a sign, normalization, or axis-order bug that could otherwise silently pass (e.g. an unnormalized `w_hat`, or an accidental dot-product-then-outer-product shape bug). | Same check already used successfully in the calibration script (Section 1) — proven useful there, reused here at the unit level. |
| 4 | Gradient check: construct a tiny synthetic case, call `torch.autograd.grad(margin_loss_w, z_i)`, and independently verify the returned gradient equals the analytical derivative of the hinge w.r.t. `z_i` for the projected metric (`∂d_ij^w/∂z_i = ±w_hat` when the projection is positive/negative, chain-ruled through the hinge exactly as `PHASE_E21_5_AUDIT.md`'s Phase 2 did for the original Euclidean formula). | The new distance function's gradient is mathematically correct, not just "produces some gradient." | Directly reuses E21.5's own finite-difference-verified methodology (max abs error 8.7e-6 at well-conditioned epsilon) — the audit playbook that already caught real issues elsewhere in this project should be applied here too, before trusting this new formula's gradient. |
| 5 | `∂L_margin^w/∂θ_seg_head == 0`: construct a small real (not synthetic) forward pass through `UNet3D_v2`, compute `L_margin^w`, call `torch.autograd.grad(margin_loss_w, seg_head.parameters(), allow_unused=True)`, assert every returned gradient is `None` or exactly zero. | The detach discipline (Section 2.3) is actually enforced by the real code, not just documented in a comment. | Directly reuses E19's own verification pattern for this exact structural claim (`e19_layerwise_gradient_attribution.py`'s own confirmed zero-gradient check for `seg_head` under the baseline `L_margin`). |
| 6 | `delta_d_w` scale sanity: with `w_hat=None` and `delta_d=3.6659` vs. `w_hat=<unit vector>` and `delta_d=0.2553`, both on the SAME fresh-init model/batch, assert `active_hinge_pct` is within the target 10-30% band for both — confirms the two calibrated constants are both live-consistent with their own calibration, not just consistent with the calibration script's own isolated measurement. | The calibration (Section 1) and the implementation (Section 2) actually agree when run through the real training-path code, not just through the standalone calibration script. | Calibration scripts and training code have diverged before in this project (E12e's own first, buggy attempt is the cautionary precedent) — this test directly closes that gap for `delta_d_w`. |

**None of these tests have been run yet — this is the specification for E25's first task, not a report of results.** Writing and running them, plus fixing anything they find, is explicitly the next concrete step before any training run.

---

## 4. What is NOT in scope for E24

Per `PHASE_E23_ALGORITHMIC_REDESIGN.md` Section 11's own progression and this phase's own stated scope:

- No training run of either baseline or proposed EGGO-M (that's E25).
- No rerun of E22's counterfactual protocol on the new method (that's E26 — the decisive falsification test, per `PHASE_E23` Section 7).
- No ablation matrix execution (`PHASE_E23` Section 8 — planned, not run).
- No claim that `L_margin^w` improves, preserves, or degrades Dice — genuinely unknown until E25/E26.
- No modification to `train_eggo_m.py` has actually been applied yet — Section 2 above is the specification for that diff, written and reviewed here, applied as the first concrete action of E25 (immediately followed by Section 3's unit tests, before any real training epoch runs).

---

## 5. Next Steps

1. Apply the Section 2.2/2.3/2.4 diff to `train_eggo_m.py` (or a copied `train_eggo_m_v2.py`, matching this project's own established pattern of never editing a frozen/prior script in place — to be decided at the start of E25, consistent with how `e18_ablation_train.py` was built as an adapted copy rather than an edit to `train_eggo_m.py` itself).
2. Write and run all 6 unit tests from Section 3. **Any failure blocks proceeding — same discipline E22's Section 12 smoke tests enforced.**
3. Only after all 6 tests pass: run E25's controlled training comparison (baseline vs. proposed vs. segmentation-only, per `PHASE_E23` Sections 7–8).

---

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e12_eggo_m/e24/e24_calibrate_delta_d_w.py` | Calibration script (this phase) |
| `experiments/exp_e12_eggo_m/e24/e24_results/delta_d_w_calibration.json` | Calibration result: `delta_d_w = 0.2553` |
| `experiments/exp_e12_eggo_m/e24/e24_calibration_log.txt` | Full calibration run log |
| `experiments/exp_e12_eggo_m/e24/test_margin_loss_w.py` | **Not yet written** — Section 3's unit test suite, first task of E25 |
| `experiments/exp_e12_eggo_m/train_eggo_m.py` | Target of Section 2's diff — **not yet modified** |
| `PHASE_E23_ALGORITHMIC_REDESIGN.md` | The design this spec implements |
| `PHASE_E12E_HYPERPARAMETER_CALIBRATION.md` | The methodology this phase's calibration directly replicates |
| `PHASE_E21_5_AUDIT.md` | The gradient/detach verification discipline Section 3's tests reuse |

---

**Status recap**: Calibration complete (`delta_d_w = 0.2553`, verified via two independent sanity checks). Implementation spec complete (exact diff, detach audit, config surface, 6-test unit test plan). **No training-script code modified. No unit tests written or run yet. No training run performed.** These are the concrete next actions, listed in Section 5, not yet started.
