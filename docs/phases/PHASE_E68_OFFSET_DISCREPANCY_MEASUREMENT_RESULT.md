# Phase E68 (Section 7.3) — Offset Discrepancy Measurement: Inconclusive

## Status: Experiment run, but result is **not usable evidence for H68-A/B/C**.
Stopped here per explicit user decision, not rescued with further tuning.

## What was run

The SC-DCU base module (Section 2.1 of `PHASE_E68_SC_DCU_DESIGN.md`) was spliced
into the v3/D4-only trunk at the `enc1` skip. The trunk was frozen; only
SC-DCU's own offset-prediction conv (3×3×3, zero-initialized) was trained, for
3 pilot epochs, on `L_seg` alone (FocalTversky + Evidential — the project's
real, active loss composition), with **no offset-fidelity term of any kind**,
per the corrected design's strict novelty bar (Section 8: only a measurement,
not a calibration).

Two implementation bugs were found and fixed during this run, both before any
result was trusted:

1. **Sign convention**: `grid_sample`'s offset semantics (`out[p] = in[p+offset]`)
   are the inverse of `torch.roll`'s (`out[p+shift] = in[p]`) — the geometric
   target to undo `torch.roll(shift=+t)` is `offset=+t`, not `-t` as the
   original design assumed. Caught by an in-script unit test comparing against
   a synthetic ground truth, not assumed.
2. **Train/test distribution mismatch**: the first pilot run trained SC-DCU
   only on real, unperturbed `enc1` (where the correct offset is already ≈0,
   since the trunk was trained assuming unshifted `enc1`), then evaluated on
   synthetically shifted `enc1` — a distribution the module never saw during
   training. This produced a superficially "significant" Spearman correlation
   (H68-B) that was actually an artifact of the fixed geometric target's own
   variation, not learned model behavior (confirmed: predicted offset was
   essentially frozen at its zero-init, std ≈0.003–0.006 voxels across
   subjects). Fixed by applying a random synthetic shift to `enc1` on every
   training batch, matching the held-out evaluation's own shift range.

## Result after both fixes

- Training loss now genuinely decreases (0.123 → 0.103 over 3 epochs) —
  confirms the module is learning *something*.
- **Added effectiveness check failed**: the module's predicted offset is
  essentially constant regardless of the injected shift's magnitude
  (mean offset at shift=2 vs. shift=4: `[0.254, 0.515, 0.492]` vs.
  `[0.254, 0.515, 0.492]` — indistinguishable). It converged to *some* fixed
  correction, not one that tracks shift magnitude at all.
- Given that, the reported outcome (H68-C, unstructured noise, Spearman
  ρ=+0.174, permutation p=0.051 — not significant, but only just) **cannot be
  trusted as a real answer to H68-A/B/C**. A module that isn't tracking shift
  magnitude at all isn't doing the thing the experiment is designed to
  measure; its discrepancy-vs-size relationship (or lack of one) reflects
  something about a degenerate/undertrained offset predictor, not the
  network's real correspondence needs.

## Decision

Per explicit user instruction: **stop here rather than scale up training to
chase a cleaner signal.** Scaling the pilot (more epochs, verifying gradient
magnitude reaching the offset conv, etc.) was considered and explicitly
declined — continuing to adjust the experiment until it produces an
interpretable result would risk exactly the kind of post-hoc
parameter-chasing the project's own constraint #9 (no rescuing killed
hypotheses without a new justifying reason) exists to prevent, especially
given the underlying novelty bar (Section 8: only a structurally new
mechanism counts) means even a clean H68-B finding here would only supply
raw material for a not-yet-designed future phase, not a result in itself.

## Honest bottom line

This experiment did not produce usable evidence for or against H68-A, H68-B,
or H68-C. It is reported as **inconclusive**, not as a null result — there is
a real difference between "we tested this and found no structure" (H68-C,
what was initially reported) and "our test wasn't sensitive enough to test
anything" (what actually happened here, given the failed effectiveness
check). The SC-DCU thread is not being pursued further at this training
scale. Two real implementation bugs (sign convention, train/test mismatch)
were caught and fixed along the way and are documented above for anyone
revisiting this design.

## Artifacts

- `experiments/exp_e12_eggo_m/e68/run_e68_offset_discrepancy_measurement.py`
  (contains the corrected sign convention and shift-augmented training loop)
- `experiments/exp_e12_eggo_m/e68/E68_offset_discrepancy_table.json` (750
  subject×shift records from the run whose training-effectiveness check
  failed — kept for the record, not for citation as a finding)
- `experiments/exp_e12_eggo_m/e68/E68_summary.json` (same caveat)
