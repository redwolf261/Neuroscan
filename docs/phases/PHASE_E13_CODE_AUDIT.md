# Phase E13 (interleaved): Code Audit — Checking the E12b–E12f Chain for Wrong Coding

**Status**: ✅ Complete — one real, moderately important finding (LR schedule mismatch, shared by baseline and EGGO-M equally); one latent miscalibration flagged but not actively distorting results; everything else verified correct

**Date**: 2026-08-05

## Purpose

Before trusting the Outcome B conclusion (E12f) further, or committing to
a paper angle built on it, audited the full E12a–E12f codebase for bugs
that could have produced a misleading null result — not re-deriving the
science, purely checking implementation correctness. Run in parallel
with E13's multi-seed training (no GPU conflict — this was static code
inspection plus a handful of small, targeted diagnostic scripts on
existing checkpoints).

## What was checked and passed

1. **`UNet3D_v2.forward()`** (`neuroscan_3d_v2.py`): line-by-line
   identical to v1's encoder/decoder path; `dec1.detach()` correctly
   placed before `boundary_head`, matching the corrected design.
   (Bit-exact equivalence to v1 was already verified separately by
   `verify_v2_matches_v1.py` in E12a.)
2. **`compute_margin_loss`'s hinge formula**: `hinge.mean(dim=1)` over
   negatives, then `losses.mean()` over anchors — matches the spec's
   `(1/|B|) Σᵢ weightᵢ · (1/|N(i)|) Σⱼ hinge(i,j)` exactly.
3. **Detach points**: `anchors_evidence.detach()` and
   `anchors_boundary.detach()` both present before use in `U_hat`/`B`
   computation — matches the gradient audit in `PHASE_E11_5`.
4. **Tumor/background anchor balance**: checked across 5 real batches at
   a trained checkpoint — stratified-by-uncertainty sampling (not
   stratified-by-class) still yields a well-balanced 40–60% tumor/background
   split among anchors in every batch checked, so the negative-sampling
   pool (`opp_local`) is never degenerate or near-empty.
5. **Epoch-index alignment** in the baseline comparison script
   (`baseline_val[e - 1]` for 1-indexed epoch `e`) — correct.

## Finding 1: `U_hat`'s apparent "near-constant" behavior was an artifact of testing at initialization, not a real bug

Initial check (on a **freshly initialized**, untrained model) found
`U_hat` compressed to mean=0.970, std=0.007 across sampled anchors —
looked like `U_hat` was contributing almost no discriminative signal,
which would mean the design's `U_hat · B_i` product was effectively just
`B_i` alone. **Re-checked on a trained checkpoint (epoch 30) and this
does not hold**: `U_hat` there has mean=0.719, std=0.188 — real,
meaningful variance. The initial measurement was measuring a degenerate
regime (a fresh model's evidence hasn't differentiated between voxels at
all yet), the same category of mistake as the earlier eval/train
BatchNorm bug from E12e, just caught and corrected within this same
audit pass rather than requiring a separate phase. **Conclusion: not a
bug.** `U_hat` behaves as designed once training has actually started.

## Finding 2 (real, unresolved): `EVIDENCE_P99_DEFAULT=23.25` is measured from a different model/dataset extraction than the one it's applied to

`EVIDENCE_P99_DEFAULT` was derived in the original Phase E1 extraction
(30 volumes, the frozen baseline's evidential head) and hardcoded into
`train_eggo_m.py`. Checked directly against E12f's own trained model at
epoch 30: `evidence_flat.max() = 43.4`, nearly double the assumed p99 of
23.25. **However, checked whether this actually distorts results**: among
sampled (stratified-toward-low-evidence) anchors, `U_hat` never
saturates at exactly 1.0 (0/16,000 anchors checked) — because the
anchor-selection process already filters to low-evidence voxels, which
never approach the high end of the range where the p99 miscalibration
would matter. **Conclusion: a real, latent calibration issue (the
constant should be re-measured on EGGO-M's own evidence distribution,
not borrowed from a different pipeline), but not one that is actively
distorting the E12b–E12f/E13 results**, since the specific anchor-sampling
strategy used happens to avoid the region where it would bite. Should be
fixed before any further phase that changes the anchor-sampling strategy
(e.g. if uniform-random sampling is ever substituted for the current
uncertainty-stratified approach).

## Finding 3 (real, moderately important): the cosine LR schedule's `T_max` does not match the actual training length

`configs/brats.yaml` sets `training.epochs: 50`, and
`CosineAnnealingLR(self.optimizer, T_max=self.config["training"]["epochs"], eta_min=1e-6)`
is built from that config value in `__init__` — **before** the `--epochs`
CLI argument is applied. Every E7/E12b/E12f/E13 run has been launched
with `--epochs 30`, meaning **the actual training loop runs for 30
epochs against a cosine schedule calibrated to decay over 50**.

Quantified the effect directly: at epoch 30 under the true
(mismatched) schedule, the learning rate is still **~0.000139** — versus
what a correctly-configured 30-epoch schedule would have at that point,
**~0.000002** (a ~70× difference). None of these 30-epoch runs actually
reached a properly annealed, low-LR endpoint — every one of them stopped
mid-decay.

**CORRECTION (added while writing up `PHASE_E13_MULTISEED_RESULTS.md`,
2026-08-06)**: the claim below that this is "equally unfair to both" was
checked directly against `exp_e7_causality/seed_0/results.json` and is
**wrong**. That file records `"final_epoch": 50` with 50 full validation
epochs logged — the baseline is a **true 50-epoch run with a correctly
scheduled `T_max=50` cosine anneal**, not a `--epochs 30`-launched run
sharing EGGO-M's mismatch. The "baseline @ epoch 30" values used
throughout E12b/E12f/E13 are this 50-epoch run's trajectory read at its
own epoch-30 index — valid as a matched-epoch Dice comparison, but the
baseline's LR schedule itself was never mismatched the way every EGGO-M
run's was. This is a **real asymmetry favoring the baseline**, not a
shared condition. See `PHASE_E13_MULTISEED_RESULTS.md` for the corrected
analysis and its practical impact (small: the observed EGGO-M deficit at
epoch 30, −0.0037, could partly reflect this asymmetry, and a
properly-scheduled rerun is now the recommended final check). The
original (incorrect) reasoning is preserved below for the record. **It
does mean**:

1. None of the 30-epoch absolute Dice numbers (baseline's 0.9087,
   EGGO-M's 0.90–0.91 range) should be compared against the *original*
   frozen baseline's true 50-epoch, correctly-scheduled result
   (0.9107±0.0005) as if they were the same measurement — they are not
   directly comparable snapshots, only E7's own 30-epoch trajectory is
   the valid, apples-to-apples comparator for EGGO-M's 30-epoch runs
   (which is what was actually used throughout E12b/E12f/E13's analysis
   — this was, by luck rather than design, done correctly already).
2. The mid-training noise seen in several runs (e.g. E12f's epoch-20 dip
   to 0.866) may be partly attributable to training stopping before full
   LR annealing, rather than being purely intervention-driven or random
   seed noise — a still-elevated LR late in training is a plausible
   contributor to instability that a properly-annealed schedule would
   have damped out.
3. **For any future run** (E14, a possible E15/write-up-quality final
   run, or a properly-scheduled re-verification of E13's seeds): either
   pass `--epochs 30` *and* explicitly override `T_max` to match, or run
   the full 50 epochs to let the existing config's schedule complete
   correctly. Flagging this now, mid-E13, rather than silently
   continuing to accumulate results under the mismatched schedule.

## What this means for E13's currently-running seeds

E13 (seeds 1, 2, 3) was launched with the same `--epochs 30` command as
E12f, so it has the identical `T_max=50` mismatch — **consistently**,
not as a new confound relative to seed 0. The multi-seed comparison
E13 is designed to produce (mean±std across seeds, all using the
identical script/schedule) remains internally valid for its stated
purpose (checking whether the Outcome B null replicates across seeds).
**Not stopping or restarting the currently-running E13 seeds** — the
audit finding doesn't invalidate what's in progress, it flags a
schedule detail to fix in any *future* run, and to caveat honestly if
these results are ever compared against the true 50-epoch frozen
baseline number in a writeup.

## Overall assessment

No coding bug was found that would flip the Outcome B conclusion (the
margin mechanism is active but doesn't improve Dice or show the expected
margin-Dice correlation). The two real findings are a latent, currently-inert
calibration constant (Finding 2) and a training-schedule mismatch
(Finding 3) — **as corrected above, this mismatch affects only EGGO-M's
runs, not the baseline**, which ran the full 50 epochs its schedule was
built for. This means EGGO-M's 30-epoch endpoint numbers carry a small,
one-sided disadvantage (badly-annealed LR at cutoff) that the baseline
comparator does not share. `PHASE_E13_MULTISEED_RESULTS.md` (4-seed
confirmation) found this plausible but small in magnitude relative to
the observed effect (EGGO-M trails baseline by −0.0037 mean Dice at
epoch 30, p=0.032, n=4) and recommends a properly-scheduled rerun as the
cleanest way to close out the caveat, without currently believing it
would reverse the Scenario 1 (null/mildly-unfavorable) conclusion.

## Files

| File | Purpose |
|---|---|
| `neuroscan_3d_v2.py`, `experiments/exp_e12_eggo_m/train_eggo_m.py` | Audited source files |
| `PHASE_E12F_RECALIBRATED_PILOT_RESULTS.md` | The result this audit checks |
| `configs/brats.yaml` | Source of the `epochs: 50` / `T_max` mismatch |

---

**Completed**: 2026-08-05 — no bug found that changes the Outcome B conclusion; one schedule-calibration issue flagged for any future run
