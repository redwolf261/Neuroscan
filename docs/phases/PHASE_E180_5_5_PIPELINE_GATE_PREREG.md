# E180 Stage 5.5 — Pipeline gate (engineering + scientific sanity): pre-registration

**Date**: 2026-09-17
**Status**: PRE-REGISTERED. Not run.
**Depends on**: Stages 0-5 (corrected: tile-ledger dedup fix, fixed-probe Γ construction,
same-operating-point Δ construction), all verified on a 10-subject diagnostic pass.

---

## Purpose

Run the full, now-corrected E180 pipeline on **5 fresh subjects** (not previously used in any
calibration or diagnostic pass) as a combined engineering feasibility check and a scientific
sanity check, per the original plan's Stage 5.5 discipline ("treated strictly as an
engineering/feasibility gate, not scientific evidence") — extended, per the review that found
the tile-overlap and probe-idempotency bugs, to also verify the specific fixes hold on unseen
subjects before the 125-subject commitment.

**This is the LOCK point.** After this stage passes, the implementation (Γ construction, probe
eps, tile ledger geometry, Δ construction, the analysis hierarchy below) is frozen. The only
permissible changes after this stage are documented bug fixes, never a search for a stronger
result.

## Engineering gate

Run Stages 1-5 end-to-end on 5 fresh subjects, checking:

- [ ] Identity/no-op sanity gates (Stage 1) still exact on new subjects.
- [ ] Tile ledger has no duplicate `window_id`s and no duplicate enc3 masks (the dedup fix
      verified on new data, not just the 10-subject set it was found and fixed on).
- [ ] Tile-coordinate construction is deterministic (re-running the ledger builder twice on the
      same subject gives identical output).
- [ ] No NaN/Inf in any Γ or Δ record.
- [ ] GPU memory stable across the run (no leak across subjects/families/tiles).
- [ ] Runtime measured and extrapolated to 125 subjects (informs whether Stage 6-8's full run is
      hours or requires further batching).
- [ ] $Z^{deg}$ baseline is **identical** between the Γ pipeline (`run_e180_gamma.py`) and the Δ
      pipeline (`run_e180_delta.py`) for the same subject/family — both must produce the same
      `dice_deg`/degraded prediction, since they are supposed to describe the same operating
      point. This was assumed but never directly cross-checked; do so explicitly here.
- [ ] Restoration in Δ's masked splice affects **only** the intended tile's enc3 region (verify
      by checking the masked-out region's activation is bit-identical to $Z^{deg}$ outside the
      mask).

## Scientific sanity gate

- [ ] Γ non-degenerate (not all-zero, not saturated) on the 5 new subjects, consistent with the
      10-subject calibration.
- [ ] Δ non-degenerate, consistent with the 10-subject pass.
- [ ] Within-subject direction reported (sign of the Γ-Δ relationship per subject), not just the
      pooled statistic.
- [ ] Subject-level `dice_deg` association with Γ and Δ reported (the confound identified in the
      10-subject pass), not hidden.
- [ ] Tile-overlap (`max_neighbor_enc3_iou`) quantified on the new subjects.
- [ ] Separated-tile (reduced-granularity) robustness check reproduced on the new subjects, using
      the **same fixed criterion** locked in Stage 1's prereg amendment (maximum enc3-center L2
      distance pair per subject) — not reselected or tuned here.
- [ ] Γ metric redundancy (pairwise Spearman among the 4 displacement metrics) quantified,
      consistent with the 10-subject finding (ρ≈0.8-0.98, i.e. largely one phenomenon measured
      four ways).
- [ ] No post-hoc transformation or severity selection — T1_rank/T4_spectral/T5_smooth at their
      Stage 2/2b calibrated severities and probe eps, exactly as locked, no new grid search here.

## What this stage does NOT do

- Does not decide KILL/CONTINUE for $H_{180}$ — that is Stage 6 (Capture@B) and Stage 8
  (incremental R² over competitors), run only after this gate passes, on the full 125-subject
  cohort.
- Does not re-tune the probe eps, the tile-separation criterion, or any transform severity.
- Does not run competitors (Stage 7) or the boundary control (Stage 9) — those need the full
  cohort to be meaningful.

## Locked analysis hierarchy for the 125-subject run (restated from Stage 1's amendment, binding)

1. **Primary**: tile-level mixed/fixed-effects analysis accounting for subject (Stage 8's nested
   regression, extended with a subject-effect term). The raw within-model R² of a subject-FE
   regression is **not** reported as a standalone reliability statistic, given the measured
   83-94% adjacent-tile enc3 overlap — it is structurally inflated by non-independent
   observations and is reported only alongside the robustness/sensitivity checks below, never
   alone.
2. **Robustness**: separated-tile subsampling (max enc3-center-distance pair per subject, fixed
   criterion, not reselected).
3. **Sensitivity**: one alternative separation criterion (minimum pairwise enc3-box IoU),
   reported as a single fixed check.
4. **Sign/permutation diagnostic**: exact binomial test on per-subject association sign,
   reported as a robustness diagnostic alongside the primary analysis — never substituted for it,
   never used to rescue a borderline primary result.

## Provisional conclusion (locked wording, carried into the 125-subject report regardless of
outcome direction, adjusted only for the actual result)

> The exploratory association between representation-induced instability and restoration
> benefit survives removal of exact duplicate tiles and persists under substantially reduced
> tile granularity. However, strong tile-level fit statistics are considered unreliable because
> neighboring inference windows share substantial enc3 spatial support. The phenomenon therefore
> remains provisional pending subject-aware analysis on the full validation cohort.

## Output

- `experiments/exp_e12_eggo_m/e180/E180_5_5_pipeline_gate_summary.json`: all checklist items
  with pass/fail and supporting numbers, plus the reproduced robustness/sensitivity/sign-test
  diagnostics on the 5 fresh subjects.

## Result and locked additions (recorded after running, binding on the 125-subject run)

**Engineering gate: full PASS.** Identity/no-op gates exact, tile ledger deduplication verified
on fresh data, deterministic, zero NaN/Inf across 240 Γ+Δ records, stable memory. Runtime
extrapolates to ~7-8 GPU-hours for Stages 3+5 on the full 125-subject cohort.

**Scientific sanity gate: mixed, not catastrophic — treated as a finding, not a failure.** The
pooled tile-level Γ-Δ correlation collapsed to non-significant on these 5 fresh subjects
(ρ=-0.11 to +0.12, p>0.4), traced to one subject (`BraTS-GLI-00731-001`) with baseline ET/TC
Dice of exactly 0.0 (a member of the project's already-documented 15-subject hard tail,
E139/E142) dominating a small pooled sample. Excluding that subject, T1_rank (ρ=+0.72) and
T4_spectral (ρ=+0.95) recover strong signal; T5_smooth stays near zero (ρ=-0.06) — **kept as a
genuine finding, not explained away**. A generic-sensitivity account of Γ would predict all
three families behave similarly; they do not.

**GO — proceed to Stage 6-9 on the full 125-subject cohort.**

### Locked corrections to the analysis design (per review, binding before the 125-subject run)

1. **The "Δ is not well-posed for zero-baseline subjects" claim is a HYPOTHESIS, not fact.**
   Zero baseline Dice tells us the subject-level prediction is poor; it does not prove every
   tile-level restoration counterfactual is undefined — some tiles could still carry latent,
   recoverable information. Tested empirically via the hard-tail diagnosis (below), not asserted.
2. **No subject is silently excluded from the primary cohort.** `00731` and every other subject
   stay in the full-125 primary analysis. Any exclusion is only the E167 110-subject **secondary**
   stratum, defined by E167's pre-existing IDs verbatim — never redefined or reselected based on
   what produces a stronger E180 result.
3. **Two-stratum reporting, identical specification in both:**
   - **A — Full cohort (primary)**: all 125 subjects, the complete Stage 6-9 matrix (Spearman,
     subject-aware β, Capture@B, per-family T1/T4/T5, competitor ladder, incremental
     $R^2_\Gamma$, boundary control).
   - **B — E167 110-subject stratum (secondary robustness)**: the identical matrix, restricted to
     E167's `E167_summary.json`/E160-table-derived 110-subject definition. Answers whether the
     phenomenon holds in the population already considered model-interpretable.
4. **Hard-tail diagnosis (full-cohort only)**: for subjects outside the E167 110, report
   `dice_intact.ET`/`dice_intact.TC` directly against Γ and Δ, stratified into well-segmented /
   poor-ET / poor-TC / zero-ET-TC. This is what actually tests correction #1's hypothesis.
5. **Decisive question, unchanged**: does Γ predict Δ beyond subject severity, magnitude,
   uncertainty, sensitivity, and boundary proximity — in the full cohort AND/OR the 110-subject
   stratum? Yes on either → Stage 10-11 earned. No on both → KILL E180.
