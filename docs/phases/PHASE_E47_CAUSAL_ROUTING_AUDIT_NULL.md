# Phase E47 — Causal Boundary-Routing Intervention Audit (NULL)

## Context

Following E46's below-target attention-gate result (+0.39pp, see `PHASE_E46_ATTENTION_GATE_RESULT.md`), the project pivoted toward a "diagnosis-driven fix" strategy for Q1 novelty: rather than trying a fourth incremental mechanism, sharpen the causal diagnosis first, since E43's own representation-change audit ended in a correlational null (real representation change, but not shown to be spatially localized to boundary regions after confound correction).

E46's trained attention gate (`attn_gate1`, `psi`) provided, for the first time in this project, a genuine *instrument* for a causal test: an already-learned mechanism that gates the bottleneck→enc1 routing per-voxel, whose effect on individual voxels can be surgically disabled at inference time without retraining.

## Method

No training. Loaded E46's `best.pth` checkpoint (val_dice 0.9102) and ran targeted causal interventions:

- **Region definition**: boundary/interior/background at 64³ (enc1/psi's own resolution), via fractional tumor occupancy (`F.interpolate(mode="area")` from the native mask — verified to produce genuine fractional values, not a nearest-neighbor 0/1 mask; a first implementation bug used nearest-neighbor resize and produced zero boundary voxels for every subject, caught before trusting any result).
- **Intervention**: manually re-implemented the v5 forward pass (verified bit-for-bit identical to the real `model.forward()` when no clamp is applied — max abs diff 0.0) with the ability to clamp `psi=1` (gate disabled, falls back to raw v3-style routing) at an arbitrary voxel subset.
- **Matched-count design**: for each subject, sampled equal numbers of boundary and interior voxels (`n_match = min(n_boundary, n_interior)`) to clamp, avoiding a trivial "more voxels intervened on → bigger effect" confound by construction.
- **Outcome**: per-subject hard-threshold Dice (matching the project's own standard metric), comparing intact vs. boundary-clamped vs. interior-clamped forward passes.
- **Statistics**: paired `drop_difference = boundary_drop - interior_drop` across all 125 validation subjects, subject-clustered (one pair per subject), sign-flip permutation test (1000 trials, exceeding the project's 500-trial minimum), plus an explicit confound check (correlation between the boundary/interior voxel-count ratio and the drop-difference).

## Result

| Metric | Value |
|---|---|
| n subjects | 125 |
| Mean dice_intact | 0.8915 |
| Mean boundary_drop | -0.00423 (±0.01230) |
| Mean interior_drop | -0.00448 (±0.01166) |
| Mean drop_difference | **+0.00025** (±0.01027) |
| Fraction boundary_drop > interior_drop | 53.6% |
| Permutation p-value (1000 trials) | **0.808** |
| Confound corr (count ratio vs. drop_difference) | 0.185 |

## Decision (pre-declared rule)

GO required: `drop_difference > 0` AND `p < 0.05` AND confound correlation `< 0.3`. Only the confound check passed; the effect itself is indistinguishable from zero (p=0.81, essentially a coin flip at 53.6% > 50%).

**Verdict: NULL.** Clamping the learned routing gate at boundary voxels does not hurt Dice more than clamping it at an equal number of interior voxels. The routing mechanism's causal effect, where it exists at all, is diffuse across the volume — not concentrated at the tumor boundary.

## Interpretation

This substantially sharpens (and revises) the picture from E43: it's not merely that E43's specific representation-magnitude metric failed to detect boundary-localization — a genuinely causal, outcome-based (Dice) test built with a real trained instrument (E46's gate) also finds no boundary-specific effect. Combined with E43's own null, this is now two independent lines of evidence (one correlational/representation-based, one causal/outcome-based) both failing to support the "coarse-to-fine boundary routing" hypothesis as the locus of the ~9pp ceiling below perfect Dice.

**This changes the diagnosis-driven strategy materially.** The working hypothesis (routing failure at the boundary) that motivated E46 in the first place is not supported causally. Continuing to build fixes aimed at "boundary routing" would not have a defensible diagnosis→fix link — exactly the failure mode this whole redirect was designed to avoid. The mean drops themselves being near-zero (and occasionally negative, i.e. removing the gate at some voxels marginally *helps*) also suggests psi's learned behavior may be closer to a mild regularizer than a decisive routing mechanism, consistent with E46's own modest, non-decisive Dice gain.

## Next step

Per this project's own kill/stop discipline: report the null plainly (this document), do not construct a new metric or region definition to rescue the boundary-localization hypothesis after seeing this result. The diagnosis-driven strategy is not abandoned, but the specific causal locus needs to be re-examined — likely by testing a different candidate mechanism (e.g., is the bottleneck's information failing to be *encoded* at all for certain lesion types/sizes, rather than failing to be *routed*?) before designing another fix, rather than assuming boundary-routing was the right target.

## Artifacts on disk

- `experiments/exp_e12_eggo_m/e47/run_e47_causal_routing_audit.py` (disk only, per `experiments/` gitignore convention)
- `experiments/exp_e12_eggo_m/e47/E47_causal_intervention_table.json` (125 per-subject records)
- `experiments/exp_e12_eggo_m/e47/E47_summary.json`
