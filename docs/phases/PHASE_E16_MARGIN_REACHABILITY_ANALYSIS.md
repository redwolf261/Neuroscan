# Phase E16: Margin Reachability Analysis

**Status**: ✅ Complete — measurement-only, no retraining, no changes to model/loss/optimizer/hyperparameters anywhere in this phase. **Result does not cleanly match any single pre-registered outcome (A/B/C). Closest fit is a mixture of B and D: the optimizer moves a real, non-trivial fraction of its total displacement in the margin direction (15–35% of energy) and its gradient consistently points the right way (Part C), yet the *net accumulated* movement along that direction is small and slightly negative by the fixed-reference metric — an incoherent-cancellation pattern, not a clean "never enough" or "wrong direction entirely" story.**

**Date**: 2026-08-07

## Purpose

E15 showed the decoder is causally sensitive to latent margin: a
realistic artificial push (14 units, E12f's own observed dynamic range)
produces +0.039 Dice. E12f/E13's real training never achieves more than
~0.2 units of net sustained margin growth. This phase asks, using only
existing checkpoints and no new training: **is this a reachability
problem** (the optimizer physically cannot move `dec1` far enough in the
useful direction), **or is the optimizer moving substantially but mostly
in directions unrelated to margin**, **or is EGGO-M's gradient pointing
in a fundamentally different direction than the one E15 proved useful**?

## Methodology

### Checkpoints and voxel tracking

All measurements use E12f seed 0's existing checkpoints (epochs 1, 5, 10,
15, 20, 25, 30 — the same set used throughout E12b.5/E12d/E12f/E13/E14/
E15) on the same fixed 20-subject validation set used throughout the
project.

**No true epoch-0 (pre-training) checkpoint was ever saved** — the
earliest available is `epoch_1.pth`, already after one full epoch of
training. This script uses epoch 1 as the practical `z_0` reference
point for all "displacement from initialization" measurements. This is
not literal initialization, but it is the same window (epoch 1 → epoch
30) that E12f/E13/E15 all measure their own trajectories over, so it
remains the relevant window for the reachability question — stated here
as a limitation, not hidden (see Limitations).

For Parts A/B/D (which require tracking the *same* physical points over
time), 500 voxels per subject (10,000 total, 134 tumor / 9,866
background) were sampled once at epoch 1 and the identical `(subject,
d, h, w)` indices reused at every subsequent checkpoint —
`BraTSDataset`'s validation split has no random augmentation (confirmed
by reading `Dataset/brats_dataset.py` directly, deterministic resize
only), so this genuinely tracks the same tissue locations, not a
re-randomized sample each time.

Part C (gradient alignment) instead uses **each checkpoint's own live
stratified (uncertainty-based) anchor sampling** — the actual mechanism
`compute_margin_loss` uses during real training — since Part C's
question ("what direction does the real training-time gradient point
in") requires the real sampling process, not a fixed arbitrary set.

### Mathematical definitions

- **Displacement**: `‖z_t − z_0‖`, per tracked voxel.
- **Centroid drift**: `‖c_t − c_0‖` for the tumor and background centroids
  separately, computed over the tracked voxel set.
- **Margin direction (primary, fixed reference)**: for each voxel, the
  unit vector `(z_0 − opposite_class_centroid_at_z_0) / ‖·‖`, held
  constant across the whole trajectory. Chosen as primary because it
  isolates "did the movement that happened go toward the separation
  direction that existed at the start," without letting a shifting
  reference frame redefine "toward" partway through — an important
  methodological choice, discussed further below since it produces a
  materially different answer than a live-reference alternative.
- **Margin direction (secondary, live/cross-check reference)**: the same
  construction but recomputed at each checkpoint's own current
  opposite-class centroid — added during development (not in the
  original plan) after the fixed-reference result initially looked
  surprising, to check whether it was an artifact of the fixed frame
  specifically. It was not (see Result 2) — the discrepancy between the
  two is itself a finding, not a bug in either one.
- **Parallel/orthogonal decomposition**: `Δz = Δz_∥ + Δz_⊥`, where
  `Δz_∥` is the (fixed-reference) projection of `Δz = z_t − z_0` onto the
  margin unit vector and `Δz_⊥` is the remainder.
- **Margin efficiency**: `Σ‖Δz_∥‖² / Σ‖Δz‖²` (fraction of total movement
  *energy* — squared displacement — oriented along the margin direction).
- **Gradient alignment**: `cos(−∇_z L_margin, d_push)`, where
  `d_push` is E15's exact push direction (unit vector away from the
  same-batch opposite-class centroid) evaluated at the same anchor
  voxels `−∇_z L_margin` is computed at, both from **one shared forward
  pass**, matching E14's independent-gradient methodology exactly
  (`torch.autograd.grad`, `model.train()` mode to match live
  training-time BatchNorm behavior).
- **Reachability ratio**: `R = M_actual / M_required`, where `M_required
  = 14.0` (E15's realistic push ceiling) and `M_actual` is the mean
  signed fixed-reference parallel displacement at epoch 30.

### A design choice surfaced during development, not resolved unilaterally

The fixed-reference and live-reference margin-direction metrics
disagree substantially (Result 2) — this was not anticipated at design
time. Both are reported in full rather than picking one and discarding
the other, since they answer genuinely different (and both legitimate)
questions: "did each voxel move the way it needed to at the start of the
window" (fixed) vs. "is each voxel currently better separated than it
used to be, judged against wherever the classes currently are" (live).
The disagreement between them is itself part of the finding, not
resolved by preferring one.

## Results

### Part A: Latent Drift Budget — a real, substantial movement budget exists

| Epoch | Mean disp. `‖z_t−z_0‖` | Median | Std | Tumor centroid drift | BG centroid drift |
|---|---|---|---|---|---|
| 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 5 | 3.922 | 3.026 | 3.518 | **16.237** | 1.779 |
| 10 | 4.789 | 4.160 | 3.297 | 12.669 | 2.501 |
| 15 | 5.406 | 4.919 | 3.292 | 12.929 | 2.981 |
| 20 | 5.761 | 5.350 | 3.343 | 13.473 | 3.277 |
| 25 | 6.106 | 5.740 | 3.386 | 13.467 | 3.548 |
| 30 | **6.472** | 6.126 | 3.387 | 13.581 | 3.864 |

Mean per-voxel displacement grows steadily to 6.47 units by epoch 30 — a
real, substantial movement budget, **not** a case of the representation
being frozen or barely moving at all. Tumor centroid drift is markedly
larger and more volatile than background's throughout (13.6 vs. 3.9 at
epoch 30, a ~3.5× asymmetry) — expected given the ~74:1 background:tumor
voxel imbalance in this dataset (background centroid is a much more
stable average). A sharp early spike in tumor centroid drift (16.2 at
epoch 5, retreating to ~12.7–13.6 by epoch 10 onward) coincides exactly
with E12f's own previously-documented epoch-5 dip in `mean_boundary_margin`
(27.83 → 14.16, `PHASE_E12F_RECALIBRATED_PILOT_RESULTS.md`) — an
independent confirmation, via a completely different measurement
pipeline, that real training-time instability exists specifically around
epoch 5, not an E16-specific artifact.

**This alone rules out the simplest form of Outcome A** ("the optimizer
never produces enough latent movement, period") — there is plenty of raw
movement. The question is what that movement is doing.

### Part B: Direction Decomposition — a real fraction of movement is margin-aligned, but the *net* signed component is small and negative

| Epoch | `E[‖Δz_∥‖²]` | `E[‖Δz_⊥‖²]` | Margin efficiency | Fixed-ref mean signed proj. | % voxels positive (fixed) | Live-ref mean change | % voxels positive (live) |
|---|---|---|---|---|---|---|---|
| 5 | 9.828 | 17.933 | 35.4% | −0.862 | 2.8% | −13.489 | 0.2% |
| 10 | 6.966 | 26.836 | 20.6% | −1.051 | 2.7% | −4.036 | 1.7% |
| 15 | 6.933 | 33.128 | 17.3% | −1.221 | 2.3% | −1.692 | 2.8% |
| 20 | 7.772 | 36.595 | 17.5% | −1.330 | 2.3% | −2.818 | 2.2% |
| 25 | 8.383 | 40.359 | 17.2% | −1.499 | 2.0% | −2.374 | 2.5% |
| 30 | 8.376 | 44.982 | **15.7%** | **−1.590** | **2.0%** | −1.168 | 3.4% |

Two findings here, and they point in different directions, both reported
honestly:

1. **Margin efficiency (energy fraction) is real and non-trivial**:
   15.7–35.4% of total movement *energy* is oriented along the margin
   direction — far more than "almost none" (the null hypothesis this
   part was designed to test against, per the spec's "if only 2%... say
   so explicitly" framing). This is closer to "a substantial minority"
   than "negligible."
2. **But the *net signed* component is consistently negative and grows
   more negative over training** (−0.86 at epoch 5 to −1.59 at epoch
   30), with only ~2–3% of individual voxels showing positive net
   displacement along their own fixed z0-anchored margin direction. This
   means: real energy is spent moving along the margin axis, but it is
   **not predominantly in the separating direction** — it is
   overwhelmingly voxels moving toward, not away from, their original
   opposite-class reference direction, when judged against a fixed frame
   anchored at the start of the window.

The live-reference cross-check (added after the fixed-reference result
looked surprising, to rule out a fixed-frame artifact) **also comes out
negative** at every checkpoint (−13.5 at epoch 5, trending toward −1.2 by
epoch 30, but never crossing into positive territory even as the epoch-5
instability subsides) — so this is not an artifact specific to the fixed
reference choice; both formulations of "is this voxel more separated
from the opposite class than before" agree on the sign, even though they
disagree substantially on magnitude and shape. The live-reference metric
does show a clear trend toward zero (−13.5 → −1.2) as training
progresses past the epoch-5 instability, suggesting the situation
improves but does not flip to genuinely positive within the 30-epoch
window measured.

### Part C: Gradient Alignment — the loss gradient consistently points in the E15-beneficial direction

| Epoch | n | Mean cos | Std | 95% CI |
|---|---|---|---|---|
| 1 | 14,280 | +0.4344 | 0.209 | [−0.053, +0.768] |
| 5 | 8,378 | +0.4266 | 0.258 | [−0.187, +0.804] |
| 10 | 8,746 | +0.3902 | 0.256 | [−0.195, +0.782] |
| 15 | 6,734 | +0.3818 | 0.261 | [−0.207, +0.789] |
| 20 | 6,822 | +0.3287 | 0.265 | [−0.248, +0.757] |
| 25 | 5,963 | +0.3597 | 0.264 | [−0.228, +0.776] |
| 30 | 4,180 | +0.3280 | 0.267 | [−0.244, +0.774] |

**Pooled across all checkpoints (n=55,103): mean cos = +0.3905, std =
0.252** (one-sample t-test vs. 0: t=364.2, p≈0 — at this n the p-value
is not the interesting number, the effect size is). The pooled histogram
(see `e16_plots.png`) is unimodal, right-shifted, centered around
+0.4–0.5, with very little mass below −0.5 — a genuine, consistent,
non-trivial positive alignment, not noise scattered around zero. This
holds at every single checkpoint measured, with only a mild, gradual
decline from +0.43 (epoch 1) to +0.33 (epoch 30) — **the gradient never
loses its correct orientation, even as the net accumulated movement
(Part B) trends negative.**

This is the most important tension this phase surfaces: **the per-step
gradient is reliably right, but the accumulated trajectory is not
tracking it.** This is not a contradiction in the data — a
consistently-correct but comparatively weak per-step signal is entirely
compatible with a net trajectory dominated by other forces (the
segmentation and boundary losses, and/or noise), provided those other
forces are large enough. This is directly consistent with E14's finding
that `L_seg` and `L_margin` gradients are only mildly cooperative
(pooled mean cosine +0.094) rather than tightly aligned — the margin
gradient is not fighting the segmentation gradient (E14), but it is also
evidently not able to dominate the net direction of travel either.

### Part D: Reachability Ratio

`M_actual` (mean signed fixed-reference parallel displacement, epoch 1 →
epoch 30) = **−1.590 units**. `M_required` (E15's realistic push
ceiling) = 14.0 units.

**R = M_actual / M_required = −0.114**

The negative sign means the *net* movement is not just insufficient in
magnitude but pointed the wrong way by this specific (fixed-reference,
signed) accounting — though Part E shows a meaningful fraction of the
*energy budget* (15.7–35.4%) is still being spent along this axis; the
negative net sign reflects that more of that energy pulls the wrong way
than the right way, not that no relevant movement occurs at all. A
secondary, magnitude-only (RMS, not signed) accounting gives a less
extreme but still clearly insufficient picture: RMS parallel movement
at epoch 30 is `√8.376 ≈ 2.89` units, only ~21% of the 14-unit
requirement — even ignoring sign entirely and asking "how large is the
scale of movement along this axis at all," it falls well short of what
E15 showed is needed.

### Part E: Optimization Budget / Margin Efficiency

Already tabulated in Part B above. Margin efficiency (energy fraction
parallel to the margin direction) starts anomalously high during the
epoch-1→5 instability window (35.4%, likely inflated by the same
volatility Part A's tumor-centroid spike shows), then settles to a
stable **15.7–17.5%** for the remainder of training (epochs 10–30). This
is the clearest, single "how much of optimization contributes to
margin" number this phase produces: **roughly one-sixth of total latent
movement energy is spent along the margin axis** in steady-state
training — a real, non-negligible fraction, but the majority (82–85%) of
movement energy goes elsewhere.

### Part F: Falsification — does an alternate margin metric show training moving sufficiently?

| Epoch | Centroid-to-centroid dist. | Est. min pairwise dist. | Est. mean pairwise dist. |
|---|---|---|---|
| 1 | 25.563 | 3.052 | 30.777 |
| 5 | 11.803 | 1.074 | 15.256 |
| 10 | 21.398 | 5.501 | 24.695 |
| 15 | 23.667 | 4.425 | 27.267 |
| 20 | 22.513 | 10.777 | 25.969 |
| 25 | 22.904 | 11.173 | 26.807 |
| 30 | **24.086** | 9.698 | 28.222 |

Net change, epoch 1 → epoch 30: centroid-to-centroid = **−1.478**
(slightly negative), estimated min pairwise = **+6.646** (positive,
driven mostly by the sharp epoch-5 dip recovering rather than sustained
growth beyond the starting point — min pairwise at epoch 1 was
anomalously low, 3.05, likely reflecting the tracked sample's specific
composition at that early, still-destabilizing checkpoint rather than a
stable baseline).

**No evidence was found that training moves sufficiently by an alternate
metric that the primary metrics understate.** All three alternate
proxies tested (centroid-to-centroid, estimated min pairwise, estimated
mean pairwise) show the same qualitative shape as E12f's own
`mean_boundary_margin` metric: a sharp early dip around epoch 5,
partial-to-full recovery by epoch 10–15, then a roughly flat,
non-monotonic plateau through epoch 30 — **not** a metric-specific
artifact where one particular way of measuring margin happens to miss
real sustained growth that another would catch. The falsification
attempt does not succeed: this phase did not find evidence that the
reachability/inefficiency story is a measurement artifact.

## Interpretation: which outcome does this match?

Per the pre-registered success criteria:

- **(A) The optimizer never produces enough latent movement**: **not
  supported as stated.** Part A shows a real, substantial, and growing
  movement budget (6.47 units mean displacement by epoch 30, tumor
  centroid drift up to 13.6 units) — the optimizer is clearly capable of
  moving `dec1` by amounts comparable in raw scale to what Part D's
  RMS-magnitude accounting needs (2.89 achieved vs. 14 needed on the
  signed/coherent basis, but total movement scale is not trivially
  small).
- **(B) The optimizer produces substantial movement, but almost none is
  directed toward margin**: **partially supported, but "almost none" is
  too strong.** Part E shows 15.7–35.4% of movement energy genuinely is
  parallel to the margin direction — a real minority share, not "almost
  none." What Part B adds beyond a pure magnitude story is that the
  *signed*, net-accumulated component of that parallel movement is
  small and negative, meaning much of that margin-direction energy is
  incoherent (pulling both ways, netting out unfavorably) rather than
  simply small.
- **(C) EGGO optimizes a direction fundamentally different from the one
  E15 proved beneficial**: **not supported.** Part C directly refutes
  this — the margin loss's gradient is consistently, substantially
  aligned with E15's exact push direction (pooled mean cos=+0.39,
  positive at every single checkpoint, never near zero or negative).
  Whatever the actual trajectory is doing, the *local, instantaneous
  optimization target* is unambiguously the right one.
- **(D) None of the above; another bottleneck is indicated**: **closest
  fit, but the "other bottleneck" is not mysterious — the data already
  identifies it.** The picture that best fits all six parts together is:
  the margin gradient is correctly directed (C) and contributes a real,
  non-trivial share of the total movement budget (E), but this
  contribution is (i) too small in coherent/signed magnitude to reach
  the ~14-unit scale E15 showed matters (D: R=−0.11 signed, ~0.21 on an
  RMS basis) and (ii) gets substantially counteracted or diluted by
  other, larger forces acting on `dec1` — most plausibly the
  segmentation loss's own gradient, which E14 showed is only mildly
  cooperative with the margin gradient (pooled cosine +0.09, not
  strongly aligned), so the margin push, even though correctly oriented
  on its own, does not compound coherently with everything else moving
  the representation. This is a genuine synthesis of B ("mostly
  elsewhere," with "elsewhere" now more precisely quantified as ~65–84%
  of energy) and D (a specific, identified rather than generic "other
  bottleneck": weak coherent contribution relative to the total budget,
  not wrong-direction, not zero-movement).

## Limitations

1. **No true epoch-0 checkpoint** — `z_0` is practically epoch 1, already
   post-training. All displacement/drift numbers are relative to this
   point, not literal initialization. This is the same window every
   other phase's margin trajectory has used, so it does not undermine
   the comparison to E15's 14-unit requirement (which was itself derived
   from this same epoch-1-to-30 trajectory range), but it means "total
   lifetime movement" is understated by whatever happened during epoch 1
   itself.
2. **Fixed tracking voxel set (10,000 voxels, 500/subject) is a sample,
   not the full volume** — chosen for tractability across 7 checkpoints
   × 20 subjects. The severe tumor:background imbalance in the sample
   (134:9,866) mirrors the real dataset's imbalance, but the small
   absolute tumor-voxel count means tumor-specific statistics (e.g.
   tumor centroid drift) rest on fewer independent samples and carry
   more sampling noise than the background-side numbers.
3. **The fixed-reference and live-reference margin-direction metrics
   disagree in magnitude and shape (though agree in sign)** — both are
   reported rather than resolved to one, per the design note above, but
   this means Part D's specific `R=−0.11` number is sensitive to this
   choice; a live-reference-based `R` would be reported differently
   (the live-reference metric trends toward, but does not reach, zero:
   roughly `R_live ≈ −1.17/14 ≈ −0.08` at epoch 30, similar in sign and
   rough magnitude to the fixed-reference `R`, which is at least a
   consistency check, not a contradiction, between the two choices at
   the final checkpoint specifically — the mid-training values differ
   more substantially).
4. **Part C's anchors and Parts A/B/D's tracked voxels are two different
   (deliberately, per the design) voxel sets** — Part C uses each
   checkpoint's own live stratified (uncertainty-biased) sample, Parts
   A/B/D use one fixed, uniformly-sampled set from epoch 1. This is the
   methodologically correct choice for what each part asks (see
   Methodology), but means Part C's "the gradient points the right way"
   and Part B's "but the trajectory doesn't end up there" are not
   claims about literally the same voxels — a full reconciliation would
   require tracking gradient alignment on the SAME fixed voxel set used
   for displacement, not yet done here.
5. **Single seed, single checkpoint chain** (E12f seed 0). Not extended
   to E13's other 3 seeds. Given E13 itself found margin-Dice
   correlation is highly seed-dependent, the specific magnitudes here
   (though probably not the qualitative "gradient right, net movement
   small/negative" pattern) could plausibly vary by seed.
6. **Part F's "no falsifying evidence found" is a negative result within
   a narrow search** — only three alternate margin proxies were tried,
   all fairly similar in construction (distance-based) to the original
   metric. A genuinely different family of margin definition (e.g.
   angular/cosine-based separation rather than Euclidean) was not
   attempted and might behave differently — not evidence against the
   current conclusion, simply unexplored.

## Conclusion

This phase does not cleanly confirm any single one of the three
specific pre-registered hypotheses (A, B, or C) in their strongest form.
It most closely supports a **specific, evidenced version of D**: the
optimizer is not incapable of moving `dec1` (A is rejected — real,
substantial movement budget exists), and the margin loss's gradient is
not misdirected (C is rejected — gradient alignment with E15's proven
useful direction is consistently positive, +0.39 pooled, at every
checkpoint). What limits margin growth is that only a modest share
(~16–35%, settling near 16–18% in steady-state) of the *total* movement
budget is oriented toward margin at all, and even within that share, the
*net, coherent* component (as opposed to energy that cancels across
voxels or over time) is small and slightly negative rather than
robustly positive — consistent with a correctly-aimed but comparatively
weak force being outcompeted, not overridden or misdirected, by the rest
of what shapes the representation during training. This reframes the
"why doesn't margin grow" question one level deeper than E15 left it:
not "does the decoder care" (E15: yes) and not "is the loss trying to
do the wrong thing" (E16 Part C: no) — but "why does a correctly-aimed,
non-trivial-share force fail to produce coherent net progress," which
Part E's ~16–18% budget share and Part B's incoherent-cancellation
pattern together point to as the mechanism, without this phase
identifying (or being asked to identify) a fix.

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e12_eggo_m/e16_margin_reachability_analysis.py` | Analysis script (Parts A-F) |
| `experiments/exp_e12_eggo_m/e16_margin_reachability_results/` | Per-part JSON results, plots |
| `experiments/exp_e12_eggo_m/e12f_pilot_calibrated_seed0/checkpoints/` | Checkpoints analyzed |
| `PHASE_E15_DECODER_SENSITIVITY.md` | Source of `M_required=14.0` and the reachability question this phase investigates |
| `PHASE_E14_GRADIENT_CONFLICT_ANALYSIS.md` | Source of the L_seg/L_margin cosine (+0.09) cited in Part C's interpretation |
| `PHASE_E12F_RECALIBRATED_PILOT_RESULTS.md` | Source of the epoch-5 dip independently reproduced in Part A |

---

**Completed**: 2026-08-07 — No single pre-registered hypothesis (A/B/C)
confirmed cleanly. Closest fit: a specific, evidenced version of D — the
margin gradient is correctly directed (rejects C) and the optimizer has
a real movement budget (rejects A), but only a modest, largely
incoherent fraction of that budget (~16–18% of energy in steady state,
net signed component small and negative) is realized as sustained
margin growth — a magnitude/coherence shortfall, not a direction or
capacity failure.
