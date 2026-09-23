# E187 — Output-Preserving Cross-Region Feasibility (paper-locked design)

**Date**: 2026-09-18
**Status**: PREREG. Not yet implemented. Continues after
[E186 killed the constructive pool-tie nullspace source](PHASE_E186_DECODER_EQUIVALENCE_DIMENSIONALITY_PREREG.md)
(0/64 candidates passed Step D — the perturbation needed to stay invisible to `pool3` was ~40%
of the activation's own norm, and the unprotected skip path could not absorb it). E187 tests a
different mechanism entirely: not "hide the perturbation from one downstream op," but "cross a
downstream ReLU sign boundary with a MINIMAL perturbation, and see whether the decoder output
survives it anyway."

## Research question

Can an intact enc3 representation move across a downstream ReLU activation boundary (i.e.
change which piecewise-linear region of the network it's in) while keeping the decoder output
within E184's already-established output-equivalence tolerance ε? **This does not yet test Γ**
and makes **no claim of novelty** — it is a feasibility/existence question, exactly as E184's
and E186's first steps were.

## Fixed setup (frozen, reused verbatim — nothing new invented here)

- 10 subjects, same checkpoint (`E131_v5control_seed0`, `best_mean_dice=0.8929357248544694`,
  asserted every run), same 128³ sliding-window/tile convention, same E180 tile ledger, same
  decoder, same `prediction_distance` output metric, same ε source (`measure_epsilon_from_t7_gentlest`,
  E184's T7-gentlest-level calibration, reused unmodified).
- No training. No autograd/gradients anywhere. No GT until a candidate has passed Step D.

## Target layer — locked, and why

**`dec3[0]`** (`Conv3d(256,128,k=3,pad=1) → BatchNorm3d → ReLU(inplace=True)`), specifically its
pre-BatchNorm, pre-ReLU **preactivation** (the raw conv output). Reads `cat3 = [upconv3, enc3]`
directly — the shallowest downstream ReLU whose receptive field over enc3 is a single, small,
well-defined patch (the kernel's own 3×3×3 spatial extent, on enc3's channel-half of `cat3`).
Chosen over `bottleneck[0]` specifically to avoid compounding receptive-field growth through
`pool3`'s own 2×2×2 pooling on top of the conv kernel — keeps the boundary construction local
and exactly derivable, not approximated.

## What $a_j$, $W_c$, and the boundary distance actually are — no dense-layer fiction

For output location $j$ = (channel $c$, spatial position $(z,y,x)$) in `dec3[0]`'s preactivation
map:

$$
a_j(Z) = \langle W_c, \text{cat3\_patch}(j) \rangle + b_c
$$

where $W_c$ = `dec3[0].conv.weight[c]` — the **real** convolution kernel for output channel $c$,
shape `(256, 3, 3, 3)` (256 = `cat3`'s input channels, 128 from `upconv3` + 128 from `enc3`),
restricted to the 3×3×3 spatial patch centered at $j$'s location. This is linear in exactly the
voxels the kernel touches — no fabricated dense vector over all of enc3, and no assumption that
a conv behaves like a fully-connected layer. Only the **enc3 half** of $W_c$ (channels 128:256
of the 256-channel kernel) and the corresponding half of `cat3_patch(j)` are relevant, since
E187 only perturbs enc3, not `upconv3`'s output.

**Normalized boundary distance** (locked): $\text{dist}_j = |a_j(Z)| / \|W_c\|$ — the exact
perpendicular distance from `cat3_patch(j)` to the hyperplane $a_j=0$, computed from the real
kernel norm. This is the ranking criterion for "K nearest eligible boundaries."

**Boundary-crossing delta** (locked): confined entirely to the enc3-half of the 3×3×3 receptive
patch feeding $j$ (never touches `upconv3`'s output, never touches other spatial locations):

$$
\delta_{\text{boundary}} = -\,\text{sign}(a_j(Z)) \cdot \text{dist}_j \cdot \frac{W_c^{\text{enc3-half}}}{\|W_c^{\text{enc3-half}}\|}
$$

takes $a_j$ exactly to 0 within that patch (verified numerically, not assumed). The actual
candidate applies a small crossing margin $\eta$ beyond that:

$$
\delta = (1+\eta)\,\delta_{\text{boundary}}, \qquad \eta = 0.05 \text{ (LOCKED)}
$$

pushing 5% past the zero boundary into the sign-flipped region — small enough to be a genuine
minimal crossing, large enough not to round back to zero under float32.

## Candidate selection — structural, not outcome-filtered

**Eligibility rule (locked, applied BEFORE ranking, never adjusted after seeing Step D results)**:
- Exclude $|a_j(Z)| < 10^{-6}$ (float32 noise floor — a preactivation already this close to
  zero is not distinguishable from numerical noise, excluded rather than used to construct a
  crossing margin on top of noise).
- Exclude boundaries outside the affected tile (only $j$ whose receptive patch lies within the
  intact enc3 tile region under study).
- Deduplicate by **output location** — two candidates are the same boundary if they share the
  identical (output_channel, spatial z,y,x) key in `dec3[0]`'s preactivation map, keep first
  occurrence only.
- Exclude constructions that would require modifying spatial regions outside the single 3×3×3
  receptive patch (structurally impossible given the delta's construction above, verified not
  just assumed).

**Selection (locked)**: rank all eligible boundaries by $\text{dist}_j$ ascending, take the
**K = 8 nearest** (LOCKED — matches E184/E186's own `N_DIRECTIONS`/`N_CANDIDATES_PER_TILE`
convention for cross-probe consistency). No inspection of candidates' downstream effect before
this selection — ranking uses only $\text{dist}_j$, a quantity computed entirely from $Z$ and
the frozen conv weights, never from $D(Z')$.

## Step D — the only feasibility gate, hard, no relaxation

For each of the K candidates per tile (10 subjects × up to 8 candidates = up to 80 total,
fewer if a subject has <8 eligible boundaries after dedup/exclusion):

1. **Boundary crossing verified**: $\text{sign}(a_j(Z+\delta)) \neq \text{sign}(a_j(Z))$ —
   structural check, computed directly from the real conv forward pass (not assumed from the
   construction alone, since BatchNorm/downstream effects could in principle interact —
   verified empirically).
2. **Output equivalence**: $d_{\text{out}} = d(D(Z+\delta), D(Z)) \le \epsilon$ (E184's ε,
   reused verbatim, not re-derived or re-tuned for this probe).

**Both required.** Candidates failing either are excluded, not adjusted. Γ is computed **only**
for candidates that pass both — same discipline as E186, explicitly not relaxed this time either.

## Decision table (locked)

| Outcome | $N_{\text{pass}}$ | Reading |
|---|---|---|
| 1 | 0 | **KILL** cross-region pathway reconfiguration at this intervention point/tolerance — crossing a meaningful downstream boundary necessarily costs more output change than ε allows here. A real negative result, reported as such. |
| 2 | few (structural, not tuned — "few" is descriptive of what's observed, not a pre-set count threshold) | INTERESTING. Proceed to measure $\Gamma(Z)$ and $\Gamma(Z')$ for passing candidates only, using the exact E180-E184 Γ definition (T6/T7-family max-pairwise-distance), no new metric. Primary question: $\Gamma(Z') \neq \Gamma(Z)$? |
| 3 | many | Caution flag — if nearly every boundary crossing is output-equivalent, this may just reflect decoder redundancy rather than a meaningful phenomenon. Same Γ-variation test as outcome 2, but interpreted more cautiously (a "many-survive" result needs the same Γ-variance test, not a stronger claim by default). |

If Outcome 2 or 3 holds AND $\Gamma$ varies: established finding is same-output +
different-computational-regime ⇒ different counterfactual stability — a mechanism one level more
concrete than E184's diagnostic-random-direction result, since the equivalence here is
constructed from an explicit, verifiable boundary-crossing event rather than an arbitrary
direction. GT/Δ correlation is a separate, later stage, not part of this prereg's scope.

## Implementation note: a construction bug caught and fixed BEFORE any result counted

While implementing this script, real subject volumes were found to span multiple overlapping
sliding-window tiles (e.g. 138×176×144, well over 128³ in every axis). The initial
implementation captured `Z_full_enc3`/`cat3` from a single 128³-crop forward pass, then applied
candidate overrides via the multi-window `sliding_window(model, image_b, ...)` — this silently
corrupts every window that is not the one the capture came from (confirmed directly: a
zero-delta identity override still produced a large nonzero `output_distance`). **Fixed before
any real 10-subject run**: all inference in this script uses a single targeted forward pass on
the one 128³ tile under study (`single_tile_forward`), never the multi-window path. This same
bug was found to also affect E186's already-reported result — see
[E186's retraction](PHASE_E186_DECODER_EQUIVALENCE_DIMENSIONALITY_PREREG.md#retraction-2026-09-18--the-kill-verdict-below-is-contaminated-not-trustworthy-as-reported).
E184's own hooks (additive, per-window-self-consistent construction) were checked and are
unaffected.

## Result (2026-09-18) — Outcome 3 (many survive), Γ does not meaningfully vary

Ran on 8/10 subjects (2 skipped: `BraTS-GLI-01161-000`, `BraTS-GLI-01168-000`, tile bounds not
divisible by 8 — same data edge case as E186, unrelated to this design).

**Step D: 35/64 candidates passed** (boundary crossed AND output-equivalent), ranging 2–7 passing
candidates per subject out of 8 eligible. This is **Outcome 3** ("many survive") per the locked
decision table, not Outcome 2 — read with the extra caution the table specifies, not as a
stronger finding by default. Passing candidates' `output_distance_from_Z0` was 0.0 or
float-noise-scale (≤1e-9) in every case — the minimal-crossing construction genuinely preserves
the output almost exactly, as designed. The 29 failures were dominated by `boundary_crossed=False`
(the 5% margin didn't flip the sign in enough cases) rather than by output-equivalence failures —
a distinct, milder failure mode than E186's.

**Γ-variance test** (the decisive question per Outcome 2/3's own follow-up): computed Γ's
range as a fraction of its own per-subject mean, across passing candidates:

| Subject | n passing | Γ mean | Γ range | range/mean |
|---|---|---|---|---|
| 01041 | 4 | 0.1118 | 1.8e-5 | 0.016% |
| 01189 | 5 | 0.0560 | 3.8e-4 | 0.680% |
| 01336 | 5 | 0.0212 | 3.8e-5 | 0.178% |
| 00250 | 3 | 0.0136 | 1.7e-5 | 0.127% |
| 00501 | 6 | 0.0606 | 2.2e-3 | 3.677% |
| 01093 | 7 | 0.0134 | 1.3e-5 | 0.093% |
| 00506 | 2 | 0.0164 | 1.4e-5 | 0.085% |
| 01065 | 3 | 0.0177 | 2.3e-4 | 1.276% |

Mean fraction across subjects: **0.77%**, max 3.68% (subject `00501`, the one outlier). By
contrast, E184's Γ-continuation (Outcome C) found diam(Γ) that was a substantial, structurally
central part of that finding, with weak/inconsistent correlation to output-displacement spread
— a qualitatively different picture from this near-flat result.

**Reading**: Γ does **not** meaningfully vary across these output-preserving boundary-crossing
candidates, at this construction's scale (η=0.05, one boundary per candidate, `dec3[0]`
specifically). This is a **negative result for this specific mechanism** — crossing a single
downstream ReLU boundary minimally, near the sign flip, does not produce the same kind of
Γ-heterogeneity E184's diagnostic random directions did. It does not contradict E184's Outcome C
(different construction, different scale, different layer) and does not retroactively affect
E184's own result. Per the decision table, this does not license moving to a Δ/GT-correlation
stage — the mechanism itself did not clear its own first bar.

**Caveats**: single boundary crossed per candidate (a much smaller perturbation event than E184's
whole-tile random-direction additions) — this may simply be too small a change to move Γ
detectably, not evidence that NO boundary-crossing construction could. `dec3[0]` specifically was
chosen for its small, well-defined receptive field; deeper or multi-boundary constructions were
not tested and remain open. Subject `00501`'s comparatively larger 3.68% is the closest to an
exception and not obviously explained — noted, not chased further without a new pre-registered
probe.

## What this does NOT do

No training. No architecture change. No autodiff/gradients (the boundary construction uses the
real conv weights directly, read from `dec3[0].conv.weight`, not computed via backward-mode
autodiff — consistent with E186's "no autograd" discipline and avoiding E185's tie-breaking
fragility entirely, since this construction never needs MaxPool3d's derivative). No GT/Dice in
construction or selection — GT enters, if at all, only in a later stage testing Δ, explicitly out
of scope here. No claim that crossing decoder boundaries is itself novel (this document does not
make that claim — see E186's prior-art posture, which applies equally here: the claim under test
is specific to NeuroScan's own Γ→Δ phenomenon, not a general method claim).
