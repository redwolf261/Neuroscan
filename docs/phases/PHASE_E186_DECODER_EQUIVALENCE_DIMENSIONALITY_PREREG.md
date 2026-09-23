# E186 — Decoder-equivalence dimensionality probe (paper-locked design)

**Date**: 2026-09-18
**Status**: PREREG. Not yet implemented. Continues the branch after
[E185 killed the gradient/Jacobian route](PHASE_E185_NULLSPACE_GRADIENT_PROBE_PREREG.md) (not
because $P_\perp\nabla_Z\Gamma=0$ was shown, but because autodiff(J) is unreliable at the real,
tie-heavy enc3 activation). User's decision: kill gradient-based repair, adopt derivative-free
search over a small candidate orbit $\mathcal{C}(Z)$ — and before designing that orbit's
optimizer, determine whether the decoder's local equivalence structure has enough dimensional
freedom to be worth searching at all.

## Prior-art posture (stated up front, not deferred)

Null-space perturbation of neural network weights/activations is NOT novel machinery on its own
— published work (2026 *Scientific Reports* "exact feature collisions", ICLR 2026 work on
causal-intervention null spaces, continual-learning task-relevant-subspace decompositions)
already occupies "what can change while the output/task stays fixed." **E186 does not claim the
null-space construction itself as novel.** What remains open, and is what E186 tests, is
NeuroScan's own question: does the output-invisible space contain DIFFERENT levels of Γ
(representation-induced counterfactual instability), and does navigating it change recoverable
Dice? That is a claim about this project's own Γ→Δ phenomenon (E180-E184), not about null-space
search as a general technique. Any eventual paper framing must describe the decoder-equivalence
orbit as a **candidate mechanism for exploiting an already-established phenomenon**, not as a
novel null-space method.

## The question, precisely

$$
D(Z) = AZ + b \quad \text{within one fixed piecewise-linear activation regime (frozen ReLU
masks + frozen MaxPool argmax selections)}
$$

$$
D(Z+\delta) = D(Z) \iff A\delta = 0 \iff \delta \in \ker(A)
$$

Does $\ker(A)$ (restricted to a computationally cheap, constructive subset — see below) contain
directions $\delta$ for which $\Gamma(Z+\delta) \neq \Gamma(Z)$? I.e., does moving within the
decoder's own output-invisible space change counterfactual instability?

## What $A$ actually is — corrected from the first draft of this idea

enc3 is $128\times32^3$ (E180-E184's convention; NOT $128\times16^3$ as an earlier draft of
this idea assumed — checked directly against `E180_tile_ledger_full.json`:
`enc3_shape: [1,128,32,32,32]`). enc3 feeds the network through **two parallel paths** that
reconverge, not one:

1. **Bottleneck path**: `enc3 → pool3 → bottleneck → upconv3 → ... → dec1 → seg_head`
2. **Skip path**: `enc3 → cat3=[upconv3, enc3] → dec3` (raw, unpooled enc3 concatenated directly)

$A$ is the Jacobian of the *whole* remaining forward pass (both paths, correctly recombined) —
not a single conv layer's weight matrix, and not reducible to one path alone.

## Dimensionality, corrected

At every patch size tested (128, 32, 16), enc3's dimensionality vs. the output's:

| Patch | enc3_dim (128×(patch/4)³) | out_dim (3×patch³) | ratio |
|---|---|---|---|
| 128 | 4,194,304 | 6,291,456 | 0.667 |
| 32 | 65,536 | 98,304 | 0.667 |
| 16 | 8,192 | 12,288 | 0.667 |

**enc3 has FEWER dimensions than the output at every scale** (ratio is scale-invariant — both
scale as $patch^3$). A **generic** linear map from a lower- to a higher-dimensional space is
full-column-rank almost everywhere, i.e. $\ker(A) = \{0\}$ generically. This is a real risk the
original proposal's dimension count (which used the wrong shapes) understated. **This is why
E186 does not attempt to compute $\ker(A)$ from a generic-rank argument or by forming the full
$A$ and checking its rank** (intractable at this size regardless — up to 4M×6M even at full
patch, and inherits E185's exact-tie ambiguity for how to freeze tied MaxPool windows as
selection matrices).

## Adopted construction: constructive nullspace from tied/non-argmax MaxPool entries

Instead of forming $A$ and computing $\ker(A)$ generically, **construct a nullspace source
directly from known structure**, sidestepping the generic-injectivity risk entirely:

For each of `pool3`'s $2\times2\times2$ non-overlapping spatial windows (per channel — MaxPool3d
pools spatially only, channels are independent), the window selects the spatial argmax. A
perturbation $\delta$ that is:
- confined to the **non-selected** entries of that window (or, in a genuine tie, confined to the
  non-selected-but-tied entries — a tie means multiple entries equal the max; perturbing any
  one of them downward, or perturbing a strictly-smaller entry that stays strictly smaller,
  leaves the selected value unchanged), and
- small enough in magnitude that the perturbed entry does not exceed the selected (argmax)
  entry's value,

is **exactly** invisible to `pool3`'s output, by construction — provable directly from
`MaxPool3d`'s definition, no autodiff, no rank computation on an enormous matrix. E185 already
measured this is not a rare edge case: **28.75% of pool3's windows on the real trained
activation have an exact tie**, and the broader non-argmax-entry freedom exists in every window
regardless of ties (a window with a unique max still has 7 non-selected entries free to move,
subject to the magnitude bound).

**This closes $\ker(A)$ exactly only for the bottleneck path.** It says nothing analytically
about the skip path, where `dec3` sees raw, unpooled enc3 directly — a $\delta$ confined to
non-argmax pool3 entries is NOT claimed to be invisible there. **Adopted resolution (explicit
user decision, not deferred)**: do not attempt a second, independent analytic nullspace
condition for the skip path (would require freezing `dec3`'s own conv+ReLU layers as selection
matrices too, doubling the derivation work for uncertain payoff). Instead, rely on Step D's
**mandatory empirical check on the real, full network** (both paths, no autodiff, plain forward
passes) — if the skip path's sensitivity to these candidates is small enough in practice that
$d(D(Z+\delta), D(Z)) \le \epsilon$ still holds end-to-end, the construction is validated
empirically even without a from-scratch analytic guarantee for that path. If it does NOT hold,
that is itself the answer (kill or need a smaller magnitude bound) — not a flaw in the design.

## Stages (10 subjects, no training, no GT/Dice until the final comparison)

**Step A — identify the local affine regime.** Run the intact forward pass. Record `pool3`'s
argmax index (or tied index set) per channel per window. This fixes which entries are
"non-selected" for the constructive nullspace source. No ReLU-mask freezing needed elsewhere —
the construction above only depends on `pool3`'s argmax pattern, not on freezing the rest of the
network (that would only be needed for the full-$A$ approach we're explicitly not taking).

**Step B — check the nullspace source is nonempty and usable.** Report, per subject: fraction
of `pool3` windows with a genuine tie (expect ≈29%, per E185); mean number of free (non-argmax)
entries per window (up to 7 per channel-window, generically). This is a sanity check, not a
kill gate by itself — Step D is the actual kill gate, since the analytic guarantee only covers
one of two paths.

**Step C — generate candidates.** For $K$ candidate directions per tile (fixed, e.g. $K=8$,
independently seeded, distinct from every prior probe seed in this project), construct
$\delta_k$ confined to non-argmax `pool3` entries within a tile's enc3 window, at a small
magnitude bound (fixed fraction of the argmax-to-non-argmax gap, calibrated per E181/E182's own
"calibrate before measuring" discipline — not assumed a priori). $Z_k = Z + \alpha \delta_k$ for
a small fixed $\alpha$ grid.

**Step D — mandatory empirical verification on the real network (no shortcuts).** For each
candidate: $\Delta_{\text{out}} = d(D(Z_k), D(Z))$ on the FULL real forward pass (both paths, no
linearization, no autodiff). Require $\Delta_{\text{out}} \ll \epsilon$ (ε fixed independently,
same discipline as E184 — not tuned to make this pass). Also verify `pool3`'s argmax pattern is
UNCHANGED after perturbation (confirms we stayed within the same piecewise-linear regime — if
the regime changed, this is no longer an exact equivalence construction and must be flagged, not
silently kept). Candidates failing either check are excluded, not adjusted post-hoc.

**Step E — measure Γ.** For surviving candidates: $\Gamma(Z_k)$ using the exact same T6/T7-family
definition from E180-E184 (no new metric). Primary question, fixed before running:

$$
\boxed{\text{Var}_k[\Gamma(Z_k)] > 0 \quad \text{while} \quad \text{Var}_k[D(Z_k)] \approx 0}
$$

## Decision table (locked, mirrors the user's own framing)

| Observation | Reading |
|---|---|
| No usable nullspace source (Step B degenerate) or all candidates fail Step D's empirical check | KILL — branch dies on structure, not on Γ |
| Nullspace source usable, candidates pass Step D, but Γ does not vary across them | KILL — nullspace exists but is Γ-irrelevant |
| Candidates pass Step D (output-equivalent) AND Γ varies across them | INTERESTING — same finding shape as E184 Outcome C, now via a constructive (not diagnostic-random) source |
| Γ varies but `pool3` argmax pattern changed under perturbation | WEAKER — not a clean piecewise-linear equivalence, confound to flag not ignore |
| (Only if INTERESTING) Lower Γ correlates with counterfactual Δ | mechanism-supporting, next stage |
| (Only if the above holds) Selecting the lowest-Γ candidate improves Δ | algorithmic potential, still not a paper claim without 125-subject/3-seed confirmation |

**E186 itself does not need to improve Dice.** It answers a structural existence question,
exactly as E184's feasibility probe did — GT/Dice stays out of construction and selection,
entering only if a later stage explicitly tests Δ (not part of this prereg's scope).

## RETRACTION (2026-09-18) — the KILL verdict below is CONTAMINATED, not trustworthy as reported

While implementing E187, a construction bug was found and confirmed in E187's own pipeline, then
found to be SHARED by this script: `Z_full_enc3` (the enc3 reference used to build
`z_override_full`) is captured from a SINGLE, isolated 128³-crop forward pass on `tile_image`,
but then applied via `sliding_window(model, image_b, ...)` — which runs the model **once per
sliding-window tile across the ENTIRE volume** (subjects here are 138×176×144 and similar, well
over 128³ in every axis, so `sliding_window` genuinely iterates multiple overlapping windows,
Gaussian-blended). Since `mask` covers the FULL enc3 spatial extent for tile 0 (all-True for a
subject whose tile 0 spans the complete 128³ region), `OverrideHook`'s
`z = output*(1-m) + z_override_full*m` degenerates to `z = z_override_full` for EVERY window's
own forward pass — silently forcing every window (most of which are NOT the window
`Z_full_enc3` was captured from) to use a mismatched enc3 value that has nothing to do with that
window's own intact activation.

**Directly confirmed this contaminates the reported numbers**: even a construction with a
literal ZERO delta (`z_override_full` = `Z_full_enc3` verbatim, no perturbation at all) produces
a large, nonzero `output_distance` purely from this window mismatch — verified directly on
`BraTS-GLI-01041-000` in the course of debugging E187. The recorded `output_distance_from_Z0`
values below (0.02218-0.02224, varying only in the 4th decimal across different pool-tie
deltas) are consistent with being DOMINATED by this artifact, not by the true magnitude of the
pool-tie construction — the near-identical values across candidates with genuinely different,
independently-seeded deltas is itself the signature of this bug (a correctly-isolated
window-local perturbation should show more spread).

**Consequence**: the "KILL, ‖δ‖/‖Z‖≈0.40" reading immediately below is NOT retracted as a
finding about the delta's true magnitude (that measurement was made directly and independently,
not through `sliding_window`, and stands) — but the **Step D pass/fail verdict itself, and the
121× ε-overshoot statistic specifically, are contaminated and must not be trusted as reported.**
E186 needs to be RERUN with the window-mismatch bug fixed (restrict to a single-window forward
pass, matching E185's own single-tile pattern, rather than running the full multi-window
`sliding_window` on a locally-captured override) before its actual verdict can be trusted. This
retraction is recorded per this project's standing discipline: investigate anomalies found
during LATER work that bear on EARLIER results, rather than leaving a contaminated verdict
standing uncorrected.

## Result (2026-09-18, ORIGINAL — SEE RETRACTION ABOVE) — KILL, decisive and unanimous at Step D

Ran on the locked design, no adjustments after seeing data. 8/10 subjects completed (2 skipped:
`BraTS-GLI-01161-000` and `BraTS-GLI-01168-000` have tile bounds not divisible by 8 in at least
one axis — undersized volumes, a data edge case unrelated to the construction, not a design
change). 64 total candidates (8 subjects × 8 candidates each).

**Construction verified correct** before the real run: unit-tested on synthetic tensors, both a
generic random tile (no ties) and a forced-all-zero tied window. In both cases `F.max_pool3d`'s
output was **byte-identical** (max abs diff = 0.0) before and after the constructed delta —
the analytic guarantee for the bottleneck path holds exactly, as designed.

**Step B (nullspace source sanity)**: `frac_tied_windows` = 0.457–0.490 across subjects (mean
0.477) — consistent with E185's independently-measured ≈0.29 on a different tile/patch scale,
confirming this is a real, substantial, reproducible property of the trained checkpoint's enc3
activations, not a one-tile artifact.

**Step D (the hard gate): 0/64 candidates passed, unanimously.** Two findings, reported
separately since they are different in kind:

1. **Output-equivalence failed by a large, consistent margin.** `output_distance / epsilon`
   ratio: min 41.6×, max 275.5×, mean 121.2× over the threshold — not a marginal miss, a
   two-orders-of-magnitude overshoot on every single candidate. Root cause measured directly:
   `||delta|| / ||Z|| ≈ 0.40` on the first subject checked — the constructed perturbation is
   ~40% of the intact activation's own norm. This is exactly the risk flagged in this document's
   own design section: the construction gives an *exact* guarantee only for the bottleneck path
   (`pool3`'s output), but `dec3` on the skip path sees the same large, dense perturbation
   (perturbing the large majority of non-argmax entries in every window, ~87.5% of entries per
   unique-max window plus most tied-window entries) with **no protection at all**. The skip
   path's sensitivity to this magnitude of change was not small enough for the empirical
   equivalence check to pass — the two-path asymmetry was real and dominant, not a minor
   correction.
2. **A small, separate float-precision leak in the analytic guarantee itself**:
   `frac_argmax_changed` was nonzero on every subject (mean 1.85e-5, i.e. ~10-15 windows out of
   ~524,288 per subject), rather than exactly 0.0 as unit tests on synthetic data showed. Likely
   float32 rounding at the GPU forward pass's own boundary (50% of a gap computed in float32,
   re-applied via the model's AMP-enabled forward, can round into the boundary in rare cases) —
   a real but minor and separate issue from finding (1); fixing it would not have changed the
   Step D outcome, since finding (1)'s ~100× overshoot swamps it entirely.

**Reading, per the locked decision table**: "Nullspace source usable, candidates pass Step D" —
**did not hold**. This falls under "all candidates fail Step D's empirical check" → **KILL**.
Per the prereg's own binding rule, Γ was never computed on any candidate (0 records in
`gamma_records_passing_only`) — the branch is closed at the structural gate, exactly as
specified, with no relaxation of `GAP_FRACTION`, ε, or the argmax-preservation criterion after
seeing this result.

**What this establishes**: the specific constructive nullspace source tested (non-argmax/tied
`pool3` entries, at 50% of the per-entry gap) does NOT give an output-preserving equivalence
class on the real two-path network at this magnitude. It does not establish that no
output-preserving nullspace exists at all — a much smaller `GAP_FRACTION`, or a construction
that also analytically accounts for the skip path (rejected earlier in this document's design
phase as requiring a second independent derivation), might behave differently. Per the explicit
instruction not to relax the criterion after seeing this result, **that is a decision for a new,
separately pre-registered probe, not a live adjustment of E186.**

## CORRECTED RERUN (2026-09-18) — same design, single-tile-inference fix only, still 0/64

Reran with EXACTLY the same locked construction (GAP_FRACTION=0.5, seeds, candidate generation,
Step B/D/Γ logic byte-identical) — the only change was replacing the multi-window
`sliding_window(model, image_b, ...)` calls with a single targeted forward pass on the tile
under study (`single_tile_forward`, matching E187's already-validated fix), per the retraction
above. No other constant or criterion touched.

**Result: still 0/64, but the picture is now completely different and far more informative.**
`output_distance/epsilon` ratio: min 0.547×, max 2.014×, mean 1.353×, median 1.512× — a **razor-
thin near-miss**, not the ~121× overshoot the contaminated run reported. 63/64 candidates land
within 2× of ε; some (subject `01189`, ratios 0.55–0.66×) are comfortably **inside** ε.

**The decisive breakdown**: checking each of Step D's two conditions separately —
- **Output-equivalence alone**: 14/64 candidates pass (subject `01189` all 8, subject `00506` 6
  of 8).
- **Argmax-pattern-preserved alone**: **0/64 candidates pass** — `frac_argmax_changed` is small
  but strictly nonzero on every single candidate (range 7.6e-6 to 2.9e-5, mean 1.85e-5 — 10-15
  windows out of ~524,288 per subject), never exactly 0.0 as the pre-run unit tests on synthetic
  tensors showed.

**This means the sole reason Step D fails, unanimously, is the argmax-preservation check** —
matching what the original (contaminated) report had already separately flagged as "a small,
separate float-precision leak... likely float32 rounding at the GPU/AMP forward boundary" and
noted "would not have changed the Step D outcome" at the time (true then, under the 121×
contamination; **not true now** — with the window bug fixed, this leak is the entire story).

**Reading, per the locked decision table**: this is still, mechanically, "all candidates fail
Step D's empirical check" → **KILL**, exactly as specified, and Γ is correctly never computed
(0 records). Per the explicit instruction against relaxing criteria after seeing results, this
verdict stands as the answer to the question E186 asked. But the REASON has changed in a way
worth stating precisely for whoever reads this next: the construction's analytic guarantee (zero
argmax change, verified exactly in unit tests) does not survive the real GPU/AMP forward pass by
a tiny margin — not because the skip-path asymmetry dominates (the original, contaminated
reading), but because of a separate, much smaller numerical effect at the pool3 argmax boundary
itself. The skip-path asymmetry may still matter (14/64 already fail on output-equivalence alone,
consistent with SOME real skip-path sensitivity), but it is no longer the dominant or unanimous
cause it first appeared to be.

## FP32/AMP vs float64 argmax precision audit (2026-09-18) — the leak IS pure float32 precision

Narrow, targeted check per explicit user request: nothing beyond this. For the 14 candidates
that passed output-equivalence alone (subject `01189`, all 8; subject `00506`, 6 of 8), recomputed
`frac_argmax_changed` at float64 (no AMP), from a fresh float64 forward-pass capture of the same
tile/delta, holding everything else (seed, construction, tile) identical to the float32 run.

**Result: float64 gives exactly 0.0 for all 14 candidates** (float32: 9.5e-6 to 2.5e-5, matching
the original run's values exactly). No candidate shows a genuine argmax change at float64 — the
leak vanishes completely at higher precision, confirming it is a pure float32/GPU numerical
artifact at the pool3 argmax boundary (values landing within float32's own rounding error of a
tie), not a real property of the construction, the skip path, or the model's actual computation.

**Frozen interpretation, per explicit user instruction — no further probing, no E188, no Γ,
no relaxed criteria**: the analytic guarantee (zero argmax change) is exact at float64 and only
appears to leak under float32 due to rounding, not due to any real skip-path or two-path
interaction. Combined with the 14/64 output-equivalence-alone pass rate, this means: **at higher
precision, this specific pool-tie construction likely WOULD produce a nonzero pass count on
Step D** — the 0/64 as literally measured is an artifact of the evaluation precision, not
evidence that the construction is fundamentally incompatible with output-equivalence. This is
recorded as a precision finding about THIS measurement, not as a re-opening of E186 or a license
to rerun it at float64 — that would be a new, separately-scoped decision, not made here.

**How this bears on the broader picture (context, not a new experiment)**: E184's large-scale,
whole-tile random-direction perturbation is still the only construction in this family that has
shown genuine Γ heterogeneity. E186 (pool-tie, "surgical") and E187 (single-boundary-crossing,
"surgical") have both, in different ways, failed to reach a stage where Γ heterogeneity could
even be tested — E187 reached it and found Γ flat; E186 never reached it, and this audit shows
its failure to reach it was itself largely a measurement artifact rather than a real structural
barrier. This leaves the "surgical vs. large-scale" question more open than the original 121×
or even the corrected-but-still-0/64 readings suggested — worth weighing, not yet resolved,
before deciding on any next step.

## Retrospective commutator audit (2026-09-19) — separate diagnostic, not a new experiment

Continued into a retrospective diagnostic analysis of the task-space commutator
$C_{\text{task}}(P) = \frac{1}{6}\sum_T d[D(T(PZ)), D(P(TZ))]$ on E184's and E187's already-
collected candidates (full detail in
`experiments/exp_e12_eggo_m/e_commutator_audit/` and its own memory files). Headline: a small,
control-surviving association between $C_{\text{task}}$ and $|\Delta\Gamma|$ in E184 (within-
subject Spearman ρ≈0.20, survives output-distance and perturbation-magnitude partials and a
2000-draw subject-preserving permutation null), but E187 does not serve as a clean negative
control for the commutator specifically (ρ=0.36–0.62 depending on outlier handling) even though
E187's own Γ was independently found flat. A follow-up per-probe decomposition ($C_{\text{mean}}$,
$C_{\text{max}}$, concentration $K=C_{\max}/\sum_T C_T$) tested whether E184 shows a distributed
commutator response while E187 shows a concentrated one — **found no distributional difference
between E184 and E187 on any of the three statistics** (Mann-Whitney p=0.38–0.77). The
"distributed vs. concentrated response" hypothesis proposed to explain the E184/E187 paradox
does not hold in this data. Frozen verdict: the commutator is a real but modest mechanistic
correlate of Γ-change in E184, not (yet) a principle that cleanly discriminates E184's regime
from E187's.

## What this does NOT do

No training. No architecture change. No autodiff/gradient anywhere (this is precisely what
distinguishes it from the killed E185 route — candidates are constructed from known `pool3`
structure, verified by plain forward passes only). No GT/Dice in construction or selection. No
claim of novelty for the null-space construction itself (prior art is real and acknowledged
above) — the claim under test is narrower and specific to this project's Γ→Δ phenomenon.
