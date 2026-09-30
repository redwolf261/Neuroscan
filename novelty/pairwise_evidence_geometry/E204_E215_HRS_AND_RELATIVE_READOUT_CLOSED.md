# E204-E215: HRS and Relative-Evidence-Readout branch CLOSED

Frozen E131_v5control_seed0 throughout. No training. Dice branch (0.868620)
untouched. Renumbered from the original E190-E197/E212-E215 working numbers
to avoid colliding with pre-existing docs/phases/PHASE_E190-E200 (unrelated
earlier experiments) -- see E204_E211_RENUMBER_NOTE.md for the mapping.

## Chain

E204(gate) -> E205/E206(Hamiltonian, killed) -> E207(Evidence Closure,
killed) -> E208/E209/E210(HRS forensic, passed) -> E211(HRS-Lite, killed) ->
E212(hypothesis selectivity, killed) -> E213(relative evidence readout,
inert control mixed) -> E214(s_h inert control, PASSED) -> E215(full-volume
Dice, KILLED).

## E204-E207: SPA / Hamiltonian correction / Evidence Closure -- all KILLED

E204 gate: SPA's gradient map untrustworthy at E3 (Spearman 0.41-0.53,
MaxPool-tie subgradient arbitrariness), exact at bottleneck (0.999-1.000).
E205: K=||grad h||^2/(h^2+eps) at E3 collapses under magnitude control
(partial r=0.114, FAIL); at bottleneck survives boundary AND magnitude
controls (partial r=0.404, p=4.8e-10) -- genuine existence.
E206: h'=h(1+alpha*K) applied at bottleneck moves dz_L the WRONG sign
(-0.0975 at alpha=+0.25), 0/74 crossings. Existence != actionability.
E207 "Evidence Closure": individual bottleneck-voxel AUC already 0.9945-1.0
(L/S vs H) via plain logistic regression -- aggregation adds Delta~=0.001.
Third independent confirmation the gap is at the READOUT (see E148-150).

## E208-E210: HRS forensic chain -- PASSED, genuinely new phenomenon

E208 (donor-texture splicing): Delta_E (shell-logit shift after inserting
donor tumor texture) discriminates L from H, p=1.4e-13, survives a
reactivity partial-correlation control (r=0.297, p=5.0e-7).
E209 (crossing test): Delta_E predicts actual segmentation crossing for L
specifically (AUC=0.747, survives reactivity control, r=0.352, p=0.0067) --
and does NOT generalise to H (partial r=0.080, p=0.50, NOT significant).
Clean dissociation, the strongest result of the whole branch.
E210 (endogenous bridge): an endogenous representation-space direction
(mu_prototype - mu_local, no GT leakage) reproduces and EXCEEDS E209's
crossing rate: 27/74 (36.5%) vs a norm-matched random direction's 0/74,
p<0.001. Monotonic dose-response, directional not magnitude-driven.

## E211: HRS-Lite (learned proposal + evaluator) -- KILLED

5-fold subject-grouped CV. Prototype replication held exactly (27/74,
matching E210). But the TRAINED proposal did not beat random on held-out
crossing (0/58 both, Fisher p=1.0) or dz (Mann-Whitney p=0.355). Evaluator
comparison: R[reactivity]->AUC 0.578, C[+dz]->AUC 0.787 (large real jump),
H[full features]->AUC 0.804 (Delta R^2 vs C = -0.0066, negligible). dz is
the informative quantity; nothing else in the richer feature set adds
anything. Proposal training (dz-maximization only, no L_rank, 30 epochs,
~46-47 train components/fold) was under-specified for learning a 256-dim
direction from scratch -- E210's precomputed prototype direction was not
successfully rediscovered by the learned proposal in this v0 attempt.

## E212: Hypothesis-Response Selectivity -- KILLED (Gate F)

S_H = Delta_L - Delta_B, H_B = 3 frozen Gram-Schmidt-orthogonalized random
directions (norm-matched to H_L), per component. Gate D/E/G all pass
cleanly (S_H(L) vs S_H(H) p=4.6e-23, AUC=0.963, H shows 0/296 crossings).
Gate F (the decisive one, built specifically to catch this) FAILS: partial
corr(S_H, crossing | R_mag) = -0.016, p=0.79. Mechanism: for a random
orthogonal direction, Delta_B~=0, so S_H~=Delta_L by construction --
orthogonalization did not create an independent axis of information, it
created a near-duplicate of raw response magnitude (E211's dominant C
feature). Third repetition this session of "signal correlates cleanly in
isolation, dies under the reactivity control built to catch exactly this."

## E213-E214: Relative Evidence Readout -- E153 pattern, then a real survivor

E213 tested 3(A/B/C) spatial/subject-relative logit scores plus 1(D)
bottleneck-relative score s_h = <h(x)-mu_B, mu_L-mu_B>/||mu_L-mu_B||, all
against the MANDATORY E153-style inert control (resample the reference
population at matched boundary-distance, z(x) held fixed -- E153's
fractional-damage responsibility matrix collapsed exactly this way).
Spatial-relative scores (A/B) FAIL the inert control (corr 0.66-0.80 under
resampling) -- real E153 artifacts, denominator-driven. s_h was never
tested by that specific control in E213 (a gap, not a result) but E213's
component-proxy Dice looked excellent (0.9175 vs z's 0.8905, no FP cost) --
built from a THIN, ARBITRARY mu_B (single fixed corner voxel bn0[:,0,0,0]).

E214 built the missing inert control for s_h specifically: 5 mu_B
constructions (corner/opposite/edge/spatial_global/spatial_subject).
s_h SURVIVED: corr(corner, spatial_subject)=0.917, and the PROPER
background reference (spatial_subject) gave a STRONGER result than the
arbitrary corner (AUC 0.920 vs 0.851, Dice-proxy 0.9356 vs 0.9272) -- the
opposite of an artifact signature. First result in the whole E204-E214
chain to survive every control applied to it.

## E215: full-volume decision-rule swap -- KILLED, decisively

Built dedicated GPU-resident sliding-window inference (existing
sliding_window_predict never exposes the bottleneck; had to hand-walk
enc->bottleneck->decoder per tile). ROOT-CAUSED AND FIXED A REAL STALL:
initial design upsampled the full 256-channel bottleneck to input
resolution BEFORE blending across tiles (2.15GB/tile, pinned an 8GB card at
7.9GB/0% util with zero progress) -- fixed by computing s_h (1 channel) per
tile before upsampling, 256x smaller footprint, entirely GPU-resident
throughout per explicit user instruction. ALSO CAUGHT a second bug after
first results looked wrong: percentile-of-pooled-brain-voxels threshold
grids are dominated by the background-voxel majority, leaving the actual
decision boundary (z~0) unsampled -- produced a spuriously low baseline
(0.659 vs production's real ~0.82). Fixed with a dense grid anchored at the
natural zero point; corrected baseline z Dice = 0.8224, matching production.

WITH THE CORRECTED BASELINE: s_h alone collapses to 0.4657 (-0.357 vs z).
z+alpha*s_h: best alpha=0.0 (monotonic degradation as alpha rises:
0.8226->0.8209->0.8202->0.8162->0.7958->0.6751). Gated substitution, even
restricted to the most extreme 1% of s_h: 0.7102, still -0.112 vs baseline.
Delta = +0.0000 for the best system.

**Discovery/validation evidence on the SAME 125-subject population
E204-E214 were built on, explicitly not a held-out test** -- but the result
is a clean, decisive negative, so no held-out follow-up is warranted.

## Terminus

E213/E214's component-level Dice-proxy improvement did NOT transfer to real
whole-volume Dice. s_h has real, control-surviving discriminative power on
a curated set of known lesion/hard-negative component sites (the exact
scale E204-E214 tested at) but produces catastrophic false positives when
applied indiscriminately across ordinary background tissue -- a scale gap
the component-level proxy structurally could not detect, since it never
sampled generic background voxels away from any lesion or hard-negative
site.

## Durable lessons

1. **Existence at a locus surviving every confound control does not imply
   the derived correction/readout is load-bearing at deployment scale.**
   Established twice this session (E205/E206's Hamiltonian correction,
   E213/E214/E215's relative-evidence readout) via two INDEPENDENT
   mechanisms: E206 failed because amplifying an already-uninformative
   signal amplifies nothing useful; E215 failed because a
   component-level-discriminative score generalizes catastrophically
   poorly to the true, overwhelmingly-background-dominated decision space.
2. **A component-level or curated-population proxy metric is not a
   substitute for whole-volume/whole-population validation** even when the
   proxy is itself methodologically sound (LOSO, subject-grouped,
   permutation-tested). The proxy answers "does this discriminate at known
   sites of interest," not "does this work as a general decision rule."
3. **Percentile-based threshold grids on severely imbalanced score
   distributions can leave the actual decision boundary unsampled.** Use a
   grid anchored at the score's natural zero/decision point, not a
   percentile of the pooled (background-dominated) distribution.
4. **GPU memory: never upsample a wide-channel field to full input
   resolution before a cheap reduction (dot product, projection) is
   available.** Compute the reduction at native (bottleneck) resolution
   first; upsample only the reduced result.
5. E211/E212 both independently rediscovered the same failure mode as
   E206: an intervention/statistic that correlates with an outcome loses
   that correlation the instant you control for total response magnitude
   / reactivity. This is now a load-bearing gate for any future candidate
   in this project, not just a one-off check.
