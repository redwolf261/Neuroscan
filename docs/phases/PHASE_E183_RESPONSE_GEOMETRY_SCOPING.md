# E183 (pre-implementation) — Response geometry: conceptual scoping, no code yet

**Date**: 2026-09-18
**Status**: SCOPING ONLY. No transforms, no compute, no implementation. This document exists to
freeze the reasoning before any code is written — the discipline explicitly requested: "I do not
want to code E183 yet... one more conceptual decomposition first."

---

## Where this sits in the program

$$
\text{Rank} \to \text{instability} \to \text{spectral mechanism (E181)} \to
\text{cross-family phenomenon (E182)} \to \textbf{response geometry} \to
\textbf{representation repair} \to \text{algorithm}
$$

E180-E182 established and cross-validated a phenomenon: representation-induced instability (Γ)
under controlled interventions (T6 spectral-shape, T7 magnitude) is associated with recoverable
segmentation benefit (Δ), surviving subject effects, competitor signals, and (with caveats)
magnitude conditioning. This document is the bridge from "the phenomenon is real" to "what
computational principle, if any, follows from it" — **before** committing to any algorithm
design or new experiment.

## Prior-art check performed before proposing anything (informal, not a formal audit doc)

Every "obvious" version of this idea was checked and rejected as occupied, consistent with
E162's structural law and the E180 novelty audit's finding pattern:

| Candidate | Verdict | Why |
|---|---|---|
| A — Γ-guided attention | Reject | Too close to uncertainty-guided/adaptive feature refinement (spatially adaptive feature refinement, UANeT-style boundary refinement) |
| B — spectral normalization | Reject | Too close to existing spectral normalization / spectral representation work |
| C — Γ-guided extra computation | Weak | Spatially adaptive refinement (selective compute by region) already exists |
| D — perturbation-consistency loss | Weak | Consistency/robustness-under-perturbation is an enormous existing literature |
| **E — counterfactual representation repair** | Worth attacking experimentally | Distinctive only if the specific chain (representation-space intervention → response geometry → constrained repair) is not itself occupied — **not yet novelty-audited**, this scoping only establishes it as worth testing, not as differentiated |

**Explicit reminder carried forward**: counterfactual methods and local sensitivity/Lipschitz
analysis are established fields on their own. Any eventual novelty claim rests narrowly on the
specific chain, not on "using perturbations" or "using counterfactuals" as such — same discipline
as the E180 novelty audit (TRUST/CertainTTA/APEX).

## The conceptual gap this document identifies (the reason not to code yet)

The current $\Gamma_i = \text{diam}\{D(T(Z_i))\}$ construction (a scalar dispersion measure) mixes
at least three distinct things:

$$
\text{representation instability} = \text{decoder sensitivity} + \text{perturbation magnitude}
+ \text{direction of perturbation}
$$

E182 controlled **magnitude** (T7's calibrated, non-degenerate severity ladder). **Direction**
remains uncharacterized. Two representations can have identical Γ (same diameter of response)
while differing completely in structure:

- Representation A: perturbations move predictions in one coherent direction.
- Representation B: perturbations scatter predictions in many unrelated directions.

A scalar Γ cannot distinguish these. The candidate next object is a **response set/manifold**,
not a scalar:

$$
\mathcal R(Z_i) = \{D(T_k(Z_i)) - D(Z_i)\}_{k=1}^K
$$

and the open question is whether the **geometry** of $\mathcal R(Z_i)$ (not just its diameter)
carries information about recoverability — e.g. via its covariance structure, effective
dimensionality, or principal directions — that Γ alone discards.

## The proposed algorithm direction — "Counterfactual Representation Repair" (working name)

**Not to be implemented from this document.** Recorded for continuity only.

$$
Z_i^* = \arg\min_{Z' \in \mathcal E(Z_i)} \Big[ \mathcal S(D(T(Z'))) + \lambda \, d(D(Z'), D(Z_i)) \Big]
$$

— search within a constrained representation family $\mathcal E(Z_i)$ (e.g. matched rank and/or
energy, per E181/E182's invariant constructions) for a representation that is more stable under
the same transformation family, while staying close to the original prediction (a self-consistency
anchor, since minimizing instability alone could produce a stable-but-wrong representation — this
failure mode is explicitly flagged, not glossed over).

This is framed as categorically different from attention ($Z' = A(Z) \odot Z$, "which features to
weight") — it asks "which representation is internally stable under controlled counterfactual
transformations," a different computational question. Whether that distinction survives contact
with the literature is exactly what a future novelty audit (not yet run) would need to determine.

## What must be resolved before E183 is implemented (open questions, not yet answered)

1. **Does the response-geometry object even exist as something measurably richer than Γ?** I.e.
   does $\mathcal R(Z_i)$'s shape (not just its diameter) carry independent information about Δ,
   beyond what the scalar Γ already captures? This is an empirical question, answerable cheaply
   with existing E181/E182 machinery (multiple transform outputs are already computed per tile;
   the geometry just hasn't been analyzed as a set/manifold rather than reduced to a diameter).
2. **Direction-vs-magnitude decomposition**: can perturbation direction be separated from
   magnitude in a way analogous to how E182 separated magnitude from shape? Not yet designed.
3. **Stability ≠ accuracy failure mode**: any repair mechanism needs the fidelity-anchor term
   ($\lambda \, d(D(Z'), D(Z))$) validated as actually preventing "confidently wrong but stable"
   outcomes — not yet tested even in the cheapest form.

## What comes first — the actual next step (still not E183's full implementation)

Per the explicit instruction, the correct next move is **smaller and more mechanistic than a
5000-record statistical run**: characterize whether response geometry (not just Γ's scalar
diameter) is measurably informative, using data structures already available from E181/E182's
per-tile, per-transform records — a re-analysis question first, before any new transform or new
GPU compute. This is scoped as a distinct, smaller step, not written here, and not started
without further direction.

## What this document does NOT authorize

- No new transform implementation.
- No new GPU compute.
- No "E183" full experiment run under this framing.
- No commitment to the repair-algorithm's exact functional form (the $\arg\min$ above is a
  sketch, not a specification).
