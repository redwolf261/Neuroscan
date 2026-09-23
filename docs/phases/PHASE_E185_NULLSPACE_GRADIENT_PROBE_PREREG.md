# E185 — Does the output-null space contain a Gamma-changing direction? (differential probe)

**Date**: 2026-09-18
**Status**: PREREG + single small script. NOT a repair operator, NOT an optimizer, NOT a scale
experiment. One tile, one subject to start, gradient-only. Explicitly NOT the next full
experiment — a cheap decisive check before spending any serious compute on E184's four design
questions.

## The question

E184's Γ-continuation established, at **finite displacement** (α ∈ [0.5, 4.0], discrete
directions), that $D(Z_a)\approx D(Z_b)$ while $\Gamma(Z_a)\neq\Gamma(Z_b)$ (Outcome C). That
result says nothing about whether this is a *local, linear* phenomenon or purely a *nonlinear,
finite-displacement* one. The distinction matters for what kind of repair operator is even
worth designing:

$$
\boxed{P_\perp \nabla_Z \Gamma(Z) \neq 0 \;?}
$$

where $P_\perp$ projects onto $\ker J$, $J = \partial D/\partial Z$ the decoder's local Jacobian
at $Z$ (output-null directions — infinitesimal moves that don't change $D(Z)$ to first order).

- **If yes**: the output-null tangent space contains a direction that changes Γ to first order.
  The phenomenon has a *linear* handle — gradient-based repair (S1 additive, or a Gauss-Newton
  style projected step) is well-motivated, not just a finite-displacement curiosity.
- **If no** ($P_\perp\nabla_Z\Gamma \approx 0$): $\nabla_Z\Gamma$ lives entirely in
  $\ker(J)^\perp$ at this point — moving along it costs output fidelity to first order. E184's
  effect is then a genuinely nonlinear/finite-displacement phenomenon only. This particular
  **differential formulation** of repair dies; it does not kill E184's finite-α finding, and
  does not by itself kill repair generally (a large-step / nonlinear search could still work) —
  it specifically kills the "small gradient step in the null space" approach to designing S1's
  optimizer.

## Why this is cheap and decisive, and why it's the right next move

No optimizer, no new subjects beyond 1 (extendable to a handful only if the single-tile result
is ambiguous), no GT, no new architecture. Two gradients through the frozen checkpoint at one
point, one JVP. This directly discriminates between two of the four design questions (search
space S1 vs S5, and whether a smooth optimizer surrogate is even sensible) **before** committing
engineering effort to either — exactly the standing discipline of resolving existence before
architecture (same ordering principle as E184's own "existence before the four design
questions").

## Definitions (reusing E180-E184's objects, one substitution recorded explicitly)

**Point**: one enc3 tile $Z$ from one subject (start: `BraTS-GLI-00250-000`, the first subject
in the existing E184 ledger, arbitrary but fixed — not cherry-picked for outcome).

**Smooth Γ surrogate** (substitution from the true nonsmooth, thresholded-Dice Γ used
throughout E180-E184 — recorded, not silently reused as if identical):

$$
p_c = D(T7(Z, c)) \quad \text{(softmax probability map, NOT thresholded)}, \quad c \in
\{0.60, 0.80, 1.10, 1.45\} \text{ (T7's existing calibrated severities, unchanged)}
$$

$$
\text{pairdist}(c_j, c_k) = \lVert p_{c_j} - p_{c_k} \rVert_2^2
$$

$$
\Gamma_{\text{soft}}(Z) = \frac{1}{\tau}\log\sum_{j<k} \exp\big(\tau \cdot
\text{pairdist}(c_j,c_k)\big), \qquad \tau \text{ fixed (start } \tau{=}50\text{, chosen only
to make the softmax reasonably peaked, not tuned against any result)}
$$

**T6 is dropped from this probe only** (SVD+SciPy-solver chain is not cleanly
autograd-differentiable end-to-end, and T7 alone is a valid family member to test the existence
question — the claim under test is about *any* Γ-defining family exhibiting the phenomenon
differentially, not specifically T6+T7 jointly). If $P_\perp\nabla\Gamma_{\text{soft}}\neq 0$
using T7 alone, that already answers the question in the affirmative; a null result here would
need T6 checked separately before generalizing to "no null-space direction exists at all" (not
required for this probe — recorded as a scope limit, not resolved now).

**Jacobian-null projection, without forming $J$**: given $v = \nabla_Z\Gamma_{\text{soft}}(Z)$
(one backward pass), compute $Jv$ via a single Jacobian-vector product (forward-mode autograd,
or the standard double-backward trick: $Jv = \nabla_u \langle u, D(Z)\rangle \big|_{u \leftarrow
\text{dummy}, \text{ then differentiate the VJP w.r.t. a seed in direction } v}$ — concretely,
`torch.autograd.functional.jvp(lambda z: D(z), Z, v)`). Then:

$$
\text{null-fraction}(v) = 1 - \frac{\lVert Jv \rVert}{\lVert J v_{\text{rand}} \rVert}
$$

where $v_{\text{rand}}$ is a random direction of the **same norm as $v$** (matched-norm control
— without this, any small vector trivially has small $Jv$ and the ratio is meaningless). This
directly operationalizes $P_\perp\nabla\Gamma \neq 0$: if $\lVert Jv\rVert \ll \lVert
Jv_{\text{rand}}\rVert$, $\nabla\Gamma_{\text{soft}}$ is disproportionately output-null relative
to a generic direction of the same size — the affirmative case. If $\lVert Jv\rVert \approx
\lVert Jv_{\text{rand}}\rVert$, $\nabla\Gamma_{\text{soft}}$ is a generic (non-null-preferring)
direction — the negative case.

## Decision rule (fixed before running)

Let $r = \lVert Jv\rVert / \lVert Jv_{\text{rand}}\rVert$ (mean over ≥5 random-direction draws,
matched norm, independently seeded, distinct from every existing probe seed in this project).

- $r < 0.3$: **YES** — $\nabla\Gamma_{\text{soft}}$ is substantially output-null. Proceed to
  design a gradient-based S1/S4 repair step around this.
- $r > 0.7$: **NO** — $\nabla\Gamma_{\text{soft}}$ is a generic direction, no preferential
  alignment with $\ker J$. This differential formulation dies; E184's effect stays a
  finite-displacement-only phenomenon (E184's Outcome C result itself is NOT retracted by this
  — it used discrete α-sized moves, this probe is strictly local/first-order).
- $0.3 \le r \le 0.7$: **AMBIGUOUS** — report as such, do not force a verdict; consider a second
  subject/tile only if this exact band is hit (not a free license to keep sampling until a clean
  answer appears).

Sanity checks required before trusting $r$: (a) $\lVert v \rVert \neq 0$ (Γ_soft actually has a
nonzero gradient at this point — if zero, the checkpoint sits at a critical point of Γ_soft and
the whole test is vacuous, report and stop); (b) JVP implementation verified against a
finite-difference check on a small subset of directions ($D(Z+hv) - D(Z) \approx h\cdot Jv$ for
small $h$) before trusting the autograd JVP numerically, given this project's own prior
GPU-autograd/SVD numerical surprises (E181-B).

## Result (2026-09-18) — probe INVALIDATED by a MaxPool3d tie-breaking artifact, not run to a verdict

Two attempts were made (corner sub-block of tile 0, then a centered sub-block), both on GPU at
float64 after a float32 attempt was found numerically untrustworthy (finite-difference sanity
check failed catastrophically, root-caused to float32 precision loss at this model's activation
scale — not a real autograd bug, confirmed via `torch.autograd.gradgradcheck` passing at
float64). The float64 GPU run OOM'd at the full 128^3 patch; fixed by shrinking to a 32^3
input patch for this probe only (`PROBE_PATCH`), which does not require full resolution to
answer a purely local/differential question.

**A second, more serious numerical issue was found and root-caused, not worked around.** At the
real (trained-checkpoint-derived) enc3 activation, the JVP-vs-finite-difference sanity check
failed again (relative error ≈1.0, cosine ≈ −0.24 to −0.32), but this time the mismatch was
**stable across every step size tested down to h=1e-10** (machine-precision territory for
float64) — ruling out ordinary precision loss. Both reverse-mode (manual double-backward) and
forward-mode (`torch.func.jvp`) AD were tested and agree with each other exactly, while both
disagree with the finite-difference estimate — ruling out an AD-mode-specific bug. Bisection
isolated the discrepancy to `pool3` (`MaxPool3d`) alone, reproduced on a 2-op subgraph
(`pool3`+`bottleneck`) with the real activation, and NOT reproduced when the same subgraph was
fed a fabricated dense random tensor of matched shape/scale. Direct inspection of the real enc3
tile found **28.75% of MaxPool3d's 2×2×2 pooling windows contain an exact tie for the maximum**
(almost entirely because the tied value is 0.0 — ReLU sparsity: 52.6% of enc3's entries are
exactly zero). PyTorch's MaxPool3d backward breaks ties by routing gradient to a fixed
(first-found) argmax index; with this many ties, the resulting Jacobian-vector product is a
well-defined but essentially arbitrary linearization that does not match the one-sided
directional derivative finite differences measure (which sees whichever tied index the
perturbation happens to nudge into becoming the strict max) — and, because it is a genuine tie
rather than a near-tie, no amount of shrinking h resolves the disagreement.

**Consequence**: the r-ratio computed at either sub-region tested (63.65 corner / 80.65 center)
is **not trustworthy as evidence for the P_perp question** — it reflects MaxPool3d's tie-breaking
convention, not a property of the decoder's true local Jacobian or of grad(Gamma_soft)'s
alignment with it. This is a property of ReLU-network representations generically (any point
with tied/zero pooling-window maxima has this issue), not specific to one subject or the
null-space question — it would recur at essentially any enc3 tile from this checkpoint, since
~50%+ ReLU sparsity at this depth appears typical, not an outlier of the two tiles tried.

**Status: probe INVALIDATED, not resolved YES/NO.** This is not evidence for or against
$P_\perp\nabla_Z\Gamma \neq 0$ — it is a finding that the differential (gradient/JVP) approach to
this question, AS IMPLEMENTED (autodiff through the real MaxPool3d-containing decoder), is not
currently answerable this way without a fix to the tie-breaking degeneracy. Recorded per this
project's standing discipline: an anomaly was investigated to its root cause rather than
reported as a number, and the finding is reported as what it actually is (a tooling limitation
discovered), not smoothed into a false YES/NO.

## Decision (2026-09-18, user) — gradient-based branch KILLED, derivative-free adopted

**Not** because E185 established $P_\perp\nabla_Z\Gamma = 0$ or $\neq 0$ — that question is
still formally open. Killed because E185 demonstrated the computational object a gradient-based
optimizer would need (a reliable Jacobian/JVP of $D$ at the actual trained enc3 representation)
is **ill-conditioned at the operating point that matters**: autodiff(J) and finite-difference(J)
disagree by ~100%, stably down to h=1e-10, localized to MaxPool3d's tie-breaking under ~53%
ReLU sparsity. This is a property of the real representation, not an artifact of one tile choice
— expected to recur broadly at this depth. A gradient-based optimizer (S1/S4 with smooth
surrogate, or the JVP-based projected-step idea E184 sketched) would be optimizing against an
unreliable primitive at exactly the point it needs to be reliable.

**Rejected fixes, and why:**
- **(a) Smoothed/soft-max pooling surrogate** — rejected. Would make the optimizer act on
  $D_{\text{smooth}}(Z) \neq D_{\text{actual}}(Z)$, a different decoder than the one E184's
  entire phenomenon is about. Would require a second, unwanted research problem (does the
  smoothed-decoder optimum transfer to the real deployed decoder?) that the project has no
  interest in taking on.
- **(b) Finite differences as the optimizer basis** — rejected as a *production* method, kept as
  a *diagnostic* only. At ~7s/Γ-evaluation (E184's measured cost) and a 128×32³-dimensional
  latent, a finite-difference gradient is computationally absurd, and Γ's own max-over-six-
  predictions construction makes its FD landscape noisy on top of that.

**Adopted: derivative-free search over a low-dimensional candidate orbit.** The decoder is never
differentiated. General form:

$$
\boxed{Z^\star = \operatorname*{arg\,min}_{Z' \in \mathcal{C}(Z)} \Gamma(Z') \quad \text{s.t.}
\quad d(D(Z'), D(Z)) \le \epsilon}
$$

where $\mathcal{C}(Z)$ is a small, predetermined candidate set (NOT full-dimensional random
search — "perturb until something works" is not a contribution). Simplest instantiation, not
yet the final design:

$$
\mathcal{C}(Z) = \{Z + \alpha V_j : j = 1,\ldots,m,\ \alpha \in A\}, \qquad
(j^\star,\alpha^\star) = \operatorname*{arg\,min}_{j,\alpha} \Gamma(Z+\alpha V_j) \ \text{s.t.}\
d(D(Z+\alpha V_j), D(Z)) \le \epsilon
$$

Uses only the model's own predictions and counterfactual response — no GT, no Dice, no
gradients, no learned repair network, no new architecture, no new loss.

## The actual open question now: candidate-orbit construction, not the optimizer

**Explicitly not yet resolved, and this is now the real invention target** (status board, most
recent user framing):

| Component | Status |
|---|---|
| E184 prediction-equivalence phenomenon (Outcome C) | PASS |
| Γ differs within the equivalence class | PASS |
| Output displacement explains Γ (outcome B) | REJECTED |
| Gradient/Jacobian route (E185) | KILLED |
| Smoothed-decoder gradient | diagnostic only, not method |
| Finite differences | diagnostic only, not method |
| Derivative-free search | PREFERRED |
| Generic random perturbation as $V_j$ | NOT novelty |
| $\min_{\mathcal{E}_\epsilon} \Gamma$ as a principle | correct, but insufficient alone for novelty |
| **Candidate-orbit construction $\mathcal{C}(Z)$** | **next invention target, unresolved** |

The constraint on $\mathcal{C}(Z)$: it must give meaningful degrees of freedom **without simply
reproducing T6 (spectral shape) or T7 (energy) or E184's own diagnostic random directions** —
those are all already-known territory, not new. The novelty has to live in *how the candidate
representation is constructed or selected*, not merely in "derivative-free beats
derivative-based" (that's a justified engineering choice, not itself the contribution). This is
the next thing to derive on paper — not yet started, not yet coded.

## What this does NOT do

No optimizer step is taken. No repair candidate is constructed. No GT/Dice anywhere (Γ_soft is
prediction-vs-prediction, matching E180-E184's confound guard). No claim about T6. No claim
about other tiles/subjects beyond establishing existence at one point, extendable only if
ambiguous. This is a differential add-on to the already-passed finite-displacement existence
gate (E184 Outcome C) — not a replacement for it and not a new full experiment.
