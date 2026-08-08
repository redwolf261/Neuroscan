# Phase E9: Trainable Latent-Boundary Head — Mathematical Specification

**Status**: ✅ Complete — mathematics only, no code yet

**Date**: 2026-08-04

## Problem being solved

`PHASE_E5_ALGORITHM_DESIGN.md` §6 (Failure Mode 6) and the E6 stress-test
review both flagged the same issue: the boundary-distance measure $d_i$
used throughout E1.3, E1.4, E6, and E7 was fit **offline**, once, via
`sklearn.linear_model.LogisticRegression` on a static, frozen, already-fully-trained
checkpoint. Using that same external-refit approach *during* active
training creates two problems: (1) computational overhead from
periodically refitting an external model mid-training, and (2) a
staleness/feedback risk — if $\mathcal{L}_{margin}$ (E10) is actively
reshaping `dec1`'s representation while $d_i$ is only occasionally
refreshed, the boundary estimate and the representation it measures can
drift apart, exactly the mechanism the original design review called out
as Failure Mode 6.

## Design: replace the external classifier with an in-network linear head

$$d_i = w \cdot z_i + b, \quad z_i \in \mathbb{R}^{32}$$

Implemented as a single 1×1×1 convolution applied directly to `dec1`,
architecturally identical in form to the existing `seg_head` and
`evidential_head`:

```python
self.boundary_head = nn.Conv3d(32, 1, kernel_size=1)
```

This is mathematically exact — a 1×1×1 convolution over a 32-channel
feature map computes exactly $w \cdot z_i + b$ per voxel, the same
functional form as the offline logistic regression's decision function
(§2.2 of `PHASE_E5_ALGORITHM_DESIGN.md`), just with $w, b$ now being live
network parameters trained jointly with everything else, not fit
externally after the fact.

### 1. Where it attaches, and what supervises it — CORRECTED after implementation-readiness review

**Original design (superseded, kept here for the record — see the
correction immediately below)**: attaches to `dec1` like the seg and
evidential heads, trained with plain BCE against ground-truth class.

**Why this was wrong, caught during the E11.5 readiness review**: BCE
against tumor labels, with gradients flowing back into `dec1`, makes the
boundary head a **second segmentation classifier** feeding pressure into
the shared trunk — `dec1` would now be optimized to be separable for
*two* independent classification objectives ($\mathcal{L}_{seg}$ via
`seg_head`, and now also $\mathcal{L}_{boundary}$ via `boundary_head`),
which is an unbudgeted, unintended architectural change beyond what E9
originally set out to do (E9's stated goal was only to keep $B_i$
*current*, not to add new pressure on what `dec1` optimizes for). This
is exactly the "Option A" problem: it's technically a working
classifier, but not a neutral measurement device anymore — it actively
reshapes the representation it's supposed to be reading.

**The alternative bare "Option B" (no supervision at all, let
$\mathcal{L}_{margin}$ alone shape the boundary head) was also rejected**:
without any classification signal, nothing forces $d_i$ to actually
track class-separability geometry — the head could learn *any* direction
that happens to help $\mathcal{L}_{margin}$ locally, breaking the
interpretability and the required E1.3 AUC/R² validation check (E9 §5)
entirely.

**Corrected design — supervised, but gradient-isolated from `dec1`**:

```python
d = self.boundary_head(dec1.detach())
```

`boundary_head` still receives `dec1` as input and is still trained with
BCE against ground-truth class (so it remains a real, checkable,
interpretable measurement — the E1.3 AUC/R² validation in §5 still
applies exactly as before). **The critical change: `dec1` is detached
before being passed into `boundary_head`.** This means:

- $\mathcal{L}_{boundary}$'s gradient still updates `boundary_head`'s own
  weights $w, b$ (so the head keeps learning to accurately *read*
  whatever separability structure already exists in `dec1`).
- $\mathcal{L}_{boundary}$'s gradient does **NOT** reach `dec1` — the
  shared trunk's only classification pressure remains exactly
  $\mathcal{L}_{seg}$ via `seg_head`, unchanged from the frozen baseline.

This is the resolution to the Option A/B tension: the head is
**supervised** (so it stays meaningful and validatable, unlike bare
Option B) but **does not create new gradient pressure on the trunk**
(so it isn't a second segmentation head competing with `seg_head`,
unlike Option A). It learns to track whatever class structure `dec1`
already has — driven by $\mathcal{L}_{seg}$ and (once active)
$\mathcal{L}_{margin}$ — without independently pushing `dec1` to become
more separable on its own account.

**Consequence for §3's gradient table below**: `dec1.detach()` here means
`boundary_head`'s weights receive gradient only from $\mathcal{L}_{boundary}$
(unchanged from the original claim), but the *shared trunk* now receives
**zero** direct gradient contribution from the boundary head's own loss
— only indirectly, via $\mathcal{L}_{margin}$'s use of $B_i$ (itself
computed from the now doubly-detached chain: `dec1` detached before
`boundary_head`, and $d$/$B_i$ detached again before use in
$\mathcal{L}_{margin}$'s weighting, per §3). This is intentional and
should be reflected in `PHASE_E11_IMPLEMENTATION_SPEC.md`'s gradient
table, updated alongside this correction.

**Attaches to `dec1`** (as input, now detached per the above), exactly
where the seg and evidential heads attach — a third head branching from
the same shared trunk output, not inserted earlier in the network.

**Supervision** (unchanged in form, only the input is now detached):
binary cross-entropy against ground-truth tumor/background
class, the same target the offline classifier was fit against in E1.3:

$$\mathcal{L}_{boundary} = \text{BCEWithLogits}(d_i, y_i), \quad y_i \in \{0, 1\}$$

where $y_i$ is the voxel's ground-truth class (available for every voxel
at training time via the segmentation mask — no new labels needed).
$d_i$ (the raw logit, pre-sigmoid) is used directly as the signed
boundary-distance analogue: same sign convention as before (positive
→ tumor-side, negative → background-side), and $|d_i|$ near zero means
the voxel sits near the boundary the head has learned, exactly
preserving the geometric interpretation E1.3–E7 established, but now a
live, differentiable, always-current quantity.

### 2. Total loss, updated from E8

$$\mathcal{L} = \mathcal{L}_{seg} + \mu \cdot \mathcal{L}_{boundary} + \lambda \cdot \hat U_i \cdot B_i \cdot \mathcal{L}_{margin}$$

$\mu$ is a new, small auxiliary-loss weight (standard practice for
auxiliary supervision heads, e.g. deep supervision literature already
surfaced in `PHASE_E2_LITERATURE_REVIEW.md`'s Area 4 results) — its
purpose is only to keep the boundary head's own classification accuracy
reasonable (so $B_i$ stays meaningful), not to be a primary training
signal competing with segmentation. **Starting value**: $\mu = 0.1$
(an order of magnitude below $\mathcal{L}_{seg}$'s implicit weight of 1,
analogous to how the frozen baseline's own `evidential_weight=0.5` sets
the evidential branch as a real but secondary signal) — not yet
validated by a sweep, flagged as an open parameter for E13.

$B_i = \exp(-|d_i|/\tau_b)$ is now computed directly from the boundary
head's live output each forward pass — no periodic refitting, no
staleness, always exactly synchronized with the current `dec1`
representation.

### 3. Gradient flow — an important design decision, made explicit

**$\mathcal{L}_{boundary}$'s gradient flows into `dec1` and the boundary
head's own weights $w, b$** (standard backprop through a supervised
auxiliary head) — this is what keeps the boundary head accurate and
current.

**$\mathcal{L}_{margin}$ uses $B_i$ (derived from $d_i$) only as a
*weight*, not as a term to backpropagate through into the boundary
head's own parameters.** Concretely: $B_i$ should be **detached**
(`.detach()` in PyTorch) before being used to scale $\mathcal{L}_{margin}$.

**Why this matters, explicitly justified (not left implicit)**: if $B_i$
were *not* detached, gradients from $\mathcal{L}_{margin}$ would flow
back through $B_i$ into $w, b$, meaning the margin loss could learn to
manipulate the boundary head's own decision function to make voxels
appear closer to (or farther from) the boundary — a route to gaming the
weighting mechanism rather than genuinely improving the underlying
representation geometry. Detaching $B_i$ ensures the boundary head is
supervised **only** by its own classification objective
($\mathcal{L}_{boundary}$), and the margin loss can only affect `dec1`
through its direct pull/push effect on $z_i$, not by indirectly warping
the yardstick used to weight it. This mirrors standard practice for
gating mechanisms in the literature found for E5 (e.g., DyCON's entropy
gate $H(p^s)$ is computed from the segmentation head's output but is not
differentiated through when used to scale the contrastive loss term —
confirmed as the standard, expected pattern for this kind of
multiplicative weighting).

### 4. Comparison to the offline approach

| | Offline (E1.3–E7) | Trainable head (E9) |
|---|---|---|
| Fit method | `sklearn.LogisticRegression`, refit periodically | `nn.Conv3d(32,1,1)`, trained every step via backprop |
| Staleness | Real risk (Failure Mode 6) — representation moves between refits | None — always current, updates every forward/backward pass |
| Compute cost | Periodic refit overhead (batched CPU sklearn call) + host-device transfer | Negligible — one extra 1×1×1 conv, same cost class as the existing `evidential_head` |
| Differentiability | Not differentiable — $B_i$ built from a non-differentiable external fit | Fully differentiable — but *deliberately* detached before use (§3) |
| Interpretability | Directly validated against E1.3's original diagnostic finding | Must be re-validated — is the learned $d_i$ still measuring the same geometric quantity? (see §5) |

### 5. Open validation question this creates

Replacing the offline classifier with a jointly-trained head changes
*what* is being measured, not just *how*. The offline classifier was fit
**after** training completed, observing a converged representation with
no incentive to shape that representation to be more separable — it was
a passive, diagnostic-only measurement. The in-network head, by
contrast, is trained **during** the same process it's meant to observe,
and (via $\mathcal{L}_{boundary}$) has its own small incentive to push
`dec1` toward better separability — a mild version of exactly the
same causal-influence concern E6/E7 investigated for the evidential head
itself. This is a new, not-yet-tested assumption, not a solved problem:
**E9's boundary head should be checked, once implemented, against E1.3's
AUC/R² benchmarks on a frozen checkpoint (baseline, no margin loss
active) to confirm it reproduces the same measurement the offline
classifier made**, before trusting it as $B_i$'s live source once the
margin loss is also active. Flagged as a required pilot-phase (E12)
sanity check, not assumed to hold.

## Files

| File | Purpose |
|---|---|
| `PHASE_E8_EGGO_V1_SIMPLIFIED.md` | The simplified design this head plugs into |
| `PHASE_E1_3_LATENT_GEOMETRY.md` | The offline method this replaces |
| `PHASE_E10_MARGIN_LOSS_SELECTION.md`, `PHASE_E11_IMPLEMENTATION_SPEC.md` | Next steps |

---

**Completed**: 2026-08-04
