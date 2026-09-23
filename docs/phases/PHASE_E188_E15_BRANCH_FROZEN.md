# E188 — E15 branch: FROZEN. Latent steering killed, readout confound proven.

**Date**: 2026-09-20
**Status**: CLOSED. Two inference-only experiments (A, F), no training, one frozen checkpoint.
Supersedes the E15 record for all forward-looking purposes.

## Headline

$$
\boxed{\text{E15's entire useful effect is mediated by the final readout row space.}}
$$

Across **40 subject × ε cells**: $D(z+\delta_E) \approx D(z+\delta_{\parallel W})$ with
$\max|D_E - D_\parallel| = 7.0\times10^{-5}$, while $D(z+\delta_{\perp W}) \approx D(z)$ with
$\max|D_\perp| = 5.2\times10^{-5}$ — **despite $\delta_\perp$ carrying 34–55% of the
perturbation energy**. Half the intervention, by norm, does nothing.

## Prerequisite correction to the record (found by reading the artifacts, not memory)

The original E15 result (+0.0393 at α=14, monotonic through α=28) was measured on the **retired
`E12f`/`UNet3D_v2` checkpoint**. On the current `E131_v5control_seed0`/`UNet3D_v5` it does not
reproduce: the effect is ~a third the size, peaks at **α=8 (+0.0139)**, and is **non-monotonic**
— declining by α=14 (+0.0035) and collapsing by α=28 (0.8796→0.8936→0.8832→0.8257). This was
already measured in `e172/smoke_oracle_e15.py` (`harness_valid: False`) but **never written up
or referenced in any phase doc or memory entry** — an orphaned result. E188 is calibrated to the
current model's real scale, not the historical one.

## Experiment A — local recoverability map

10 subjects × ε∈{4,8,12,14} × 32 per-voxel-unit random directions, norm-matched to $\delta_E$
(identical per-voxel and total tensor norm; only direction differs). Site `dec1`, single centered
128³ crop, binary fg = ET∪TC∪WT — all matched to E172's verified reproduction.

| ε | E15 ΔDice | E15 improves | headroom $R>0$ | E15 at 100th pct |
|---|---|---|---|---|
| 4 | +0.0094 | 9/10 | **0/10** | 9/10 |
| 8 | **+0.0111** | 7/10 | **0/10** | 9/10 |
| 12 | +0.0051 | 7/10 | **0/10** | 10/10 |
| 14 | −0.0024 | 6/10 | **0/10** | 10/10 |

Two findings:

1. **E15's direction is not arbitrary** — 100th percentile in 38/40 cells.
2. **There is no recoverable neighborhood** — the best of 32 random directions never beat native
   Dice in any of 40 cells. Random perturbation is purely destructive, degrading monotonically
   with ε (mean $R$: −0.0009 → −0.0077 → −0.1029 → −0.2230).

The "region of recoverable latent states" model is dead. There is no region; there is one
direction. This killed the original Experiment B ("what geometry makes a direction good?")
before it was run.

3. **The E15 effect is heterogeneous across subjects** — positive in 7/10 at the peak, negative
   in 3/10. No aggregate number in the prior record showed this.

## Experiment F — readout decomposition

`seg_head` is `Conv3d(32,3,k=1)` — **rank-3, not rank-1**. Measured: row cosines ET–TC +0.848,
ET–WT +0.671, TC–WT +0.774; singular values [5.134, 1.867, 1.176]; leading direction holds 84.4%
of energy, top two 95.6%. Approximately but not exactly rank-1 — so projecting onto a single $w$
would leak 15.6% of the readout's action into the "orthogonal" arm and invalidate the inference.
F therefore projects onto the **full 3D row space**, $P_W = W^{+}W$, giving $W\delta_\perp = 0$
exactly. Verified at runtime: projector idempotent to $7.5\times10^{-8}$, leak
$\|W\delta_\perp\|_{max} \approx 10^{-6}$ against a full-scale readout action.

Seven arms per cell, 280 forward passes. Results (mean over 10 subjects):

| ε | full | par_nat | perp_nat | par_renorm | perp_renorm | reversed |
|---|---|---|---|---|---|---|
| 4 | +0.0094 | +0.0094 | −0.0000 | +0.0107 | −0.0000 | −0.0461 |
| 8 | +0.0111 | +0.0111 | −0.0000 | +0.0087 | −0.0000 | −0.3726 |
| 12 | +0.0051 | +0.0051 | −0.0000 | −0.0122 | +0.0000 | −0.6123 |
| 14 | −0.0024 | −0.0024 | −0.0000 | −0.0241 | −0.0000 | −0.6946 |

**H1 fires.** Supporting detail:

- **`reversed` is catastrophic and monotonic** (−0.046 → −0.69): a signed logit push, not a
  geometric rearrangement. E15 is an axis-with-a-sign, and the sign is what GT supplies.
- **`par_renorm` reveals $\delta_\perp$'s actual role**: at ε=4 concentrating the whole budget in
  the readout-visible direction slightly *beats* full E15 (+0.0107 vs +0.0094), but by ε=12–14 it
  turns sharply negative (−0.0122, −0.0241) while full E15 stays near zero. The orthogonal
  component is **ballast** — it absorbs displacement that would otherwise overshoot the logit.
  This mechanically explains the non-monotonic peak at ε=8: that is where the logit push
  saturates, not where latent structure is optimally exploited.

## Ledger

| Item | Verdict |
|---|---|
| E15 as a method | **KILL** |
| E15 as latent-geometry discovery | **KILL** |
| E15 as evidence of hidden decoder capacity | **KILL** (confounded by readout) |
| E15 as a readout-steering diagnostic | PASS (it is exactly that) |
| **F as a reusable exclusion test** | **KEEP** — see below |

This also retroactively explains E172's label-free null cleanly: the selectors failed because
there was nothing to select beyond "which way moves the logit," and that sign is precisely what
GT was supplying. E172 becomes a consistent corollary, not an independent puzzle.

## The durable output: Gate F, a reusable inclusion criterion

### The structural fact E188-F establishes

$$
\boxed{\text{At dec1, the behavioral quotient of } z \text{ is } Wz.}
\qquad
\Delta z_\perp \in \ker W \Rightarrow \Delta\text{logits}=0 \Rightarrow \Delta\text{Dice}=0
$$

Everything in $\ker W$ is invisible to the prediction. Verified, not assumed: 40/40 cells,
$\max|\Delta D_\perp| = 5.2\times10^{-5}$ with 34–55% of the perturbation energy living there.

### Gate F, stated correctly

An earlier draft of this gate said "any candidate's benefit must not reduce to movement in the
readout row space." **That is too strong and is superseded.** Every prediction necessarily passes
through $W$; an upstream method may legitimately act through nonlinear decoder processing and
ultimately produce a useful $W\Delta z$. Requiring a large $\ker W$ component would rule out
valid mechanisms.

The correct requirement is **causal, not geometric**. For a candidate intervention $I_l$ at
upstream representation $h_l$, propagate $h_l + \delta_l$ through the frozen decoder to obtain
$\Delta z$, decompose $\Delta z = P_W\Delta z + (I-P_W)\Delta z$, and require:

$$
\boxed{\Delta D(\delta_l) \;\not\approx\; \Delta D\big(P_W \Delta z\big)}
$$

In words: **if an upstream intervention's benefit is fully reproduced by simply pushing `dec1`
along the terminal readout component it induces, the upstream location is doing no work and it
is not a NeuroScan mechanism.**

| Pattern | Verdict |
|---|---|
| $\delta_l \to \Delta z \to P_W\Delta z \to$ better logits, with no evidence the upstream site mattered | **KILLED** — terminal readout steering in disguise |
| $\delta_l \to$ nonlinear decoder transformation $\to \Delta z \to W\Delta z \to$ better segmentation, where the *useful logit configuration depends on the upstream computation* | potentially interesting |

### Standing intervention rule (order matters)

1. Does it improve Dice?
2. Does the effect survive the Gate-F readout-quotient screen?
3. Is the mechanism distinguishable from steering the terminal readout?
4. **Only then** investigate label-free selection and generalization.

Gate F costs ~280 forward passes and no training. It is a cheap architectural gate, not a
research branch of its own — which is the point.

## Why no reachability experiment (E189) was built

A reachability screen — comparing $\mathcal{R}_l = \{W\Delta z(\delta_l)\}$ against
$\mathcal{R}_{\text{dec1}}$ — was considered and **declined**. Two reasons:

1. **The originally-proposed form was vacuous.** Matching a control on $W\Delta z$ (3 numbers per
   voxel) while $\Delta z$ has 32 leaves 29 dimensions unconstrained — but at `dec1` any two
   $\Delta z$ agreeing on $W\Delta z$ produce *identical* logits and hence identical Dice, since
   `seg_head` is the last operation. "Same logit displacement, different Dice" is impossible
   there, not discoverable.
2. **A positive result would not yield a method.** Establishing $\mathcal{R}_l \supsetneq
   \mathcal{R}_{\text{dec1}}$ only raises the next question — which upstream perturbation reaches
   a *useful* member of the expanded set without GT — and E172 already showed label-free
   direction selection fails at this site. That path is the diagnostic treadmill this project is
   explicitly trying to leave.

Additionally, "do upstream representations carry task structure `dec1` lacks" is **already
answered** by E169b/E174 ($R^2 = 0.518$ at enc3 vs $0.004$ at dec1, both size controls). The
open question is narrower: *is that upstream structure an actionable computational degree of
freedom?* — and Gate F is the instrument for testing any candidate answer, so a separate
instrument was not built.

## Structural map of the architecture, as now verified

```text
                    upstream representation
                           │
                    [task structure: R^2=0.518 at enc3, E169b]
                           │
                    nonlinear decoder
                           │
                           ▼
                         dec1
                           │
                    ┌──────┴──────┐
                    │             │
                 row(W)         ker(W)
                    │             │
                    ▼             X   <-- E188-F: behaviorally dead,
                 logits               34-55% of energy, zero effect
                    │
                    ▼
                  Dice
```

The right-hand branch is experimentally dead at `dec1`. Manipulating arbitrary latent geometry
at the terminal representation is therefore closed as a source of novelty. If interesting
territory remains, it is upstream computation that changes what the nonlinear decoder can
*construct* before the terminal quotient is applied — and per this project's own
phenomenon→mechanism→prior-art→method rule, that requires a **prior-art audit before any code**,
specifically on: *upstream representation → nonlinear decoder → altered reachable output
configuration, where the intervention is not reducible to terminal activation/readout steering.*
If that audit shows no differentiated principle, E188-F has done its job: this entire
latent-steering branch is the wrong place for the remaining budget.

## Orphaned artifacts flagged (not reconstructable)

`E151_dec3_sweep.json`, `E152_resolution_recovery.json`, `E153_responsibility.json` contain
results with **no surviving generating script and no phase doc**. `E151_dec3_sweep.json` appears
to be the geometry-vs-Jacobian comparison (`d_G`/`d_J`/`d_R`/`d_perp`) that
`PHASE_E15_PRIOR_ART_AUDIT.md` explicitly **retired before compute** ("no compute is
authorized") — so it was evidently run anyway, with no record of the conclusion. The exact
definitions of `d_J`, `align`, `perp_norm`, `jnorm` cannot be recovered from what is in the repo.
**Do not cite these files.** Marked orphaned rather than reverse-engineered.
