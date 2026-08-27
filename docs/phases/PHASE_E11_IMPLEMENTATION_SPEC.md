# Phase E11: EGGO-v1 Implementation Specification

**Status**: ✅ Complete — mathematics and mechanics fully specified, no code written yet

**Date**: 2026-08-04

## Purpose

Answers the concrete engineering questions before any PyTorch is
written: which tensors require gradients, where the margin is computed,
which branch receives gradients, whether inference changes, and what
happens under evidential-head failure modes. This is the last
specification document before Phase E12's pilot training run.

## 1. Full forward pass, with tensor shapes

Building directly on `neuroscan_3d_fixed.py`'s existing structure
(frozen, unmodified — `UNet3D.forward`), with two additions: the
boundary head (E9) and the margin-loss computation (E10).

```
x: (B, 1, D, H, W)                          [input, B=8, D=H=W=64]

... existing frozen encoder/decoder path, unchanged ...

dec1: (B, 32, D, H, W)                      [shared trunk output, unchanged]

probs = seg_head(dec1)          : (B, 1, D, H, W)     [existing, unchanged]
alpha, beta = evidential_head(dec1)  : (B, 1, D, H, W) each  [existing, unchanged]
d = boundary_head(dec1)         : (B, 1, D, H, W)      [NEW, E9]
```

`boundary_head` is `nn.Conv3d(32, 1, kernel_size=1)`, added to
`UNet3D.__init__` alongside the existing `seg_head`/`evidential_head`
definitions — this is the one point where the frozen architecture file
must be touched. Per `baseline_frozen_milestone`, this requires an
explicit exception discussion and likely re-tagging (e.g.
`baseline-frozen-v2`) before implementation, since it adds a parameter to
the shared model class, not just external training-loop code (unlike
ABO, which never touched `neuroscan_3d_fixed.py` at all).

## 2. Per-training-step computation, in order

1. Forward pass through the (now four-headed) network on the full batch,
   producing `probs`, `alpha`, `beta`, `d` for every voxel.
2. Compute $\mathcal{L}_{seg}$ from `probs` against the ground-truth mask
   — **unchanged from the frozen baseline**.
3. Compute $\mathcal{L}_{boundary}$ = BCEWithLogits(`d`, ground-truth
   class) over **every voxel** (this term is cheap — no pairwise
   computation, no subsampling needed, same cost class as
   $\mathcal{L}_{seg}$ itself).
4. **Subsample anchor voxels** for the margin loss: per volume, sample
   2,000 voxels (matching E1.4's diagnostic rate — see
   `PHASE_E5_ALGORITHM_DESIGN.md` §5.2), **stratified toward high
   uncertainty** (top-k by $\hat U_i$, not uniform random — per the same
   mitigation, since only ~0.14% of voxels are actually incorrect and
   uniform sampling would rarely include the voxels that matter). This
   gives $8 \times 2{,}000 = 16{,}000$ anchors per batch.
5. Compute $\hat U_i$ (from `alpha`, `beta` at the sampled voxels) and
   $B_i = \exp(-|d_i|/\tau_b)$ (from `d` at the sampled voxels, **with
   `d.detach()` applied before computing $B_i$** — per E9 §3, this is
   the load-bearing detach that prevents the margin loss from gaming the
   boundary head).
6. **Cap the negative set per anchor**: for each anchor voxel, sample at
   most 50 same-batch opposite-class voxels as $N(i)$ (a new, explicit
   mitigation this document adds — see §5 for why the 16,000-anchor
   reduction from `PHASE_E5_ALGORITHM_DESIGN.md` §5.2 is, on its own,
   still insufficient: full pairwise cost within a 16,000-voxel anchor
   set is ~8×10⁹ operations, which the 50-negative cap reduces to
   ~2.6×10⁷ — a further ~300× reduction, needed on top of the
   volume-level subsampling already specified).
7. Compute $\mathcal{L}_{margin}$ (E10's formula) over the capped
   anchor/negative pairs.
8. Total loss: $\mathcal{L} = \mathcal{L}_{seg} + \mu \mathcal{L}_{boundary} + \lambda \mathcal{L}_{margin}$
   (note: $\hat U_i \cdot B_i$ is already inside $\mathcal{L}_{margin}$'s
   per-voxel sum, per E10's formula — not a separate top-level
   multiplier).
9. Single `.backward()` call, single `optimizer.step()` — same pattern
   as the frozen baseline (and unlike ABO, which required two independent
   `backward()`/`autograd.grad()` calls; EGGO-v1's three loss terms are
   simply summed before one backward pass, since none of them require
   separately-scaled gradient assembly the way ABO's trunk-only
   application did).

## 3. Which tensors require gradients, and which branch receives what

**Updated per E9's supervision-design correction** (see
`PHASE_E9_TRAINABLE_BOUNDARY_HEAD.md` §1 — `boundary_head` now reads
`dec1.detach()` as input, not raw `dec1`, to avoid becoming an
uncontrolled second segmentation head):

| Tensor | requires_grad | Receives gradient from |
|---|---|---|
| `dec1` (shared trunk) | Yes | $\mathcal{L}_{seg}$ (existing) + $\mathcal{L}_{margin}$ (new, via $z_i$'s direct appearance in the pairwise distance term). **NOT** $\mathcal{L}_{boundary}$ — that gradient is blocked at the `dec1.detach()` boundary before entering `boundary_head` |
| `boundary_head` weights ($w, b$) | Yes | $\mathcal{L}_{boundary}$ **only** — never from $\mathcal{L}_{margin}$, because $B_i$ is separately detached again before use in $\mathcal{L}_{margin}$'s weighting (E9 §3) — i.e. two independent detach points on this path, not one |
| `seg_head`, `evidential_head` weights | Yes | Unchanged from frozen baseline — $\mathcal{L}_{margin}$ does not touch either head, only the shared trunk |
| $\hat U_i$ (derived from `alpha`, `beta`) | **Should also be detached** | This was not explicit in E9's original spec — added here: exactly the same reasoning as $B_i$'s detach applies. If $\hat U_i$ were not detached, $\mathcal{L}_{margin}$ could gradient-hack the evidential head into reporting artificially low uncertainty to reduce its own weight, rather than genuinely improving geometry. **Both gate inputs ($\hat U_i$ and $B_i$) must be detached before use in $\mathcal{L}_{margin}$'s weighting** |

**Net effect (revised)**: the shared trunk's only classification pressure
is $\mathcal{L}_{seg}$ (via `seg_head`, unchanged from the frozen
baseline) plus whatever geometric pull/push $\mathcal{L}_{margin}$
applies directly. `boundary_head` is a genuine, checkable measurement
device — trained to accurately read `dec1`'s existing structure — but
does **not** itself add new pressure shaping that structure. This is a
materially different (and better-justified) design than the original E9
draft, which would have let `boundary_head`'s own BCE loss directly
reshape `dec1` as an uncontrolled second segmentation objective.

**Original (superseded) net-effect claim, for the record**: the shared trunk (`dec1`-producing layers) is the only
part of the network that receives gradient signal from all three loss
terms simultaneously. Both auxiliary heads (`boundary_head`,
`evidential_head`) are supervised only by their own dedicated losses,
never by $\mathcal{L}_{margin}$ — this is the key design property that
keeps the three objectives from fighting each other through
gradient-hacking side channels.

## 4. Does inference change?

**No.** At inference time, only `probs` (from `seg_head`) is used for
the actual segmentation output — exactly as in the frozen baseline. The
evidential head's `alpha`/`beta` are already computed at inference for
uncertainty reporting (existing behavior, unchanged). The new
`boundary_head`'s output `d` is **not needed at inference** — it exists
purely to supply $B_i$ during training. It can be left computed (cheap,
one extra 1×1×1 conv) or explicitly skipped at inference via a flag, a
minor implementation detail deferred to actual coding, not a design
question. **No change to the model's deployed behavior or output
format.**

## 5. Complexity and memory overhead (concrete numbers, not estimates)

| Quantity | Value |
|---|---|
| Voxels per volume | $64^3 = 262{,}144$ |
| Anchors sampled per volume | 2,000 (matching E1.4's diagnostic rate) |
| Anchors per batch (batch_size=8) | 16,000 |
| Max negatives sampled per anchor | 50 (new mitigation, this document) |
| Naive full-pairwise cost within the 16,000-anchor set | $16{,}000^2 \times 32 \approx 8.19 \times 10^9$ ops |
| Cost with the 50-negative-per-anchor cap | $16{,}000 \times 50 \times 32 \approx 2.56 \times 10^7$ ops |
| Reduction from capping | **~320×** |

**This 50-negative cap is a new, explicit addition this document makes**
— `PHASE_E5_ALGORITHM_DESIGN.md` §5.2 specified volume-level subsampling
(2,000/volume) and in-batch-only sampling, but did not specify a per-anchor
negative cap; working through the concrete arithmetic here shows that
mitigation alone is insufficient (still ~8×10⁹ ops/batch), and the
additional cap is required to reach a tractable number. Flagged
explicitly since this is a load-bearing addition to the design, not a
minor implementation detail — without it, EGGO-v1 would still be
computationally impractical.

**Memory**: boundary head adds one $32 \times 1$ conv (negligible
parameter count, ~33 parameters). The main new memory cost is holding the
16,000 sampled anchor embeddings ($16{,}000 \times 32$ floats,
~2MB) plus their capped negative-pair distances during the margin loss
computation — small relative to the model's own activation memory.
**Expect overhead in the same broad category as ABO's own verified
overhead** (ABO: 5.3GB peak / ~194s per epoch vs. baseline's 4.4GB /
~140-160s) since both involve an extra per-batch geometric computation
on top of the standard forward/backward pass — **not yet measured**,
must be verified with a smoke test (§7) before the pilot run.

## 6. What happens if the evidential head is wrong (a specific failure scenario)

If the evidential head reports **spuriously high evidence** (falsely
confident) on a voxel that is actually near the latent boundary and/or
incorrect: $\hat U_i \to 0$ for that voxel, $\mathcal{L}_{margin}$'s
weight for it $\to 0$ regardless of $B_i$ — **the margin loss simply
does not act on that voxel**. This is a silent failure mode, not a
crash: EGGO-v1 would fail to correct exactly the voxels where the gate
itself is miscalibrated, but would not produce corrupted gradients or
training instability from this case — the risk is *under-correction*,
not active harm. This is consistent with Failure Mode 1 (gate
degeneracy) already flagged in `PHASE_E5_ALGORITHM_DESIGN.md` §6 and
should be monitored in the pilot run (E12) via the same
`pct_batches_damped`-style diagnostic logging pattern ABO used — logging
what fraction of sampled anchors receive non-negligible margin-loss
weight each epoch, to catch gate degeneracy empirically rather than
assuming it away.

## 7. Required smoke test before the pilot run (E12)

Per the same discipline as ABO's `verify_refactor.py` (refactor verified
before the active-mode launch): before E12's full pilot, run 1 epoch
with EGGO-v1 active and confirm:
1. No crash, no NaN/Inf losses.
2. Peak GPU memory and epoch time are in a sane range (not 5-10× the
   baseline, matching the "similar category to ABO" expectation from §5,
   not wildly beyond it).
3. The boundary head, evaluated with $\mathcal{L}_{margin}$ and
   $\mathcal{L}_{boundary}$ both active, reproduces AUC/R² values broadly
   consistent with E1.3's offline-classifier findings on the same
   checkpoint — the open validation question E9 §5 flagged, now given a
   concrete, required check rather than left as a vague caveat.

## Files

| File | Purpose |
|---|---|
| `PHASE_E8_EGGO_V1_SIMPLIFIED.md`, `PHASE_E9_TRAINABLE_BOUNDARY_HEAD.md`, `PHASE_E10_MARGIN_LOSS_SELECTION.md` | The three specs this implementation plan integrates |
| `baseline_frozen_milestone` (memory) | The constraint requiring explicit discussion before touching `neuroscan_3d_fixed.py` |

---

**Completed**: 2026-08-04 — ready for Phase E12 (pilot training), pending review
