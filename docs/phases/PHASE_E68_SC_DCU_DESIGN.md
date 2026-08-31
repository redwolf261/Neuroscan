# Phase E68 — SC-DCU Design: Self-Supervised Correspondence-Verified Deformable Skip Correction

## Status: DESIGN ONLY. No code written, no training run. Per project discipline
(design before implementation), this document is the complete mathematical
specification, reviewed for correctness and novelty before any implementation
begins.

## 1. What this targets, precisely

- **E65** (causal, empirical, this project): a 3-voxel roll of `enc1` — preserving
  every local neighborhood and the full value distribution exactly, changing
  only spatial correspondence — produces a Dice drop of **0.214**, roughly 8×
  larger than local rearrangement (0.027) or smoothing (0.028), and is
  size-specific (ρ=−0.553, smaller lesions hurt more).
- **Dynamic U-Net's DCU** (arXiv:2403.07303, 2024): the one existing published
  mechanism that directly targets this class of problem — a learned deformable
  offset realigning the upsampled decoder feature to the skip-connected encoder
  feature before concatenation:
  $$
  \Delta_i = \mathcal{F}_{\text{conv}}([F_i, S_i]), \qquad
  F_i^* = \mathcal{F}_{\text{LReLU}}\big(\mathcal{F}_{\text{deform}}(F_i; \Delta_i)\big)
  $$
  with **no supervision of $\Delta_i$ itself** — it receives gradient only
  indirectly, through the final segmentation loss backpropagating through the
  deformable sampling operation. Verified (fetched primary source, not assumed)
  to have no offset-fidelity term, no boundedness constraint, and no
  scale-conditioning.
- **The one prior fix for offset-instability in deformable alignment**
  (Chan et al., offset-fidelity loss, video super-resolution) supervises the
  offset against **optical flow** — a ground-truth correspondence signal that
  exists between two video frames but has **no analogue** between `enc1` and
  `upconv1` in a single-volume segmentation forward pass (they are not two
  images of the same scene at different times; they are the same volume at two
  processing stages of one forward pass).

**The gap**: no existing deformable-skip-correction module for segmentation has
ever been given a way to verify that its learned offset actually predicts
spatial correspondence, because no admissible ground-truth correspondence
signal existed for this setting. E65 provides one for free: a controlled
synthetic translation applied during training IS a ground-truth correspondence
signal, with a known answer, requiring no extra annotation.

## 2. The mechanism: SC-DCU (Self-supervised, Correspondence-verified DCU)

### 2.1 Base module (adapted from DCU to 3D, one variable changed at a time)

For the `enc1` skip only (matching E62–E65's own scope — the highest-resolution,
most lesion-relevant skip; NOT touching `enc2`/`enc3`'s skips, isolating this to
exactly the connection this project's evidence concerns):

$$
\Delta = \mathcal{F}_{\text{off}}\big([\,U(D_2),\ E_1\,]\big) \in \mathbb{R}^{3 \times D \times H \times W}
$$

where $U(D_2) = \texttt{upconv1}(\texttt{dec2})$ (the real decoder path,
unchanged), $E_1 = \texttt{enc1}$ (the real skip tensor, unchanged),
$\mathcal{F}_{\text{off}}$ is a single $3\times3\times3$ Conv3d producing a
3-vector offset field (one $(\delta_x,\delta_y,\delta_z)$ per voxel, the 3D
extension of DCU's own $\Delta_i$ — DCU's original is 2D with a 2-vector
offset per position; this is the direct, minimal 3D generalization, not a
redesign).

The skip feature is then resampled at the corrected locations via trilinear
deformable sampling (3D analogue of DCU's modulated deformable convolution;
implemented as a grid-sample-based deformable resampling, since PyTorch has no
native 3D deformable-conv op — this is an implementation detail, not a change
to the mathematical mechanism):

$$
E_1^* = \mathcal{F}_{\text{resample}}(E_1;\ \Delta)
$$

$E_1^*$ replaces $E_1$ in the existing `cat1 = torch.cat([upconv1, E_1], dim=1)`
concatenation — everything downstream of that point (dec1, seg_head,
evidential_head) is **completely unchanged**.

### 2.2 The actual contribution: synthetic-translation offset-fidelity loss

**Training-time only**, once per training step, independent of the real forward
pass's own randomness:

1. Draw a random integer voxel shift $t = (t_x, t_y, t_z)$, each component
   drawn uniformly from $\{-k, \dots, k\}\setminus\{0\}$ for a small fixed $k$
   (e.g. $k=4$, chosen to bracket E65's own tested offset of 3 — not tuned
   post-hoc against any result).
2. Construct $E_1^{\text{shifted}} = \text{roll}(E_1, t)$ — EXACTLY E65's own
   translation intervention (same construction, same guarantee: every local
   neighborhood and the full value distribution preserved exactly, only
   address changes).
3. Run $\mathcal{F}_{\text{off}}$ on the shifted pair:
   $$
   \Delta^{\text{shifted}} = \mathcal{F}_{\text{off}}\big([\,U(D_2),\ E_1^{\text{shifted}}\,]\big)
   $$
4. **Offset-fidelity loss** (the actual new term, this project's own
   contribution — not present in DCU or in Chan et al.'s video-domain version):
   $$
   L_{\text{off}} = \frac{1}{|\Omega|}\sum_{p \in \Omega} \big\| \Delta^{\text{shifted}}_p - (-t) \big\|_2^2
   $$
   where $\Omega$ ranges over all voxel positions and $-t$ is the target
   (the offset that would exactly undo the known applied shift $t$, since the
   module's job is to realign $E_1^{\text{shifted}}$ back toward correspondence
   with $U(D_2)$, which was never shifted). This is a direct, closed-form,
   free ground-truth signal — no optical flow, no extra annotation, no
   approximation.

Total training loss:
$$
L_{\text{total}} = L_{\text{seg}} + \mu \cdot L_{\text{boundary}} + \lambda_{\text{off}} \cdot L_{\text{off}}
$$

added as one new term alongside the existing composition, matching the
project's "one variable at a time" discipline — $\lambda_{\text{off}}$ is a new
hyperparameter requiring its own calibration pass (not guessed), same
convention as `mu`/`lambda_margin`'s own calibration history in this project.

### 2.3 Why this specifically closes the identified gap

- DCU's offset has never been checked to do what it claims. $L_{\text{off}}$
  makes this checkable and enforceable: **at evaluation time**, one can measure
  $\|\Delta^{\text{shifted}} - (-t)\|$ on held-out synthetic shifts as a direct,
  interpretable diagnostic of whether the module is actually learning
  correspondence correction — independent of whether Dice improves. This
  directly operationalizes the project's own constraint #10 ("mechanism ≠
  performance": measure both separately).
- It requires zero new annotation (synthetic, closed-form target).
- It is derived directly from this project's own causal evidence (E65's exact
  intervention), not from intuition or generic regularization.

## 3. What this does NOT change (isolation discipline)

- No change to `enc2`/`enc3` skip connections.
- No change to the loss weights already established (`mu`, `lambda_margin`
  remain as they are; this is an ADDITIVE term).
- No change to the checkpoint/training recipe otherwise (optimizer, LR
  schedule, seed convention, batch size — all identical to the project's
  established recipe).
- No claim about `heads`/`boundary_head` — unaffected, still reads
  `dec1.detach()` as before.
- No architecture change beyond replacing `enc1`'s direct concatenation with
  the SC-DCU-corrected `E_1^*` — a single, localized, isolable modification.

## 4. Pre-declared falsifiable predictions (before any training)

1. **Offset-fidelity check** (cheapest, run first, no full training needed):
   at random initialization, $\mathcal{F}_{\text{off}}$'s output should NOT
   correlate with $-t$ (no prior reason it would). After even a short
   training run with $L_{\text{off}}$ active, held-out synthetic-shift
   recovery error should decrease substantially and become significantly
   better than a shuffled-label control. **If this fails, KILL before any
   further training** — the module isn't learning correspondence at all,
   and nothing downstream can be trusted.
2. **Mechanism check**: if $L_{\text{off}}$ succeeds (prediction 1 holds),
   test whether the *causal* effect from E65 shrinks specifically on the
   corrected checkpoint — repeat E65's own translation intervention (now on
   $E_1^*$ instead of raw $E_1$) and check whether $\Delta$Dice$_{\text{translation}}$
   is significantly smaller than the original 0.214, using the same
   statistical discipline (paired test, permutation test, bootstrap CI).
   **This is the test of whether the mechanism does what it claims**,
   independent of final Dice.
3. **Performance check** (the actual project bar): full 125-subject
   validation Dice, ≥3 seeds, compared against the exact matched baseline
   (v3/D4-only, same recipe, no SC-DCU) — needs ≥1pp mean improvement with a
   CI that excludes 0, per the project's own multi-seed confirmation
   standard (established after the CCABA variance-correction finding).

Only if prediction 1 AND 2 hold does prediction 3's result mean what it
appears to mean (a real mechanism causing the Dice change, not an unrelated
side effect of adding parameters/capacity). If prediction 1 or 2 fails but
prediction 3 somehow still shows +1pp, that must be reported honestly as an
unexplained performance gain, not attributed to the claimed mechanism
(project constraint #10).

## 5. Novelty statement (honest, per E67/E67b's search)

**Not novel**: deformable skip-connection realignment as a mechanism (DCU,
2024). **Not novel**: offset-fidelity supervision as a general principle
(Chan et al., video domain). **Believed novel** (not found in the E67/E67b
search, 8 queries across 4 rounds): the specific combination of (a) a
synthetic-translation self-supervision signal for offset-fidelity in a
single-volume (non-video) encoder-decoder segmentation setting, where no
optical-flow-like ground truth exists, and (b) deriving the choice, scale,
and target of that synthetic perturbation directly from a pre-registered
causal audit (E62→E65) rather than from intuition. This is a real,
checkable, falsifiable claim, but a narrower one than "new architecture" —
the honest framing throughout should be: **a verifiable, causally-motivated
correction to a known gap in an existing mechanism**, not a new mechanism.

## 6. Open implementation questions (to resolve before writing code)

1. 3D deformable resampling has no native PyTorch op — needs a
   `grid_sample`-based implementation; must be verified bit-for-bit
   correct (identity offset → identity resampling) before any training,
   matching this project's own sanity-check discipline (E47/E48/E58/E62–E65).
2. $\lambda_{\text{off}}$ and $k$ (shift range) need a calibration pass,
   not a guess — small pilot runs, not full 30-epoch training, per the
   project's established calibration convention (E25b's own lambda_ds
   calibration is the template).
3. Whether $\mathcal{F}_{\text{off}}$ should be conditioned on a lesion-scale
   signal (E67's "unclaimed calibration" angle) is a SEPARATE, later
   question — not conflated with this phase's already-scoped contribution,
   per the "one variable at a time" rule. If SC-DCU's base form (this
   document) clears its own falsifiable checks, size-conditioning becomes
   phase E69, not folded in here.

## Next step (not yet started)

Implement and run prediction 1 (offset-fidelity check) ONLY — the cheapest,
first, most falsifiable test. Do not implement prediction 2 or 3's
infrastructure until prediction 1 passes.
