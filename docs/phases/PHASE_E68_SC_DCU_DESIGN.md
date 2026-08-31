# Phase E68 — SC-DCU Design: Self-Supervised Correspondence-Verified Deformable Skip Correction

## Status: SUPERSEDED novelty claim; design retained as the substrate for a
sharper, corrected hypothesis (H68-B, Section 7 below). A follow-up targeted
novelty check found this phase's original novelty claim (Sections 1–5 below)
too strong, and identified a real conceptual gap in the original design
(Section 6 below) that the original document did not address. **Do not cite
Sections 1–5's novelty claim; it is retracted.** See Section 7 for the
corrected direction. No code has been written and no training has been run at
any point in this phase — the correction below happened entirely at the
design stage, which is what that stage is for.

## Correction record (read this first)

A follow-up targeted novelty search (not run by this session — supplied
directly, then independently checked against the cited sources) found:

1. **Synthetic-deformation-as-ground-truth is a standard technique in medical
   image registration** (e.g. Eppenhof et al., training on synthetically
   deformed image/segmentation pairs with the deformation field known exactly).
   The original Section 1 claim ("no admissible ground-truth correspondence
   signal existed for this setting") is **too strong** — an admissible
   synthetic signal has existed in the registration literature for years; it
   simply had not (as far as either search found) been applied to a DCU-style
   skip-connection offset specifically.
2. **Transformation-consistency regularization is well-established** in
   semi-supervised medical segmentation (Bortsova et al.) and image-to-image
   learning generally. The general pattern "apply a known transformation,
   supervise the network to respect/recover it" is not new.
3. **Chan et al.'s offset-fidelity loss is conceptually closer than the
   original document credited** — its contribution is the general principle
   of constraining a learned deformable offset toward *any* correspondence
   field, not specifically "uses optical flow." Swapping the target from
   optical flow to a synthetic translation is a **domain adaptation of the
   supervision signal**, not a new loss-computation primitive.
4. **A deeper, more important problem**: the original design asserted
   $\Delta^{\text{shifted}} \approx -t$ as the supervision target without
   justifying why the network's *task-optimal* offset should equal the
   *externally-imposed geometric* shift. These are not obviously the same
   quantity — the deformable module estimates whatever sampling displacement
   minimizes the downstream objective, not necessarily the literal geometric
   transformation applied to the input, especially given `roll`'s wraparound
   artifacts, feature ambiguity, and the fact that encoder and decoder
   representations are not guaranteed to share the same coordinate/semantic
   structure. Forcing equality by construction (as the original $L_{\text{off}}$
   does) could suppress a real, useful distinction rather than reveal one.

**Corrected novelty status**:

| Component | Novel? |
|---|---|
| Deformable skip alignment (DCU) | No |
| Offset-fidelity-style loss (any correspondence target) | No |
| Synthetic-transformation-as-ground-truth | No (standard in registration) |
| Transformation-consistency supervision | No (standard in semi-supervised segmentation) |
| DCU + synthetic translation specifically, in 3D segmentation | Possibly unreported, but this alone is an *application*, not a *mechanism* claim |
| Whether task-optimal deformable offset equals the externally-imposed geometric transformation, and what it means when it doesn't | **Open** — this is the actual question worth pursuing (H68-B, Section 7) |

The honest reframing: Sections 1–6 (original design) are retained below as the
*substrate* — the offset module and the synthetic-shift construction are still
useful — but $L_{\text{off}}$ forcing $\Delta \to -t$ is no longer proposed as
the contribution. Section 7 replaces it.

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

### 2.2 SUPERSEDED: the original hard-equality offset-fidelity loss

The original design forced $\Delta^{\text{shifted}} \to -t$ by direct L2
penalty (a hard-equality offset-fidelity loss, in the style of Chan et al.'s
video-domain version, retargeted to a synthetic shift). **This is retracted**
as the phase's contribution — both because the underlying supervision pattern
is not novel (Section "Correction record" above) and, more importantly,
because nothing justified assuming $-t$ IS the correct target for the
network's task-optimal offset (Section "Correction record," point 4). Forcing
equality by construction would have measured nothing — it manufactures
agreement rather than testing for it. See Section 7 for the replacement.

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

## 4. SUPERSEDED pre-declared predictions

The original three predictions (offset-fidelity recovery, causal-effect
shrinkage, Dice bar) assumed the hard-equality target from Section 2.2 and are
superseded by Section 7's H68-A/B/C. Retained here only for the historical
record: predictions 2 and 3's underlying logic (test the mechanism
independently of Dice; require ≥1pp with multi-seed CI before any performance
claim) carries forward unchanged into Section 7 — only prediction 1's
"recovery should converge to $-t$" framing is replaced.

## 5. SUPERSEDED novelty statement

The original novelty statement (claiming the synthetic-translation
self-supervision signal itself as likely novel) is retracted — see the
"Correction record" section at the top of this document. Section 7 states the
corrected, narrower novelty target.

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

## 7. CORRECTED direction: does the geometric target even match the task-optimal offset?

This replaces Sections 2.2, 4, and 5's original content as the phase's actual
contribution.

### 7.1 Two distinct notions of displacement

**Geometric displacement** — known exactly by construction, no learning
involved:
$$
\Delta^{G} = -t
$$
the offset that would exactly undo a synthetic shift $t$ applied to $E_1$.

**Task-optimal displacement** — whatever the deformable module actually
converges to when trained on the real segmentation objective alone (no
$L_{\text{off}}$ term at all):
$$
\Delta^{T} = \arg\min_{\Delta}\ L_{\text{seg}}\Big(\text{decoder}\big(\text{Align}(E_1^{\text{shifted}}, \Delta)\big),\ Y\Big)
$$

There is no a priori reason $\Delta^G = \Delta^T$. `roll`'s wraparound,
feature ambiguity, and the fact that `enc1` and `upconv1` are not guaranteed
to encode the same coordinate/semantic structure could all make the
task-optimal correction systematically different from the literal geometric
inverse.

### 7.2 Pre-declared hypotheses (mutually exclusive outcomes, not a single GO/KILL)

**H68-A (geometric-supervision hypothesis, LOW novelty per correction
record)**: $\Delta^G \approx \Delta^T$ — the task-optimal offset converges to
(or close to) the known geometric inverse. If true, hard-equality supervision
(the original Section 2.2 design) is a reasonable, if unoriginal, engineering
choice — an application/adaptation contribution at most, matching DCU +
synthetic-shift supervision, nothing more.

**H68-B (feature-correspondence discrepancy hypothesis, the interesting one)**:
$\Delta^G \neq \Delta^T$ **systematically** (not just noisily) — the
task-optimal offset reliably diverges from the geometric inverse in a
structured way (e.g. correlated with lesion size, decoder depth, or local
feature ambiguity). This would mean the network's real correspondence need is
not "undo my synthetic corruption," and the discrepancy field itself,
$$
\Delta^R = \Delta^T - \Delta^G,
$$
becomes the object of interest — is $\Delta^R$ predictable from lesion size
(continuing the E48/E65 size-specificity thread), from local feature
ambiguity, or from something else entirely?

**H68-C (no exploitable structure)**: $\Delta^T$ is real (nonzero, the module
does learn to move things) but its deviation from $\Delta^G$ is unstructured
noise, uncorrelated with anything measurable. This would mean neither H68-A
nor H68-B holds, and the whole direction should be reported as a clean null,
not rescued.

### 7.3 The discriminating experiment (cheap, no full training, matches project
discipline)

1. Train a SMALL number of steps (pilot scale, not full 30-epoch) with
   **no** $L_{\text{off}}$ term at all — pure `L_seg` on the SC-DCU-augmented
   architecture (Section 2.1's base module only), so $\Delta$ is free to
   converge to whatever the task actually needs.
2. At evaluation, for held-out subjects, apply a range of KNOWN synthetic
   shifts $t$ (matching E65's own tested range) to `enc1`, and record the
   module's own $\Delta^{\text{shifted}}$ (never forced toward anything).
3. Compare $\Delta^{\text{shifted}}$ against $-t$ directly:
   - **H68-A supported** if $\Delta^{\text{shifted}} \approx -t$ within a
     pre-declared tolerance, consistently across subjects and shift
     magnitudes.
   - **H68-B supported** if the discrepancy $\Delta^R = \Delta^{\text{shifted}} - (-t)$
     is significantly correlated with lesion size (Spearman, permutation
     test, matching this project's own established discipline) or another
     measurable covariate, tested BEFORE looking at which covariate "wins"
     (pre-register the candidate list: native lesion size, local gradient
     magnitude of $E_1$ near the shifted region, decoder-stage depth if
     extended beyond `enc1`).
   - **H68-C supported** if neither A nor B holds.

This experiment is strictly cheaper than the original Section 4's full
pipeline (no $L_{\text{off}}$ to calibrate, no multi-seed Dice run needed to
get a first answer) and answers the actually load-bearing question before any
larger investment.

### 7.4 Corrected novelty statement

Whatever H68-A/B/C's outcome, the base SC-DCU module (Section 2.1) and the
comparison methodology (measuring geometric-vs-task-optimal discrepancy under
a controlled synthetic shift, in a 3D single-volume segmentation skip
connection) still appear unreported in the E67/E67b search — but per the
correction above, that alone is a methodological/application observation, not
a claimed new principle. If H68-B holds, the discrepancy-structure finding
itself (not the base module) would be the actual candidate for a defensible
contribution, on the same evidentiary footing as E48/E65's own causal
findings — discovered, not invented, and reported honestly regardless of
which way it comes out.

## Next step (not yet started)

Implement Section 7.3's discriminating experiment ONLY: the base SC-DCU
module (Section 2.1) trained briefly with `L_seg` alone (no offset-fidelity
term of any kind), then measure $\Delta^{\text{shifted}}$ against $-t$ on
held-out synthetic shifts and test for H68-A vs. H68-B vs. H68-C. Do not
implement any offset-fidelity loss, calibrate any $\lambda_{\text{off}}$, or
run a multi-seed Dice comparison until this experiment resolves which
hypothesis holds.
