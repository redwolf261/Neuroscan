# Marginal Recovery Decomposition (MRD): Formal Method Specification

Frozen after E281–E285. This document fixes the exact mathematical
object, its empirically-established properties, and its precise
distinction from related prior art, as the basis for the paper's
method section and the next prior-art audit pass.

## 1. Setting and notation

- A frozen, pretrained segmentation network produces, for each voxel
  `x` in a subject's D1 feature map, a *production* probability
  `p = P(x) = sigmoid(w_P . x + b_P)`, where `w_P, b_P` are the
  frozen weights of the production model's own linear segmentation
  head for the target class (here: enhancing tumor, ET). `P` is
  never updated after this point; no gradient from anything below
  ever touches `w_P, b_P`.
- `y in {0,1}` is the voxel's ground-truth label for the same class.
- `x` is the same frozen intermediate feature representation `P`
  itself consumes (D1, 32-dim, `relu2_out` in this codebase) — MRD
  does not require its own feature extractor; it reads the
  representation already computed for `P`.
- A second, independently-trained linear readout `R_phi(x) =
  sigmoid(w_phi . x + b_phi)` is fit on top of the SAME frozen `x`.
  `R_phi` is the only object MRD trains.

## 2. The target transform

MRD's defining choice is not architectural — `R_phi` has the same
form as `P` (both are linear-plus-sigmoid readouts on the same
features) — it is a **transformation of the supervision target**
given to `R_phi` during training.

Ordinary supervision would fit `R_phi` to predict `y` directly. MRD
instead fits `R_phi` to predict:

```
T_MRD(y, p) = clip( (y - p) / (1 - p + eps), 0, 1 ),   eps = 1e-6
```

For binary `y` the clipped target has an EXACT closed form — this
corrects an error in an earlier draft of this document, which
incorrectly stated `T_MRD(y=1,p) = 1` for all `p`. The correct
closed form is:

```
T_MRD(y=1, p) = (1-p) / (1-p+eps)        (decays from ~1 toward 0 as p -> 1)
T_MRD(y=0, p) = 0                        (clip floors every negative at 0;
                                           the unclipped value -p/(1-p+eps)
                                           is <= 0 for all p in [0,1], so
                                           clipping to [0,1] always sends
                                           it to exactly 0)
```

So **every negative voxel's target is exactly 0** (identical to
ordinary supervision) — clipping removes the negative branch
entirely, and this transform's entire effect is on how POSITIVES are
weighted. `T_MRD(y=1,p)` is NOT constant at 1: it decays smoothly
from ~1 (for small `p`) toward 0 as `p -> 1`, with the decay's shape
controlled by `eps`. The useful comparison is against `T1`'s own
positive-class target, `clip(1-p, 0, 1) = 1-p` (a bare linear decay),
which decays MUCH faster than `T_MRD`'s own positive-class target
over most of the probability range, verified directly on the real
R1-B training pool (5,003 ET positives, seed=999) rather than
asserted:

| `p` range | n (of 5,003) | mean `T1`=`1-p` | mean `T2`=`T_MRD` |
|---|---|---|---|
| `[0, 0.5)` | 908 | 0.930 | 0.999999 |
| `[0.5, 0.9)` | 249 | 0.255 | 0.999995 |
| `[0.9, 0.99)` | 343 | 0.037 | 0.999959 |
| `[0.99, 0.999)` | 404 | 0.0038 | 0.999597 |
| `[0.999, 0.9999)` | 531 | 0.00038 | 0.995985 |
| `[0.9999, 0.999999)` | 1015 | 0.000022 | 0.853967 |
| `[0.999999, 1.0)` | 1549 (31%) | ~0 | 0.082 |

**This is the actual mechanism, stated precisely rather than loosely**:
`T1` (bare `1-p`) collapses to a numerically negligible target almost
immediately once `p` exceeds ~0.9 — by the time `p=0.99`, `T1`'s mean
target is already down to 0.004, essentially discarding that voxel's
training signal. `T2` keeps the target near its maximum (>0.99) all
the way out to `p=0.9999`, only decaying substantially in the final
two bins — and even in the very last bin (`p>0.999999`, the
`eps`-dominated regime E282-A characterized, 31% of the real positive
pool), `T2`'s mean target (0.082) is still roughly 4 orders of
magnitude larger than `T1`'s (~0, i.e. `T1` has already flattened to
zero everywhere `T2` is still contributing meaningfully). Overall,
on this real pool: mean `T1`=0.184, mean `T2`=0.685 — a large,
real, pool-wide difference, not a razor-thin-band effect. `T2`'s
"stays near full weight across nearly the entire range" description is
therefore correct RELATIVE TO `T1`'s much faster decay, not an
absolute claim that `T2≈1` everywhere; an earlier draft of this
document also asserted, without justification, that clipped-away
negative-branch signal is "realized instead as positive-side training
signal around that voxel's neighbors in representation space" — this
claim does **not** follow from the stated loss and has been removed;
no such neighbor-transfer mechanism exists in this pipeline.

**The useful equivalent form** for intuition (obtained without the
clip, i.e. describing the pre-clip quantity whose sign and magnitude
motivate the construction; this is an algebraic rewriting of the SAME
already-tested T2, not a new quantity, confirmed in E285's audit):

```
T_MRD(y, p) = y - (1-y) * p/(1-p+eps)
```

For `y=1`: contributes `+1` (pre-clip; the actual POST-clip positive
target is the smoothly-decaying `(1-p)/(1-p+eps)` derived above, not
a constant). For `y=0`: contributes `-p/(1-p+eps)`, the NEGATIVE of
the reference model's own foreground **odds** — this quantity is
always `<=0` and is clipped to exactly 0, so it never directly enters
training; it is a useful INTERPRETIVE lens on why the positive-class
decay has the odds-complement shape it does (the same algebra that
produces `(1-p)/(1-p+eps)` for positives is the complement of the
odds-ratio expression for negatives), not a description of a second,
separately-acting training signal. This is the mechanism E283's
decomposition (`T0`/`T1`/`T2`/`T3`) and E285's focal-style control
make precise and falsifiable, not merely descriptive:

- **Removing production's existing explanation matters.** `T0 = y`
  (ordinary) vs `T1 = clip(y-p, 0, 1)` (bare subtraction): subtracting
  `p` from the label before fitting is responsible for the majority
  of MRD's gain over ordinary supervision (E283: T1 captures ~94% of
  T2's total improvement over T0 at FP~1000 on the G1 endpoint).
- **Normalizing by the remaining probability mass adds a further,
  real, variance-stabilizing refinement on top of subtraction.**
  `T2 = clip((y-p)/(1-p+eps), 0, 1)` beats `T1` at every tested
  operating point and has visibly tighter cross-seed variance
  (E283).
- **Dividing without first subtracting is degenerate.** `T3 =
  clip(y/(1-p+eps), 0, 1)` is statistically indistinguishable from
  `T0` (E283) — confirms the gain is not an artifact of "dividing by
  something probability-shaped," it specifically requires the
  subtraction.
- **The gain is not explained by "any monotonic down-weighting of
  easy positives."** A focal-style target with the same qualitative
  shape (`T_focal = y*(1-p)^gamma` for positives, 0 for negatives,
  `gamma=2`) beats `T0` but sits clearly below `T1` and well below
  `T2` at every budget, with the largest gap at the tightest FP
  budget (E285). The specific **odds-shaped** decay — staying near
  full weight across nearly the entire probability range and
  discounting sharply only very close to `p=1` — is doing real,
  non-interchangeable work.

## 3. Training objective

`R_phi` is fit by ordinary logistic regression against a continuous
soft target `t in [0,1]` via the standard sample-duplication /
sample-weighting construction (verified bit-exact against ordinary
hard-label logistic regression when `t in {0,1}`, E281):

```
L(phi) = - sum_i [ t_i * log(R_phi(x_i)) + (1 - t_i) * log(1 - R_phi(x_i)) ]
```

i.e. each training voxel contributes a weighted mixture of the
positive-class and negative-class log-loss terms, weighted by `t_i`
and `1-t_i` respectively. This is the unique natural generalization
of binary cross-entropy to a continuous `[0,1]` target and reduces
exactly to ordinary logistic regression when `t_i in {0,1}`.

`R_phi`'s training **data distribution** (which voxels contribute
positives/negatives, independent of what target value they're given)
is held FIXED across the entire E281–E285 campaign: `R_phi` is fit
on the previously-established "R1-B" pool — lesion-interior ET
voxels as positives, a curated mixture of local lesion-boundary
("shell") voxels and distant background voxels as negatives (E257-B,
reused unchanged). **The target transform (Section 2) and the
training-data distribution are orthogonal design choices** — E284
confirms the transform's benefit does not depend on which specific
prior model supplied `p` (see Section 5), and does not require any
particular sampling scheme beyond ordinary curated hard negatives;
conversely the companion finding (E280) is that the negative-sampling
distribution's own benefit is explained by the hardness it achieves,
not by boundary-anchoring specifically. These are two separable
contributions and should be described as such in the paper, not
conflated into one "MRD algorithm."

`R_phi` never receives `p` as an *input feature* — only `x`. Adding
`p` as an extra input dimension (tested as H2/H4 in E281, as the
`[x,p]`-input variants in E282-B) changes nothing when the target is
ordinary `y`, and changes nothing beneficial when the target is
`T_MRD` either (the best-performing variant throughout is always the
one using `x` alone). `p` is used ONLY to construct the training
target, never consumed by the trained model at inference time.

## 4. Inference rule

**MRD deploys `R_phi(x)` directly as the recovery score — it does
NOT reconstruct a combined probability.** This is a load-bearing,
counterintuitive, and explicitly tested design choice, not an
oversight:

```
score(x) = R_phi(x)            [CORRECT — this is what MRD is]

score(x) = p + (1-p) * R_phi(x)   [INCORRECT for this purpose —
                                    falsified in E282]
```

The algebraically-motivated reconstruction `p + (1-p)*R_phi(x)`
("production's own contribution plus the remaining recoverable
contribution") is the natural inverse of the marginal-target
construction and was the user's original formulation's inference
rule. E282-A/B tested it directly: at full scale (67 subjects, 113
G1 lesions, 5 seeds), this reconstruction collapses recovery to
**exactly 0.0%+-0.0%** at every tested FP budget (100/500/1000),
while the raw-`R_phi(x)` score behaves normally and reproduces every
later positive finding (E283–E285). Root cause (traced via the full
FROC curve, not merely observed): the reconstruction's
G1-lesion-relevant score mass only becomes accessible above
FP~1560/subject, roughly 15–25x looser than the budgets of interest
— it is not that the reconstruction "doesn't work," it works at an
impractically loose operating point. **The paper must state this
explicitly as a finding, not omit it**: the two scoring rules are not
interchangeable, and the mathematically "more complete" one is
empirically much worse for this task.

At deployment, `score(x) = R_phi(x)` is swept over a threshold exactly
as `P(x)` itself would be, to select an operating point; MRD is used
as an independent recovery channel (e.g. flagging additional
candidate lesions at a controlled false-positive budget), not as a
replacement for or modification of the production model's own output.

## 5. What the transform does and does not depend on (causal scope, E284)

A dedicated causal battery (E284, 7 conditions, same frozen pool/
architecture/evaluation as E283) established the following boundary
conditions on when `T_MRD`-style training helps:

| Condition | What changes | Result | Conclusion |
|---|---|---|---|
| `T_P = clip(y-p,0,1)` | true, subject-matched production `p` | works (= E283's T1) | baseline |
| `T_pi = clip(y-p_pi,0,1)` | `p` swapped for a DIFFERENT subject's `p`, same architecture | collapses to ~ordinary-target level | **subject-matching is required** |
| `T_pbar = clip(y-pbar,0,1)` | `p` replaced by a frozen scalar population mean | collapses to ~ordinary-target level | **a genuine per-voxel estimate is required, not a constant** |
| `T_Q = clip(y-q,0,1)` | `p` replaced by an INDEPENDENTLY TRAINED model's own `q`, same subject | tracks `T_P` almost exactly | **the specific identity of the reference model does NOT matter** |
| `T_neg = clip(p-y,0,1)` | sign of the subtraction reversed | collapses to EXACTLY 0% at every budget | **the sign/direction of `y-p` is essential** |

The resulting scope statement: **MRD requires a genuine,
subject-matched probability estimate from SOME reasonably competent
prior segmentation model, used with the correct sign** — it is not
specific to the particular frozen production checkpoint that happens
to be complemented in this pipeline, and it does not work with any
constant or mismatched substitute. This is a materially different
(broader, more portable, but more modest) claim than
"production-conditioned residual learning," and the paper should use
this exact, narrower phrasing.

## 6. Distinction from residual-correction / error-prediction methods

Per the user's own prior-art audit, the following distinctions are
the ones that must appear explicitly in the paper, because each one
is a point where a naive reading of "MRD" would collide with existing
disclosed work:

1. **Not `y - p` alone.** Plain residual targets (`r = y - p`) are
   disclosed prior art in brain-tumor segmentation (recent residual-
   correction/diffusion-refinement lines of work operate on exactly
   this quantity). MRD's target is `(y-p)/(1-p+eps)`, which E283
   shows is empirically and not merely notationally different from
   `y-p` alone (T2 > T1 at every tested budget, with tighter
   variance) — but the paper must never describe `y-p` itself as the
   novel ingredient, since it is not.
2. **Not "frozen first stage -> second refinement stage."** This
   general architectural pattern (freeze an initial prediction, train
   a second model conditioned on it) is also disclosed prior art
   (first-stage-probability-as-frozen-input refinement architectures;
   error-estimator patents trained against a ground-truth-derived
   error representation). MRD's architectural shape is consistent
   with this general pattern and must not be claimed as novel on its
   own.
3. **Not "error prediction" generically.** "Learn the errors of an
   existing segmentation model" is an old, multiply-disclosed idea
   (SESV/DEP-Net-style approaches; error-estimator patents). MRD's
   target is a SPECIFIC transform of the error (odds-normalized, not
   a raw or differently-shaped error signal), which is the part that
   must carry the claim, not the generic "predict the error" framing.
4. **Not hard-negative mining generically.** R1-B's negative-sampling
   construction (lesion-shell + distant background) is a SEPARATE
   contribution from the target transform (Section 3's "orthogonal
   design choices" point) and should not be bundled into "MRD" as if
   it were one inseparable invention — hard-negative mining itself is
   old and multiply disclosed (including very recent frozen-feature +
   hard-negative work), and E280 already found R1-B's own specific
   construction is explained by the hardness it achieves rather than
   by the particular boundary-anchoring heuristic.
5. **The one component with no exact single-reference match located
   in the audit performed so far**: the specific composition
   `X_frozen -> p (same-subject, frozen reference model) ->
   (y-p)/(1-p+eps) -> R_phi -> deploy R_phi(X) directly as an
   independent recovery score (NOT reconstructed via p+(1-p)*R_phi)`.
   This is the claim the paper should center, phrased exactly this
   narrowly. Per the audit's own finding, "no exact match located" is
   evidence of non-location, not a novelty certification — the paper
   should state the claim's scope precisely enough that a reader (or
   examiner) can test it directly, rather than asserting novelty.

## 7. Summary chain (for the method section's opening paragraph)

```
y                                    ordinary supervision (baseline, T0)
  |
  v  subtract what the reference model already explains
y - p                                 (T1; captures ~94% of the total gain)
  |
  v  scale the remainder by the reference model's own foreground odds
(y - p) / (1 - p + eps), clipped to [0,1]      (T2 = T_MRD; best single transform,
  |                                              tightest cross-seed variance)
  v  fit R_phi against this target, using ONLY x (not p) as input
R_phi(x)
  |
  v  deploy directly -- do NOT reconstruct via p + (1-p)*R_phi(x)
score(x) = R_phi(x)
```

Each arrow in this chain corresponds to a specific, already-run
ablation that empirically justifies keeping it (E282 for the last
step's "do not reconstruct" warning; E283 for the first three; E284
for the scope/boundary conditions on what `p` may be substituted by;
E285 for ruling out "any similarly-shaped decay would do").
