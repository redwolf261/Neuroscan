# Phase E10: Margin Loss Selection — One Choice, Four Rejections, Justified

**Status**: ✅ Complete — De Brabandere-style pairwise hinge margin selected

**Date**: 2026-08-04

## The five candidates under consideration

1. Contrastive loss (SupCon-style)
2. Triplet loss
3. Hinge margin (De Brabandere et al.'s discriminative loss form)
4. ArcFace/CosFace-style angular margin
5. A fifth "plain hinge" option — see note below on why this collapses into #3

All five formulas and their source papers are already documented in
`PHASE_E5_LOSS_MATH_LITERATURE.md`; this document does not re-derive
them, only evaluates and selects among them against EGGO-v1's specific
constraints.

## Constraints the chosen loss must satisfy

From `PHASE_E8_EGGO_V1_SIMPLIFIED.md` and `PHASE_E9_TRAINABLE_BOUNDARY_HEAD.md`:

1. **Per-voxel, dense prediction** — not per-image; must operate on tens
   of thousands of embeddings per volume, batch-tractable (per
   `PHASE_E5_ALGORITHM_DESIGN.md` §5's complexity findings — subsampled,
   in-batch only).
2. **Cleanly weightable by a scalar multiplier** ($\hat U_i \cdot B_i$)
   **without conflating attraction and repulsion** — since EGGO-v1 only
   uses the separation force (no compactification in v1), the loss
   itself doesn't need two decomposable terms the way full EGGO would,
   but it must still accept a clean per-voxel weight without internal
   normalization effects that would fight the weighting (a concern that
   ruled out SupCon-style losses even for the *simpler* single-term case
   — see below).
3. **Binary class setting, extreme imbalance** (~1% tumor voxels) — must
   not require per-class weight vectors sized for many-class problems.
4. **No implicit temperature/normalization coupling that competes with
   the explicit boundary-proximity weight $B_i$** — since $B_i$ already
   encodes "how much to weight this voxel," the loss's own internal
   mechanics shouldn't introduce a second, uncontrolled weighting effect.

## Evaluation and rejections

### Rejected: Contrastive loss (SupCon-style)

**Rejection reason**: `PHASE_E5_LOSS_MATH_LITERATURE.md`'s search
explicitly found SupCon's positives appear only in the numerator while
negatives appear only in the denominator normalizer — push and pull are
**mathematically entangled in one log-ratio, not cleanly separable**.
The SINCERE paper (cited in that search) demonstrated this causes
*unwanted intra-class repulsion* as a side effect of the normalization
term, even for same-class examples not being actively pulled. This
directly violates constraint 4: SupCon's internal log-sum-exp
normalization would interact with $\hat U_i \cdot B_i$'s weighting in
ways that are hard to reason about or attribute cleanly — a bad fit for
a "smallest testable hypothesis" whose entire purpose is clean,
attributable measurement (per E8's stated rationale).

### Rejected: Triplet loss

**Rejection reason**: requires an explicit anchor/positive/negative
**sampling strategy** as a first-class design decision (which anchors,
how positives/negatives are selected per anchor) — this adds a
meaningful additional hyperparameter and implementation-complexity
dimension on top of the in-batch negative sampling already required by
the complexity constraints in `PHASE_E5_ALGORITHM_DESIGN.md` §5. E8's
explicit goal was minimizing moving parts for the first test; triplet
sampling strategy is exactly the kind of extra knob that would make a
null result ambiguous (was it the mechanism, or the sampling strategy?).
Not rejected because triplet loss is bad in general — rejected because
it adds an avoidable degree of freedom at the stage where degrees of
freedom should be minimized.

### Rejected: ArcFace/CosFace-style angular margin

**Rejection reason**: per `PHASE_E5_LOSS_MATH_LITERATURE.md`'s formula
extraction, both require a **per-class learnable weight vector** inside
a softmax normalized over all classes — a mechanism built for many-class
face recognition (thousands of identity classes), not naturally suited
to a 2-class (tumor/background), 1%-vs-99%-imbalanced, per-voxel dense
setting. Adapting it correctly would require real design work (e.g., how
does the "class weight vector" formulation behave with 99% of the
minibatch concentrated in one class?) that has no existing precedent to
borrow from safely — a nontrivial, unvalidated adaptation, not a
drop-in fit. Deferred, not permanently excluded, should ArcFace-style
margins later prove relevant for a different formulation.

### Selected: De Brabandere-style pairwise hinge margin

$$\mathcal{L}_{margin} = \frac{1}{|B|}\sum_{i \in B} \hat U_i \cdot B_i \cdot \frac{1}{|N(i)|}\sum_{j \in N(i)} \big[\,2\delta_d - \lVert z_i - z_j \rVert\,\big]_+^2$$

This is the separation term already derived in
`PHASE_E5_ALGORITHM_DESIGN.md` §4.1, unchanged by the E8 simplification
except that $\mathcal{L}_{comp}$ is dropped entirely (not folded into
this term). $N(i)$ = same-minibatch opposite-class voxels (in-batch
negative sampling, per the complexity analysis). $\delta_d$ is the
margin hyperparameter (subject to the sweep in Phase E13, per the
original design's Problem 7).

**Why this satisfies all four constraints directly**:
1. Native per-pixel/per-voxel design (De Brabandere et al. 2017 built it
   for exactly this — dense instance segmentation embeddings).
2. The hinge structure $[\,\cdot\,]_+^2$ has **no internal normalization
   term** (no softmax, no log-sum-exp denominator) — the external weight
   $\hat U_i \cdot B_i$ multiplies the loss directly and cleanly, with no
   competing internal weighting mechanism to reason about.
3. Operates on raw pairwise Euclidean distance between two voxels'
   embeddings — no per-class weight vectors, trivially handles the
   binary, imbalanced setting (the imbalance is handled by the sampling
   of $N(i)$, not by the loss formula itself).
4. As established in point 2, no competing internal normalization.

**Note on the "fifth option"**: a generic "plain hinge margin" is not a
distinct fifth candidate — De Brabandere's $L_{dist}$ *is* the plain
pairwise hinge margin form, just already correctly attributed to its
source rather than presented as a separate invention. Listing it
separately in the original five-way framing would have double-counted
the same mathematical object under two names.

## Summary table

| Candidate | Fits per-voxel dense setting? | Cleanly weightable, no competing normalization? | Fits binary/imbalanced class setting without adaptation? | Verdict |
|---|---|---|---|---|
| Contrastive (SupCon) | Yes | **No** — entangled push/pull, SINCERE-documented side effects | Yes | Rejected |
| Triplet | Adaptable | Yes | Yes, but adds sampling-strategy DOF | Rejected (unnecessary complexity for v1) |
| ArcFace/CosFace | No (needs adaptation) | Yes | **No** — designed for many-class softmax | Rejected |
| **De Brabandere pairwise hinge** | **Yes, native** | **Yes, no competing terms** | **Yes, native** | **Selected** |

## Files

| File | Purpose |
|---|---|
| `PHASE_E5_LOSS_MATH_LITERATURE.md` | Source of all five candidates' exact formulas |
| `PHASE_E8_EGGO_V1_SIMPLIFIED.md` | The design this loss plugs into |
| `PHASE_E11_IMPLEMENTATION_SPEC.md` | Next step |

---

**Completed**: 2026-08-04
