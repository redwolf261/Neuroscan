# E197 — Is the tail's missing ET evidence inferable? A partial REOPENING.

**Date**: 2026-09-20
**Status**: RESULT. Feasibility/upper-bound test. **Deliberate in-subject leakage** — this is an
inferability bound, NOT a generalization estimate. No method proposed, no novelty claimed.
**Follows**: E196's question — *what exactly is missing in the 15, and can it be inferred from
what is present?*

---

## Design

For the 15 catastrophic-tail subjects (ET<0.5 or TC<0.5) and a random 15-subject control from
the good stratum, train a per-voxel MLP (4→64→64→1) **on the subject's own labels and test on
the same subject**. The leakage is intentional: it measures whether the discriminative
information is *present in the voxel intensities at all*, giving an upper bound.

Four feature sets: `delta only` ($t1c-t1n$), **`no_t1c`** ($t1n,t2f,t2w$ — the channel E142
calls dead is removed), `all-4`, and `all-4 + spatial` (local mean/std of $\delta$, $5^3$).

## Result

| group | model ET | delta-only | **NO t1c** | all-4 | all-4+spatial |
|---|---:|---:|---:|---:|---:|
| **TAIL** (n=15) | 0.227 | 0.343 | **0.361** | **0.681** | **0.741** |
| CONTROL (n=15) | 0.861 | 0.821 | 0.563 | 0.901 | 0.926 |

### The leakage control passes

If the in-subject oracle were merely memorising voxels, CONTROL would also reach ~1.0. It does
not — CONTROL all-4 = 0.901 vs model 0.861, a gap of only **+0.039**. The estimator is
constrained by real evidence, not fitting noise.

### The finding

$$
\boxed{\text{TAIL oracle-minus-model gap} = +0.454 \quad\text{vs}\quad \text{CONTROL} = +0.039 \;\;(\approx 10\times)}
$$

On the tail, a *context-free per-voxel intensity classifier* with access to the subject's own
labels reaches **0.681** where the full 3D CNN reaches **0.227**.

## This partially contradicts the strong reading of E142

E142 (causal, n=40) showed zeroing t1c collapses ET 0.8433 → 0.0015, and the inference drawn was
that the tail subjects "functionally *are* that ablation" — i.e. information-absent.

E197 does **not** support that in full strength:

- `no_t1c` on the tail = **0.361**, not ~0. With t1c *removed entirely*, the other three
  modalities still carry substantial ET structure (they carry 0.563 on controls).
- `all-4` on the tail = **0.681**, and dropping t1c costs **−0.320**. So t1c is contributing a
  great deal on these subjects — it is **not** an informationally dead channel for them.

The correct restatement: **E142 measured what the trained network's ET pathway depends on, not
what the images contain.** The network routes ET through t1c, so ablating t1c destroys its ET
output. That is a statement about the learned model, not about information availability.

## What this does NOT establish (the decisive caveat)

**In-subject leakage is doing unknown work.** E136 made exactly this error: its "oracle 0.531"
was fitted and evaluated on the same subject's labels, and E143's clean 5-fold cross-fitting
corrected the associated transfer estimate by **4.6×**. The relevant contrast:

| | in-subject oracle | cross-fitted (E143) |
|---|---|---|
| tail-type subjects | E136: 0.531 → E197: 0.681 | — |
| full cohort mlp_deep | — | 0.8082 |

E143's cross-fitted $O_i$ on tail subjects is **low** (the low-$O_i$ population, mean 0.279) —
which is precisely the quantity E195 showed caps the achievable gain at +0.377pp.

$$
\boxed{\text{E197 (in-subject, 0.681) and E195 (cross-fitted, 0.279) are measuring different things.}}
$$

The gap between them **is** the open question:

- If the tail's ET boundary is **subject-specific** (each needs its own decision threshold),
  then in-subject 0.681 is unreachable without that subject's labels → E195 stands, closed.
- If a **transferable** rule exists that the current global threshold misses, there is real
  headroom → E196's dichotomy has a third row after all.

E136's original framing of this ("the decision boundary that works on these subjects DIFFERS
from the population boundary") was **explicitly retracted by E143** as an estimator-calibration
artifact. E197 revives the question without resolving it.

## The value at stake, if it were reachable

| scenario | 3-region mean | gain |
|---|---:|---:|
| lift tail ET to in-subject oracle 0.681 | 0.8783 | **+1.864 pp** |
| lift tail ET to 0.741 (with spatial) | 0.8806 | **+2.088 pp** |

Both clear the ≥1pp bar — **ET alone, TC untouched.** This is the first candidate in the entire
project with oracle headroom above the bar that is not already closed by a causal test.

Also noted: tail ET fraction of tumour = **0.082** vs control **0.211**. The tail is partly a
small-target regime, a known confound that must be controlled in any follow-up.

## The one experiment that decides it

**Cross-fitted transfer restricted to the tail.** Train observers on the *other* tail subjects
(leave-one-out within the tail, never the target subject), test on the held-out tail subject:

- If held-out tail Dice ≈ 0.68 → the rule transfers → real headroom → E196 reopens.
- If held-out tail Dice ≈ 0.28 (matching $O_i$) → subject-specific boundary → E195/E196 stand
  and the closure is complete.

This is **hours, no training of any segmentation network**, and it is the correct next step. It
must be run before any method is proposed, and before any prior-art audit — the audit is wasted
if the transfer fails.

**Discipline note**: E197 is deliberately leaky by construction. It must never be quoted as a
performance figure. Its only legitimate use is the comparison against a cross-fitted number.

**Artifacts**: `E197_inferability.json`; script `e197_inferability.py` (scratchpad).
