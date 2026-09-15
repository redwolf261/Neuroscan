# E157 — Bottleneck Self-Knowledge: Mechanism Reduction & Prior-Art Audit

**Date**: 2026-09-15
**Status**: Steps 2–4 of the E156 workflow. No compute run; all numbers verified against
primary artifacts on disk.
**Follows**: E156 (Case B — all five positive results are known methods or oracle interventions)

---

## Why this asset

E156 established that the project holds no result that is both robust and novel. Two assets
have unexhausted novelty, both mechanistic rather than performance. This audits the stronger
one.

**Selection criterion**: it is the only phenomenon in the project that has (a) been replicated
across two independent architecture/checkpoint families, (b) survived a size control that
*strengthened* it, and (c) an intact, runnable generator on disk.

---

## Step 1 — Exact provenance (verified, not inherited)

### The phenomenon

A small auxiliary head, reading only the un-ablated bottleneck representation $z_i$, predicts
the subject's own bottleneck-ablation sensitivity

$$d_i = \text{Dice}(f_\theta(x_i)) - \text{Dice}(f_\theta^{\text{ablate}}(x_i))$$

on held-out subjects, where $f^{\text{ablate}}$ zeros the bottleneck before the decoder.

### Two independent measurements

| | E71 (original) | E109 (replication) |
|---|---|---|
| Checkpoint | `e70/MM_seed0`, UNet3D_v3, multimodal | `e46/AttnGate_seed0`, UNet3D_v5, single-modality |
| Held-out Spearman | **+0.701** | **+0.6328** |
| Permutation p | <0.001 | 0.0 |
| Partial ρ controlling lesion size | **+0.904** (p=4.1e-47) | **+0.7908** (p=5.46e-28) |
| n held-out | 125 | 125 |
| Aux-head training set | disjoint 200 subjects | disjoint 200 subjects |
| Generator on disk | **NO** — checkpoint deleted | **YES** — `e109/run_e109_self_prediction_basis.py` |

Verified directly from `experiments/exp_e12_eggo_m/e109/E109_summary.json`.

**Correction to E156**: E156 quoted "ρ≈0.87–0.90" and "+0.904". Those are **E71's** numbers, on
a checkpoint that **no longer exists on disk** (confirmed in the E109 memory record). The
reproducible figures are E109's: **raw +0.633, partial +0.791**. Still strong, still
size-independent, but the correct numbers to cite going forward.

**Why the partial exceeds the raw correlation**: lesion size mildly *suppresses* the raw
association. The self-predictive signal is not a size proxy — controlling size makes it
cleaner. This is the single most important statistical property of the finding.

### What was already ruled out as the basis (E109 Step 5)

| Candidate basis | Result |
|---|---|
| Probe reads high-magnitude channels | ρ = −0.054, p = 0.389 — **NULL** |
| Probe reads high-variance channels | ρ = +0.053, p = 0.402 — **NULL** |
| A few dominant channels | top 10% of channels carry 23.7% of sensitivity (uniform = 9.8%) — concentrated ~2.4×, but not dominated |

The representational basis is **genuinely unexplained**. Both cheapest hypotheses are cleanly
dead.

### Methodological caution carried forward

E109 caught an ablation-construction bug that first produced a false `DOES_NOT_REPLICATE`
(ρ=+0.019). The cause: conditioning the attention gate on the always-intact bottleneck instead
of the possibly-ablated one, contradicting E48's own deliberate convention. Detected via a
**14× N_b scale mismatch** against E48's stored table. Any future work unrolling v5's attention
gate outside `forward()` must check E48's convention directly
(`e48/run_e48_bottleneck_encoding_audit.py:137-140`), not assume it.

---

## Step 2 — What is the actual mechanism?

Not "we added an auxiliary head." The operation is:

> A representation predicts the **counterfactual degradation of the system it is part of**,
> under a model-space intervention on itself, per input.

Reduced further, the phenomenon is a statement about *what information the bottleneck contains*:

$$\boxed{z_i \;\longrightarrow\; \text{(how much the decoder's output depends on } z_i)}$$

The representation encodes its own **causal load-bearingness** for this specific input —
information about its own downstream necessity, not about the task label.

This is the claimable object. It is a property of learned representations, discovered by
intervention, and it is what distinguishes the finding from ordinary difficulty estimation
(which the size control already excludes).

---

## Step 3 — Separating mechanism from implementation

This is the decisive section, because E71/E72/E73 all failed and the question is **whether the
principle failed or three specific levers did**.

### The levers that were tried

| Lever | What it assumed | Outcome |
|---|---|---|
| **E71 CDCG gate** — blend coarse/fine at `enc1` by $\sigma(\hat d_i)$ | bottleneck-dependence and skip-dependence are **substitutable opposing resources** | **KILLED** |
| **E72 FWL** — weight the loss by fragility | fragility marks subjects that benefit from more weight | KILLED (aggressive harms, gentle inert) |
| **E73 SDLR** — spatial gate on predicted sensitivity | the signal survives distillation into a spatial map | KILLED (signal inert once wired in) |

### The E71 kill is the informative one

From `PHASE_E71_GATE_MECHANISM_KILLED.md`, an **inference-only** test using the *measured*
$d_i$ (so prediction fidelity cannot be blamed):

| Quantity | Value |
|---|---|
| Spearman($d_i$, enc1-suppression-tolerance slope) | **−0.290** |
| Parametric p | 1.0e-3 |
| Permutation p | 0.001 |
| **Sign required for the gate to work** | **positive** |

Subjects with higher bottleneck-sensitivity are hurt **more**, not less, by skip suppression.

**Diagnosis recorded there**: bottleneck-dependence and skip-dependence are **not substitutable
resources**. They are correlated indicators of overall fragility — remove either major pathway
and a fragile subject suffers.

### What this does and does not kill

**It kills**: every intervention of the form *"use $\hat d_i$ to trade pathway A against
pathway B."* The trade-off those designs presuppose does not exist in this network. E72 and
E73 are variants of the same allocation assumption and died the same way.

**It does not kill**: the phenomenon itself, which is a statement about *information content*,
not about *substitutability*. The three levers share one unexamined premise —
**that a self-knowledge signal's use is to reallocate capacity** — and that premise was
falsified, not the signal.

This is the same structural error the E147 branch hit independently: E147's own record
(`e147_representation_demand_inverse_kills_allocation`) found the demand–capacity correlation
runs **backwards** for allocation, and that 74/88 subjects are served by rank ≤4 of 256, i.e.
**no capacity scarcity exists to allocate**. Two independent branches, same conclusion:
*this network has no scarce resource to reallocate.*

> **Therefore: any future exploitation route must not be an allocation/trade-off design.**
> That family is closed by direct evidence, three times over.

---

## Step 4 — Targeted prior-art audit

### Novelty statement under test

> A network trained with an auxiliary objective to predict its own **component-ablation-derived,
> per-sample causal sensitivity** — measured by a real model-space intervention, used during
> training to shape inference-time behaviour, with no intervention at test time.

### What E71's own search covered (≈12 queries, structural not keyword-only)

| Prior art | Relationship | Classification |
|---|---|---|
| Counterfactual distillation (CFKD family, Clever-Hans fixing) | Intervenes on the **input** ($x \to x'$); distils label flips; purpose is debiasing | 🟡 adjacent, differs on a load-bearing axis |
| Component-ablation studies (InfiltrNet, MedSAM-CA, DMAF-Net, "No Modality Left Behind") | Ablation as a **post-hoc validation table**, never a training signal | 🔵 established, different use |
| Curriculum from confidence / gradient norm | Uses a **proxy**, not a measured causal quantity | 🔵 established |
| Mechanistic interpretability predicting unit-ablation impact | Post-hoc, unit-level, for interpretability | 🟡 adjacent |

The load-bearing distinction is $x \to x'$ (input-space) versus $f \to f^{\text{ablate}}$ with
$x$ held fixed (model-space). That is a genuine structural difference, and it is the same
discipline the project's E43–E65 causal chain used throughout.

### Honest classification

$$\boxed{\text{🟡 adjacent but potentially distinguishable}}$$

**Not 🟢.** Three reasons, stated plainly:

1. E71's own doc rates its confidence "moderately-high, not certain" and notes a search of that
   scope cannot prove absence. That assessment stands.
2. The search is **dated ~2026-09-02** and predates E109's replication. It has not been
   re-run.
3. Most importantly: **the novelty statement describes an intervention that does not work.**
   The "used during training to shape inference-time behaviour" clause was the CDCG gate — and
   that is dead. What survives is the *measurement*, and "a representation predicts its own
   ablation sensitivity" as a **finding** sits closer to mechanistic-interpretability prior art
   than the training-signal framing does.

### The real gap in the audit — NOW SEARCHED (2026-09-15)

No search had been run on the surviving object in its own right:

> Do learned representations encode their own causal necessity, and is that encoding
> size-independent and cross-architecture?

**Result: the space is OCCUPIED, and closer than E71's 2026-09-02 search found.**

#### Hit 1 — decisive

**"Can Neural Networks Learn by Experimenting on Themselves? Self-Interventional Learning from
Functional Consequences to Predictive Self-Knowledge"** — arXiv 2608.14894.

This paper trains a network to predict the **functional consequences of interventions on its
own structural components**, and uses those predictions to guide later structural decisions.
Its stated contribution — *"self-intervention can serve as a source of predictive
self-knowledge"* — is the same object E109 measured, under nearly the same name.

| Axis | CDCG / E109 | Self-Interventional Learning (2608.14894) |
|---|---|---|
| Intervention space | **model-space** (zero the bottleneck) | **model-space** (lesions, substitutions) |
| Intervention types | single component (bottleneck) | singleton lesions, **pairwise** lesions, directed substitutions |
| Target predicted | Dice degradation | task-performance change |
| Use of prediction | gate inference behaviour | guide structural decisions |
| Framing | "causal self-knowledge" | **"predictive self-knowledge"** |
| Controls | permutation, size-partial | held-out firewall, matched budgets, Holm correction, seed-level analysis |

The model-space-vs-input-space distinction that E71 identified as its **load-bearing novelty
axis** does not separate us from this paper — it is model-space too, and strictly more general
(pairwise and substitution interventions, not just one component).

#### Hit 2 — the framing is an established field

**"Unexpected Benefits of Self-Modeling in Neural Systems"** — arXiv 2407.10188. Networks
predict their own internal states as an auxiliary task; the finding is that self-modeling acts
as **self-regularization** (narrower weight distribution, lower RLCT). Different target
(activations, not ablation sensitivity), but it establishes "auxiliary self-prediction head"
as a known, named technique with its own literature.

Adjacent and active: Anthropic's introspection work (transformer-circuits.pub, 2025),
"Emergent Introspective Awareness in LLMs", "Self-Interpretability" (2505.17120), "Intrinsic
Robotic Introspection" (2011.01880), "Latent Introspection" (2602.20031). The general claim
*"a network can report on its own internal/causal state"* is a populated research area.

#### Hit 3 — input-dependence of ablation sensitivity is known

The ablation-studies literature already records that **layer/component importance varies by
input and by class** — *"some classes suffer more from ablating filters in a given layer than
others"* (1901.08644, 1804.06679). The Hydra Effect (2307.15771) documents input-dependent
self-repair under ablation. So "ablation sensitivity is per-sample heterogeneous" is not itself
a discovery.

#### What survives the search

Genuinely not found, after targeted searching:

- The specific application to **3D medical segmentation** / per-subject Dice sensitivity.
- The **size-independence** result — that partial ρ controlling lesion size (+0.791)
  *exceeds* the raw (+0.633), i.e. the signal is not a difficulty proxy. The prior work does
  not report an analogous confound control, because per-sample difficulty confounds are not
  the concern in a Fashion-MNIST/CIFAR structural-repair setting.
- **Cross-architecture replication** of the same phenomenon (v3/multimodal → v5/single-modality).

These are real, but they are **a domain instantiation plus two confound controls**, not a new
computational principle. Under this project's own stated standard — *"do not call a method
novel because its exact implementation was not found"* — that is not sufficient.

---

## Verdict and recommendation

**Phenomenon**: 🟢 real, replicated across two architecture families, size-independent,
generator intact.
**Basis**: ⚪ unexplained — magnitude and variance both cleanly excluded.
**Allocation-style exploitation**: 🔴 closed by three independent kills plus E147's
no-scarcity finding.
**Novelty as a training signal (CDCG framing)**: 🔴 — rests on a dead mechanism *and* the
model-space axis it claimed does not separate it from 2608.14894.
**Novelty as a representation-science finding**: 🔴 **OCCUPIED** — self-interventional
predictive self-knowledge is published, with a strictly more general intervention set.

$$\boxed{\text{Branch CLOSED for novelty. The phenomenon is real; the principle is taken.}}$$

### Why this closes rather than continues

E71's novelty case rested on one load-bearing distinction: **model-space** intervention
($f \to f^{\text{ablate}}$, $x$ fixed) versus **input-space** ($x \to x'$). That distinction
was correct against the CFKD family it was checked against in 2026-09-02. It does **not**
survive contact with arXiv 2608.14894, which is model-space, uses a superset of our
interventions (pairwise lesions and directed substitutions, not just one component), frames the
result as *"predictive self-knowledge"*, and carries stronger statistical controls (held-out
intervention firewall, matched budgets, Holm correction, seed-level analysis).

What remains ours is: the medical-segmentation instantiation, the size-independence control,
and cross-architecture replication. Those are a domain application plus two confound controls.
By this project's own principle 2 — *do not call a method novel because its exact
implementation was not found* — that is not a contribution claim.

### What this does NOT invalidate

The **finding** is untouched and stays in the record: on this architecture family, a bottleneck
representation predicts its own ablation sensitivity at held-out ρ=+0.633 (partial +0.791
controlling size), replicated across two checkpoint families, with magnitude and variance both
excluded as the basis. That is a correct, well-controlled measurement. It is simply not a
novel *principle*.

If the project ever writes up its causal-diagnostic methodology, this is a legitimate result
inside it — cited alongside 2608.14894, not against it.

### Not recommended

**(B) Explaining the representational basis** (per-channel co-activation, spatial-position
sensitivity) is now a question about *why an already-published phenomenon works in our domain*.
It is scientifically interesting and it is **not** a route to a novelty claim. It should not
consume the main effort. Do not start it as a paper track.

---

## Open items

- E71's original checkpoint is gone; its +0.701/+0.904 figures cannot be re-verified. Cite
  E109's +0.633/+0.791 as the reproducible numbers.
- The prior-art search predates E109 and has not been re-run.
- No search has ever been run on the representation-science framing (the actual gap).
