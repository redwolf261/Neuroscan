# The Optimization Controller: Mathematical Design

**Status**: Algorithm design, pre-implementation. Baseline (`final_model.py`) is frozen as of this document — everything below is a *new*, separate module that consumes the baseline's existing gradients; it does not modify the baseline's architecture, forward pass, or loss definitions.

**Grounding rule**: every design choice below is traced to a specific measured finding from Phase 7/8 (cited inline), not to what's fashionable in the literature. Where the evidence is weak or absent (training-stage dynamics), the design is deliberately kept minimal rather than invented.

---

## 0. What we are actually replacing

Baseline (confirmed exactly, `NEUROSCAN_MATHEMATICAL_SPECIFICATION.md` Priority 2/3):

$$L^{(t)} = \underbrace{0.5\,L_{Dice}^{(t)} + 0.5\,L_{FT}^{(t)}}_{L_{Hybrid}^{(t)}} + L_{Evid}^{(t)}, \qquad g^{(t)} = \nabla_\theta L^{(t)}, \qquad \theta_{t+1} = \text{AdamW}(\theta_t, g^{(t)}, \eta)$$

Every loss is summed into **one scalar** before a **single** `.backward()` call. AdamW never sees the individual $L_{Dice}$, $L_{FT}$, $L_{Evid}$ gradients separately — it only ever sees their pre-summed total. This is the one and only point our controller intervenes on.

---

## 1. Notation — branches, not just losses

The network is not a set of independent task towers; it is a **shared trunk with two output branches** (confirmed exactly, spec Priority 1/9):

- **Shared trunk** $\theta_s$: `AdaptiveSliceSelector`(frozen after MAE) → `Conv2D5Stem` → 4 encoder stages → `CBAM` → decoder up-blocks/skip-convs.
- **Segmentation branch** $\phi_{seg}$: `prob_head` only.
- **Evidential branch** $\phi_{evid}$: `causal_shared` → `{anatomy_head, pathology_head, noise_head}` → `causal_weights`.

$$\mathcal{I}(seg) = \{Dice, FT\}, \qquad \mathcal{I}(evid) = \{Evid\}$$

At training step $t$, for each branch $b \in \{seg, evid\}$:

$$g_b^{(t)} = \sum_{i \in \mathcal{I}(b)} \nabla_{\phi_b} L_i^{(t)} \qquad \text{(branch-local gradient — cheap, no extra backward needed, these are disjoint parameter sets)}$$

$$g_s^{(t)} = \nabla_{\theta_s}\Big(\sum_i L_i^{(t)}\Big) \qquad \text{(shared-trunk gradient — already merged by ordinary backprop)}$$

This branch/trunk split is the key structural fact the controller uses that a generic "per-loss" method (GradNorm, PCGrad) does not: **it treats the merge point where branches rejoin the shared trunk as the thing to control, not the abstract loss labels.**

---

## 2. What the evidence rules in and rules out

| Measured finding (Phase 7/8) | Implication for algorithm design |
|---|---|
| $\cos(g_{Dice}, g_{FT})$ consistently **positive** (epoch-mean 0.38–0.71, zero sign changes across 29 epochs) | No conflict between the two segmentation losses. **A conflict-resolution mechanism (PCGrad-style projection) has nothing to resolve here.** |
| $\cos(g_{Hybrid}, g_{Evid})$ consistently **near zero** (never a sustained sign, magnitude tiny) | Evidential is orthogonal, not competing. **No projection or Pareto trade-off needed between branches either** — the two branches' gradients are close to statistically independent directions. |
| Evidential/causal heads: **2–5× higher per-parameter gradient norm** than `prob_head`, despite $\lambda_{Evid}=10^{-3}$ | A real, structural **magnitude imbalance concentrated in specific branches** — this is what needs correcting, not conflict. |
| Total gradient norm: **CV ≈ 1.09**, sporadic spikes up to 22× baseline, concentrated encoder-side | A **temporal/batch-relative instability**, distinct from the magnitude imbalance above — needs its own mechanism, not the same one. |
| Static, hand-picked 1/3 scaling on evidential heads (Phase 8): **no significant quality change** at 25-epoch/9-patient scale | A **fixed global constant is the wrong shape of intervention** — motivates something continuously adaptive and branch-relative, not a single tuned number. |
| No clear phase transition in loss composition across early/mid/late thirds (pilot run) | **Do not invent an elaborate stage classifier.** Any stage-awareness in the controller must be minimal and honestly scoped to what the training loss itself signals, not a hidden phenomenon we haven't actually observed. |

**Conclusion**: the right class of algorithm is not conflict-resolution (PCGrad/MGDA — no conflict found) and not a static reweighting (already tried, inconclusive). It is an **adaptive, per-branch, self-relative gradient-magnitude controller** — reacting to each branch's own history rather than a fixed target.

---

## 3. The Controller

### 3.1 Per-branch statistics tracker (EMA)

For each branch $b$, maintain a running mean and variance of that branch's gradient norm, updated every step (smoothing constant $\beta \in (0,1)$, e.g. 0.9 — same role as Adam's own second-moment EMA, but tracked per-branch for controller decisions, not fed directly into the parameter update):

$$\mu_b^{(t)} = \beta\,\mu_b^{(t-1)} + (1-\beta)\,\|g_b^{(t)}\|_2$$
$$\nu_b^{(t)} = \beta\,\nu_b^{(t-1)} + (1-\beta)\,\big(\|g_b^{(t)}\|_2 - \mu_b^{(t)}\big)^2$$

Same tracking for the shared trunk, $\mu_s^{(t)}, \nu_s^{(t)}$.

### 3.2 Magnitude-equalizing base scale — addresses the 2–5× imbalance

$$\alpha_b^{(t)} = \left(\frac{\mu_s^{(t)}}{\mu_b^{(t)} + \epsilon}\right)^{\gamma}, \qquad \gamma \in [0,1]$$

The shared trunk's own gradient norm is used as the equalization *target* (not an arbitrary reference branch, not a fixed constant) — a branch whose typical magnitude already matches the trunk gets $\alpha_b \approx 1$ (no correction); a branch running consistently hot (like the evidential heads) gets $\alpha_b < 1$. $\gamma$ controls correction strength (0 = off, 1 = full equalization) and is the one knob to tune/ablate. **This generalizes the static 1/3 already tested** — replacing one hand-picked constant with a continuously-recomputed, principled ratio.

### 3.3 Branch-relative anomaly damping — addresses the CV≈1.09 instability

$$z_b^{(t)} = \frac{\|g_b^{(t)}\|_2 - \mu_b^{(t-1)}}{\sqrt{\nu_b^{(t-1)}} + \epsilon} \qquad \text{(self-relative z-score, not a global threshold)}$$

$$\delta_b^{(t)} = \frac{1}{1 + \max\!\big(0,\ z_b^{(t)} - \tau\big)}, \qquad \tau \approx 2$$

$\delta_b = 1$ under normal variation; shrinks smoothly only once a branch's *own* gradient norm exceeds $\tau$ standard deviations above *its own* recent mean. This is deliberately **not** global gradient clipping (which discards magnitude information above a fixed absolute cutoff regardless of what's normal for that branch) — it targets exactly the measured phenomenon (transient, branch-relative spikes), and it does not penalize a branch that is consistently, stably large — only deviations from its own baseline.

### 3.4 Combined per-branch multiplier

$$m_b^{(t)} = \alpha_b^{(t)} \cdot \delta_b^{(t)}$$

### 3.5 Training-stage gate — deliberately minimal, evidence-honest

No phase transition was confirmed in the pilot data, so this is **not** a learned classifier or hidden-state model. It is a single windowed relative-slope of total loss over two trailing windows of size $W$:

$$\rho^{(t)} = \frac{\bar{L}^{(t-W:t)} - \bar{L}^{(t-2W:t-W)}}{\bar{L}^{(t-2W:t-W)} + \epsilon}$$

Used only to **gate how aggressively the controller acts**, via a stage factor $s^{(t)} = \text{clip}(1 - |\rho^{(t)}|/\rho_0,\ 0,\ 1)$ ($\rho_0$ a small reference slope): while the loss is still moving fast (early, healthy optimization), $s^{(t)}\to 0$ and the controller mostly gets out of the way; once the loss plateaus, $s^{(t)}\to 1$ and the controller intervenes at full strength — exactly the regime where a stuck imbalance is most likely to be the actual bottleneck, and least likely to be confused with ordinary early-training dynamics. The final multiplier becomes:

$$m_b^{(t)} \leftarrow 1 + s^{(t)}\big(m_b^{(t)} - 1\big) \qquad \text{(interpolates between "no correction" and the full correction above)}$$

### 3.6 Merge and step

$$\tilde{g}^{(t)} = g_s^{(t)} + \sum_{b} m_b^{(t)}\, g_b^{(t)}$$

$$\theta_{t+1} = \theta_t - \eta_t\cdot\text{AdamW-update}\big(\tilde{g}^{(t)}\big)$$

The controller reshapes the gradient *before* it reaches AdamW; AdamW's own per-parameter adaptivity (its moment estimates) operates unchanged on top of this. The two mechanisms act at different granularities — branch-level (ours) vs. parameter-level (AdamW's own) — and don't fight each other. This matches your own pipeline sketch's "Merge → AdamW" step exactly.

---

## 4. Full pipeline (your requested format)

```
Forward Pass
   ↓
Compute L_Dice, L_FT, L_Evid separately
   ↓
Backprop to get g_seg, g_evid (branch-local) and g_s (shared trunk, standard backprop)
   ↓
Update EMA trackers: μ_b, ν_b, μ_s  (Sec 3.1)
   ↓
Compute α_b (magnitude equalization, Sec 3.2)
Compute δ_b (anomaly damping, Sec 3.3)
Compute s^(t) (stage gate, Sec 3.5)
   ↓
m_b = 1 + s·(α_b·δ_b − 1)
   ↓
g̃ = g_s + Σ_b m_b · g_b        ← Adaptive Optimization Controller output
   ↓
AdamW.step(g̃)
   ↓
scheduler.step()
```

Mapping onto your five requested pieces:

1. **Optimization Controller** = Sections 3.1–3.4 combined (the $\mu_b, \nu_b \to \alpha_b, \delta_b \to m_b$ pipeline).
2. **Adaptive Weight Update Rule** = Section 3.2 ($\alpha_b^{(t)}$, replacing the static per-loss $\lambda$ with a continuously-recomputed, trunk-relative ratio).
3. **Adaptive Gradient Scaling Rule** = Section 3.3–3.4 ($\delta_b^{(t)}$ and $m_b^{(t)}$, the actual multiplier applied to each branch's gradient before merging).
4. **Training Stage Detection Rule** = Section 3.5 ($\rho^{(t)}, s^{(t)}$) — intentionally the simplest component, since it's the one area current evidence doesn't strongly support anything more elaborate.
5. **Complete mathematical formulation** = Section 3.6 ($\tilde{g}^{(t)}$, the merge rule) plus this whole document.

---

## 5. Why this is not PCGrad, not GradNorm, not a Pareto solver

| Method | Its premise | Why it doesn't fit what we measured |
|---|---|---|
| **PCGrad** | Projects away the conflicting component of $g_i$ onto $g_j$ whenever $\cos(g_i,g_j) < 0$ | We measured $\cos(g_{Dice},g_{FT}) > 0$ always, $\cos(\cdot,g_{Evid}) \approx 0$ always — the negative-cosine trigger condition essentially never occurs. PCGrad would rarely-to-never activate on this model. |
| **GradNorm** | Learns per-**task loss weights** by matching each task's gradient norm to a common target, scaled by relative training-rate differences | Operates at the loss-weight level across nominally-independent task towers. Our network isn't independent towers — branches share a trunk. GradNorm also has no mechanism analogous to Sec 3.3's self-relative anomaly damping; it targets a fixed cross-task ratio, not a branch's own temporal stability. |
| **MGDA / Pareto solvers** | Solve a min-norm convex-combination problem to find a common descent direction when objectives genuinely trade off | Presupposes a real trade-off (improving one objective costs another). We found the two branches statistically independent (orthogonal gradients), not trading off — there is no Pareto frontier to search here. |
| **Gradient clipping** | Hard cutoff at a fixed global norm | Discards magnitude information indiscriminately and uses one global threshold; Sec 3.3 instead uses a **per-branch, self-relative** threshold, preserving each branch's own normal operating scale. |

**The genuine novelty claim**: this is a branch-structure-aware controller that separates two previously-conflated phenomena — (a) *sustained* magnitude imbalance between branches that share a trunk (Sec 3.2), and (b) *transient*, branch-relative instability over training time (Sec 3.3) — and corrects each with an independent, continuously-adaptive mechanism, motivated directly by measuring both phenomena separately in the same model and finding a static, single-constant fix insufficient for either.

---

## 6. Open questions this design still owes the semester's Stage 3/4 work

Being direct about what's still unverified, not glossing over it:

1. **Does $\mu_s^{(t)}$ (shared-trunk norm) behave as a sane equalization target across a real, longer, better-converging training run** — not just the 25-29 epoch pilots? Needs checking before trusting Sec 3.2 in practice.
2. **Hyperparameter sensitivity**: $\beta, \gamma, \tau, W, \rho_0$ — five knobs. At minimum $\gamma$ (correction strength) and $\tau$ (anomaly threshold) need an ablation grid; the others can likely be fixed at reasonable defaults ($\beta{=}0.9$, $W$ ≈ one epoch's worth of steps) and justified rather than swept, to keep the ablation tractable in a semester.
3. **Does this succeed where the static 1/3 scale failed?** That comparison (static vs. this adaptive controller, under matched conditions) is the direct, necessary follow-up experiment — the whole design's justification rests on "static was insufficient," so the semester's central result should be showing whether adaptive is not.
4. **The convergence-regime risk already flagged in the EDI proposal** applies here too: if the baseline still doesn't learn the segmentation task at all on 9 patients, it will be hard to tell "the controller helped" from "nothing helps at this data scale." The multi-slice supervision fix (already validated in this investigation) is the proposed mitigation, exactly as scoped in the proposal's Risks section.

**Recommendation before writing code**: validate Sections 3.1–3.3 as a pure *measurement* pass first (log $\mu_b, \nu_b, \alpha_b, \delta_b$ without yet touching the gradient) — this reuses the already-built gradient-logging infrastructure almost unchanged, costs one afternoon, and tells you whether $\alpha_b$ and $\delta_b$ actually behave sensibly on real data before you commit to implementing the full merge-and-step loop.
