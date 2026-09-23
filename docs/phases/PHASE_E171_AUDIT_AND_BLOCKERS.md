# E171 — Demand-Gated Task-Sensitive Transport: audit and blockers

**Date**: 2026-09-16
**Status**: NOT IMPLEMENTED. Two blockers identified before writing code, per instruction.
**Verdict**: 🟡 mechanism-family OCCUPIED; one axis possibly differentiated; **one required
component does not exist**.

---

## Step A — repository inspection (read from disk, not recalled)

| Artefact | Path | State |
|---|---|---|
| E165 R\* measurement | `e165/run_e165_per_stage_rank.py` | present; SVD truncation, dyadic grid, threshold 0.90 vs own undegraded prediction |
| E165 results | `e165/E165_per_subject.json` (n=125) | present |
| E169 features + predictor test | `e169/run_e169_rstar_predictability.py` | present; 77 features, 5-fold by subject, seeded |
| E169b nested controls | `e169/run_e169b_nested_volume_controls.py` | present |
| Frozen R_hat predictor | `e170/rhat_enc1_predictor.pkl` | present; OOF R²=0.632, ρ=0.787, MAE 4.45 ranks |
| E170 training harness | `e170/train_e170_tdm.py` | present but **NOT RUNNABLE** — Blocker 2 |
| Baseline harness | `e141/train_e141_evidence.py` | present; 3-arm structure, AMP, accum, bn_momentum=0.01 |
| Model | `neuroscan_3d_v5.py` | `UNet3D_v5(in_channels=4, out_channels=3)` |
| Dataset | `Dataset/brats_multimodal_dataset.py` | val = full native volume; train = 128³ foreground-biased patches |

### A definition mismatch that must be resolved before any E171 code

Three different "effective rank" computations exist in this codebase:

| Where | Method | Scope |
|---|---|---|
| E165 `R*` | **SVD truncation**, smallest dyadic r with ≥0.90 agreement | full native volume |
| E169 `eff_rank` feature | **entropy** exp(H(p)) of the spectrum, 20,000-voxel subsample | per sliding-window tile, averaged |
| E170 `effective_rank()` | entropy exp(H(p)), **8,192**-voxel subsample | per training patch |

$R^*$ and $R_{\mathrm{eff}}$ are therefore **not the same functional**, and the training-time
$R_{\mathrm{eff}}$ uses a different subsample size from the one the predictor was fitted against.
The deficit $[\widehat R^* - R_{\mathrm{eff}}]_+$ subtracts two differently-defined quantities.

Not fatal — E170's premise test found ρ($R_{\mathrm{eff}}$, $R^*$) = −0.09 at enc1, i.e. near
independent, which is *why* the deficit is non-trivial rather than algebraic — but it must be
stated, and the 8192-vs-20000 subsample mismatch unified before training.

---

## Blocker 1 — a label-free $d_l$ DOES NOT EXIST in this repository

Searched. E15's direction is explicitly GT-derived:

> "For each voxel, let `mu_opp` be the same-volume centroid of the opposite **ground-truth**
> class." — `PHASE_E15_EVIDENCE_FREEZE.md:18`

and that same document forecloses precisely the move E171 proposes:

> "No decoder-calibration method should be derived from E15 without a separate, non-oracle
> scientific basis." — `:55`

**No candidate 1/2/3 implementation exists.** E171 cannot be written without first constructing
and validating a label-free direction — an experiment in its own right, not an implementation
detail.

**And the readout geometry argues the obvious fallbacks collapse.** The segmentation head is
$D(z) = \sigma(w^\top z)$, so

$$\nabla_z D = D(1-D)\,w$$

is **rank-1**: any "prediction-gradient direction" at the readout reduces to $\pm w$, and the
*sign* is exactly what ground truth supplied in E15. A self-supervised objective
$\mathcal L_{\text{self}}(\hat y)$ (entropy, confidence) yields a direction that **sharpens
existing predictions** — it cannot correct a confidently wrong one. That is the regime that
matters here: E133 found 5/8 ET failures predict **exactly zero** voxels at max probability
0.0000. Sharpening zero gives zero.

Candidate 2 (Jacobian subspace) is the only one not obviously collapsing, because
$\partial \hat y / \partial z_l$ at $l=$ enc1 passes through the whole decoder and is not rank-1.
It is also the most expensive and is unvalidated.

## Blocker 2 — `train_e170_tdm.py` draws R_hat from a DISTRIBUTION, not per subject

`rhat_for_batch()` samples $\mathcal N(\mu,\sigma)$ because the 77 features require a
**full-volume** forward pass unavailable inside a 128³ patch loop. As written, the "true" arm
assigns random R_hat and is therefore **indistinguishable from the shuffled arm by
construction**.

Fix is as specified in §7 of the brief: one intact full-volume pass per training subject, cache
`sid → R_hat`, look up during patch training. ~2h GPU, one-time, reusable across arms and λ.
**Until then no arm comparison is meaningful.**

---

## Step D — exact-mechanism prior-art audit

Audited the chain, not the ingredients:

> instance-specific predicted task-required rank → effective-rank deficit → conditional gate →
> task-output-sensitive representation transport

### Finding: the WHEN × WHERE structure is an established, named subfield

**Conditional activation steering** does exactly "gate whether to inject a direction into a
representation, per input":

- **CAST** — conditional activation steering; decides *when* to intervene.
- **GAPS** (arXiv 2609.01878) — states the field's own organising axis verbatim: *"Recent
  conditional methods like CAST and DSAS decide **when** to intervene, but apply the full dense
  vector to all dimensions regardless of concept information. GAPS introduces dimension-level
  conditioning... decides **which** neurons to intervene on."*
- **GSS** (2602.08901) — gated subspace steering; conditional suppression of components.
- **DSAS**, **GCAD**, flow-based activation steering (2605.05892).

So $\tilde z = z + \beta\, g\, d$ with a per-input gate $g$ is **not a new computational form** —
it is the standard conditional-steering equation, and E171's "WHEN × WHERE" framing is that
subfield's organising axis stated in its own words.

### What is not found

No located work gates steering by a **predicted representational-capacity deficit**. Every gate
found is concept- or behaviour-detection based (is this input toxic / memorised / on-concept),
not capacity based. Rank-1 steering-cost geometry (2605.16362) is adjacent but concerns search
budget, not capacity gating.

### Honest classification

| Component | Status |
|---|---|
| Conditional per-input gated activation steering | 🔴 **OCCUPIED** (CAST / GAPS / DSAS / GSS) |
| Effective-rank deficit hinge penalty | 🔴 **OCCUPIED** (NeurIPS 2024 eff-rank regularisation; E170b) |
| Adaptive / task-adaptive rank allocation | 🔴 OCCUPIED |
| **Gate signal = predicted task-required-rank deficit** | 🟢 not found |
| Domain: 3D medical segmentation (steering literature is ~entirely LLM) | 🟡 thin, but a domain transfer is not a principle |

**The mechanism family is occupied; only the *gating signal* is possibly new.** That is one axis,
and by this project's own standard (E161, E166, E168) a single differentiated axis inside an
occupied mechanism has not previously survived as a novelty claim.

Per the brief's rule, this is **not** an exact-match kill: no located paper performs the full
chain. It is a 🟡 requiring the four-part justification — what is known, what combination is new,
why it is not trivial juxtaposition, what experiment distinguishes it. Parts 1 and 2 are answered
above. **Part 3 cannot yet be answered, because Blocker 1 means the mechanism cannot be built.**

---

## What is required before E171 is worth compute

1. **Construct and validate a label-free $d_l$** as a standalone, falsifiable experiment, with a
   pre-registered check that it is not a disguised confidence-sharpening direction — it must do
   something where the model is confidently wrong.
2. **Run the per-subject R_hat precompute** (~2h) so any arm comparison is meaningful.
3. **Unify the $R_{\mathrm{eff}}$ functional** (subsample size; entropy vs truncation).
4. Success criterion is **true > shuffled AND true > constant** — beating baseline alone
   replicates occupied prior art.

## Recommendation

Do **not** write E171 execution code yet. Blocker 1 is not an implementation gap but an unsolved
research question, and the rank-1 readout geometry argues the obvious solutions collapse to
$\pm w$ with a sign only ground truth supplies.

Correct ordering: **solve $d_l$ first**, as its own experiment. If no label-free direction
survives, E171 is unbuildable regardless of the audit verdict.

## Sources

GAPS (arXiv 2609.01878) · GSS (2602.08901) · Flow-based activation steering (2605.05892) ·
Rank-1 steering geometry (2605.16362) · Pre-intervention prediction of SAE steering side effects
(2606.08365) · Minimizing collateral damage in activation steering (2605.01167)
