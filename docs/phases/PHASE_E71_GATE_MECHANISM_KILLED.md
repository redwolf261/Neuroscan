# Phase E71 — Gate Mechanism Premise Test: KILLED

## Result

The smallest discriminating experiment for CDCG's gating mechanism (Section
3.3 of the design doc) — inference-only, no training, testing the mechanism's
own core premise before ever touching the learned predictor $\hat{d}_i$ —
**fails cleanly and significantly, in the direction opposite to what the
mechanism requires.**

| Quantity | Value |
|---|---|
| Spearman(measured d_i, suppression-tolerance slope) | **−0.290** |
| Parametric p | 1.0×10⁻³ |
| Permutation p | 0.001 |
| Required sign for the mechanism to work | **positive** |

## What was tested

Per the Observe→Hypothesize→Attempt→Test directive: before training any full
gated model (which would conflate the gate's own correctness with training
dynamics and loss-weight choices), this experiment isolated a single question —
**is it even true that bottleneck-dependent subjects tolerate having their
`enc1` skip suppressed better than bottleneck-independent subjects?** — using
the real, measured $d_i$ (not the learned $\hat{d}_i$), so a failure here
cannot be blamed on prediction fidelity.

Method: for each of the 125 validation subjects, `enc1`'s contribution to the
decoder was suppressed by a swept factor $s \in \{0, 0.25, 0.5, 0.75, 1.0\}$
(verified: $s{=}0$ reproduces `model.forward()` exactly, 0.0 max abs diff), and
a per-subject linear slope of Dice vs. suppression strength was fit.

## Result and its meaning

The correlation is real, significant, and **backwards**: subjects with
*higher* bottleneck-ablation sensitivity ($d_i$) show a **steeper** (more
negative) Dice-vs-suppression slope — i.e. they are hurt **more**, not less,
by having their skip content suppressed.

**Diagnosis**: bottleneck-dependence and skip-dependence are not two
substitutable, opposing resources the network can trade off per-subject (which
is what the gating mechanism's design implicitly assumed). They appear instead
to be **correlated indicators of overall subject difficulty** — a subject
whose segmentation is fragile to bottleneck ablation is also, independently,
fragile to skip suppression. Removing either major pathway hurts hard
subjects more than easy ones; there is no compensatory relationship between
the two pathways for the gate to exploit.

## Decision

Per the design doc's own pre-registered rule and the project's own
no-rescue discipline: **the gating mechanism (Section 3.3) is KILLED.** This
holds independent of predictions 1 and 2's earlier results (both of which
concerned whether $\hat{d}_i$ predicts $d_i$ well, not whether acting on that
prediction via suppression is the right thing to do). Do not proceed to
training a full end-to-end gated model — the premise the mechanism needs has
been tested directly and found false, with a clean, significant, correctly-
signed-in-the-wrong-direction result.

## What survives from the CDCG line

- **Prediction 1** (the network's bottleneck representation can predict its
  own causal ablation sensitivity) remains a real, validated finding —
  Spearman 0.87 on held-out subjects, with a strong signal beyond lesion size
  (partial ρ=0.855). This is a genuine, if narrow, empirical result: networks
  carry recoverable information about their own causal fragility.
- **The specific USE of that signal proposed here (skip-suppression gating)
  is dead.** A different downstream use of $\hat{d}_i$ — one that doesn't
  assume bottleneck- and skip-dependence trade off against each other — would
  need its own fresh hypothesis and its own discriminating test, not a retry
  of this mechanism with different hyperparameters.

## Why this is a good result, not just a null

Per the directive's own framing: this experiment produced a **falsified
hypothesis with a diagnosed reason**, not merely "the number didn't improve."
It reveals something real about this network's failure structure — difficulty
is not partitioned into independently-tradeable capacity pools (coarse vs.
fine), it is a shared, correlated property across pathways. That is itself a
finding worth recording, and it closes this specific design direction cleanly
rather than leaving it ambiguous.

## Artifacts

- `experiments/exp_e12_eggo_m/e71/check_cdcg_gate_sweep.py`
- `experiments/exp_e12_eggo_m/e71/E71_gate_sweep_table.json` (125 subject records, full sweep)
- `experiments/exp_e12_eggo_m/e71/E71_gate_sweep_summary.json`
