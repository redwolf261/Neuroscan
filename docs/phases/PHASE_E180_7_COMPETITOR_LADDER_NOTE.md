# E180 Stage 7 — competitor ladder: novelty-audit addendum

**Date**: 2026-09-17
**Status**: Addendum to the Stage 7 design (competitor hierarchy), fixed before Stage 7 runs.
**Depends on**: `PHASE_E180_NOVELTY_AUDIT.md`.

---

## Why this addendum exists

The novelty audit found the application shape of E180 (instability/uncertainty signal →
prioritized test-time refinement in segmentation) densely occupied by an active 2025 literature
(TRUST, CertainTTA, and the broader TTA-for-segmentation subfield), and all of it selects using
**generic predictive entropy / output uncertainty**, not a controlled representation-space
transform validated against a restoration counterfactual.

Stage 7's competitor ladder already includes a Level 3 uncertainty competitor
($H(p_i)$, via `StageFeatures.pred.maxp_*`/`meanp_fg_*`). This addendum elevates that competitor's
role: it is not merely one baseline among several, it is **the literature-matched signal** that
any future novelty claim must show Γ is not redundant with. Stage 8's incremental-$R^2$ test
therefore does double duty — it is both the scientific decision gate for $H_{180}$ and the direct
test of whether Γ carries information beyond what the occupied entropy/uncertainty-based
literature already provides.

## Binding addition to Stage 7

- Report the Level 3 uncertainty competitor's standalone predictive power for Δ (Spearman,
  Capture@B) with the same prominence as Γ itself, not folded silently into the ladder.
- In Stage 8's nested regression, report the **specific** increment $\Delta R^2$ of adding Γ
  *after* uncertainty is already in the model, as its own named quantity — not just the pooled
  increment after magnitude+uncertainty+sensitivity together. This isolates the exact comparison
  the novelty audit's narrow claim depends on.
- If Γ's increment over uncertainty alone is negligible, the honest conclusion is that E180's
  signal — even if Γ→Δ holds — is likely a re-derivation of existing entropy-based TTA-selection
  signals, and any Stage 12+ algorithm would need to argue differentiation on grounds other than
  the selection signal itself (e.g. the restoration-counterfactual validation methodology, not
  the signal).

## What this does NOT change

No change to Γ's construction, the locked probe severities, the tile ledger, or the two-stratum
(full 125 / E167 110) analysis hierarchy already locked in `PHASE_E180_5_5_PIPELINE_GATE_PREREG.md`.
This is a reporting emphasis within the already-planned Stage 7-8 design, fixed now, before Stage
7 code is written, so it is not chosen post-hoc based on which framing looks better.
