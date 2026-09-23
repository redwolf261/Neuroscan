# Phase E115: CausalX-Net-Motivated Causal Loss Term + PCGrad Gradient-Conflict Resolution

**STATUS: DESIGN ONLY, SAVED FOR LATER. NO CODE WRITTEN. NO TRAINING RUN.**
Per explicit user instruction (2026-09-09/10 session): "lets improve upon their
architecture, but lets not start just save it stored, i also want to see pcgrad."
This document exists to be picked up and executed in a future session, not to be
built now.

---

## 1. What this responds to

`CausalX-Net: a causality-guided explainable segmentation network for brain
tumors` (Suman Prakash, Kiran Rao, Jahir Pasha et al., **Frontiers in Medicine,
vol. 12, published 24 October 2025**, DOI 10.3389/fmed.2025.1693603) was read
directly and completely (not from a search summary) before this design was
started, per this project's own established discipline of verifying primary
sources (the same discipline that caught E98's misleading "+0.9% Dice"
post-processing paper).

**What the paper actually claims, verified against its own tables (not its
abstract's framing):**
- A "structural causal modeling" (SCM) layer -- a sparse, learnable DAG with
  domain-prior adjacency masking (e.g. blocking a biologically implausible
  T1-modality-to-enhancing-tumor edge) -- inserted on the bottleneck features
  of a standard 3D U-Net-style encoder-decoder (4 encoder levels + bottleneck +
  4 decoder levels, strided-conv downsampling, channel progression
  32-64-128-256-320).
- Loss: `L = L_Dice + alpha*L_CE + beta*L_Causal + gamma*L_Smooth`.
- The headline "92.5% Dice, +4.3% over baselines" is misleading when checked
  against the paper's own tables: 92.5% is specifically the WHOLE-TUMOR Dice
  for MEDIUM-sized lesions (15-40 cm^3) in a size-stratified table, not an
  overall figure; the real comparison against their actual cited baseline
  (SwinUNETR) shows +0.8% to +1.3% per BraTS region (Table 4), with proper 95%
  BCa bootstrap CIs and paired t-tests (p<0.001) -- genuinely rigorous FOR
  THAT comparison, but nowhere near "4.3%."
- **No ablation isolates the SCM/causal layer's own contribution** -- beta=0
  is never tested against beta>0 in their own reported experiments.
- **No seed count is reported anywhere** in the paper.
- Small-lesion performance IS reported and IS a real weak point (Micro <5cm^3:
  ET Dice 77.8%+/-2.1) but the paper **never diagnoses why** -- no discussion
  of information loss in their own 4 strided-pooling stages (16x spatial
  reduction) despite this being the most obvious architectural place to look.
- Their own "causal" analysis targets MODALITY-LEVEL and REGIONAL
  interventions (do(X_modality=0), regional feature clamping) -- e.g. finding
  T1CE dropout costs 5.3% ET Dice -- not internal bottleneck/pooling
  mechanism causal tracing. This is a fundamentally coarser, different kind of
  "causal" analysis than this project's own E48-E114 chain.

**The actual gap this motivates (not a guess -- a documented absence in a
specific, dated, peer-reviewed paper):** CausalX-Net demonstrates that a
causal-modeling addition to a standard U-Net-style segmentation backbone can
plausibly help (though not decisively, given the missing ablation), on the
SAME kind of architecture and the SAME general problem (BraTS brain tumor
segmentation, small lesions underperforming) this project has spent E48-E97
causally diagnosing in exhaustive, verified detail. CausalX-Net's authors did
not have -- and did not attempt to build -- a causal account of WHY their own
architecture's small lesions underperform. This project already has exactly
that account: MaxPool3d(2,2) between enc3 and the bottleneck causally,
verifiably, specifically destroys disproportionately more small-lesion
information than large-lesion information (E92, cross-checked exactly against
E91's probe-decodability localization), and this loss is not recoverable from
wider linear context downstream (E94) -- a mechanism CausalX-Net's own paper
never investigates despite reporting the exact same symptom.

## 2. The proposed intervention (design sketch, not yet built)

Add ONE new loss term to this project's own canonical architecture
(UNet3D_v5 family, the same backbone used throughout E44-E114), inspired by
CausalX-Net's `L_Causal` term in FORM (a causally-motivated auxiliary loss
added to the standard Dice+CE+evidential composition) but grounded in THIS
PROJECT'S OWN VERIFIED MECHANISM rather than CausalX-Net's under-validated
modality-causal term:

- **Candidate L_Causal_ours**: an auxiliary loss term computed specifically
  over the population/voxels this project has already causally localized as
  information-scarce -- i.e., conditioned on the ALREADY-VALIDATED N_b
  (per-subject bottleneck causal-necessity measure, E48) and/or the
  MaxPool3d-specific information-loss localization (E92) -- NOT a
  re-derivation of any of the twelve already-KILLED size/necessity-conditioned
  losses (CCABA, IECG, CCAG, ASR, E93, E97, and others), which all failed
  for reasons already diagnosed in this project's own memory. The specific
  functional form of this term is NOT yet designed -- this is the open
  design question a future session must resolve BEFORE writing any code,
  explicitly reviewing why each of the twelve prior attempts failed so this
  is not a thirteenth repeat of the same idea in different clothing.

- **The genuinely new ingredient, not present in any of the twelve prior
  attempts**: NONE of E44-E97's loss-modification attempts ever checked
  whether their new loss term's gradient CONFLICTS with the main segmentation
  loss on shared parameters. E66 (Gradient Topology Audit) checked this for
  the EXISTING two-term loss (FocalTversky/main-seg vs EvidentialBeta) and
  found clean cooperation everywhere (cosine similarity +0.845 encoder,
  +0.813 bottleneck, +0.463 decoder) -- correctly concluding gradient surgery
  (PCGrad or similar) was unnecessary for THAT composition. But E66 never
  re-ran this check for any of the NEW loss terms added afterward (CCABA's
  amplification loss, IECG's sensitivity loss, ASR's reweighting, etc.) --
  this is a real, disclosed gap in this project's own diagnostic coverage,
  not an assumption.

## 3. PCGrad integration (Yu et al., NeurIPS 2020, "Gradient Surgery for
   Multi-Task Learning") -- what it is and exactly where it would apply

For two loss terms L_main (segmentation) and L_causal_ours (the new term),
with gradients g_main and g_causal on SHARED parameters (encoder/bottleneck,
where both terms' backward passes overlap):

```
if dot(g_causal, g_main) < 0:          # gradients CONFLICT
    g_causal_projected = g_causal - (dot(g_causal, g_main) / |g_main|^2) * g_main
else:
    g_causal_projected = g_causal      # no conflict, no surgery needed
final_gradient = g_main + g_causal_projected
```

(Symmetric projection of g_main onto g_causal's normal plane, per PCGrad's
own published algorithm, when there are more than 2 tasks and random pairing
order matters; with exactly 2 terms the order reduces to this simpler form
without loss of generality per the original paper's own 2-task special case.)

**Pre-registered diagnostic BEFORE any training (reusing E66's own exact
methodology, zero additional training cost):** compute cos(g_causal_ours,
g_main) at the SAME checkpoint(s) and depths E66 already used, for the NEW
candidate loss term once its functional form is finalized.
- If cosine similarity stays clearly positive (matching E66's own finding for
  the existing loss pair): PCGrad is unnecessary, matching E66's own honest
  null -- do NOT add gradient-surgery machinery to a loss pair that doesn't
  need it, per this project's own preference for the simplest sufficient
  intervention.
- If cosine similarity is negative or near-zero (conflict): PCGrad is
  well-motivated and should be implemented; this would also itself be a new,
  reportable finding (this project's first documented instance of genuine
  gradient conflict, if found), distinct from and complementary to E66's own
  null result on the pre-existing loss composition.

## 4. Explicit risks and required checks before committing further effort

1. **The twelve-failed-mechanisms review.** Before finalizing L_Causal_ours's
   functional form, explicitly re-read CCABA/IECG/CCAG/ASR/E93/E97's own
   memory files and confirm the new term is NOT mechanistically equivalent to
   any of them under a different name. This project's own established
   discipline (E97's explicit lesson: "do not infer the next intervention
   from the location of a failure once a straightforward recovery attempt
   there has failed") applies directly here.
2. **Novelty audit of "causally-motivated auxiliary loss + PCGrad" as a
   combination**, once the specific loss form is fixed -- NOT yet done. Given
   PCGrad itself is a well-established, widely-used technique (NeurIPS 2020,
   thousands of citations), and causally-motivated auxiliary losses are
   individually explored in prior art (E33's own novelty audit already found
   Component-Adaptive Tversky as an adjacent occupied technique), the novelty
   claim -- if any survives -- would need to rest specifically on the
   COMBINATION being applied to THIS project's own verified causal mechanism,
   not on either ingredient alone. This audit must happen BEFORE claiming
   novelty, matching this project's own repeated E23/E67/E68-style
   self-correction discipline.
3. **CausalX-Net's own code is not confirmed public.** No repository link was
   found or verified during this design pass. Any claim of "improving upon"
   CausalX-Net must be based on the MECHANISM/GAP described in their paper
   (verified directly, as done above), not on their actual codebase, unless a
   public implementation is later located and verified.
4. **Zero-training gate first, matching this project's own universal
   practice**: the gradient-cosine diagnostic in Section 3 must run and pass
   (or fail informatively) BEFORE any training campaign is launched, exactly
   as E95's zero-training diagnostic gate preceded E96, and E99's zero-training
   gate preceded any LCA training.

## 5. Explicit non-goals for this document

This document does NOT commit to:
- A specific mathematical form for L_Causal_ours (open question, Section 2).
- Any claim that this WILL clear the +1.0pp Dice bar (twelve prior attempts on
  the same architecture family have not; per the user's own separate decision
  to try a different architectural lever, this stays within the SAME backbone
  family with an ADDED loss term + gradient-surgery, which is a narrower
  change than a full backbone swap -- this tradeoff should be revisited
  explicitly when work resumes).
- A timeline or resource commitment. This is saved, not scheduled.
