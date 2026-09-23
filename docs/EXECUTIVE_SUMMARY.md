# NeuroScan Executive Summary

**Date**: 2026-09-20  
**Status**: FROZEN at E188. Latent-steering branch proven dead. All representation-space intervention axes occupied (E169–E176).

---

## One Sentence
A 2+ year causal-intervention empirical study hunting for a ≥1pp Dice improvement on brain-tumor segmentation; found none despite 188 phases, but mapped the failure structure in exceptional detail and validated three durable measurement contributions.

---

## The Core Problem
- **Task**: Binary whole-tumor segmentation (BraTS 2023 GLI, FLAIR only, 125-scan validation set)
- **Baseline**: Dice 0.9063 (pooled) / 0.8872 (per-subject, corrected E56)
- **Goal**: Find any modification clearing +1.0 percentage point AND justifiable as novel
- **Hardware**: NVIDIA RTX 5050 8GB, 30-epoch training, batch size 8

---

## What We Know Works

### Deep Supervision (D4, +0.33pp) — VALIDATED
- **Mechanism** (E36): Improves *quality* of small-tumor segmentation (1–50 voxel Dice +40% relative), not detection rate
- **Reproducible**: Across seeds, robust
- **Status**: Sub-1pp but mechanistically closed (no deeper lever found in E37–E42)

### Critical Resolution (α_c) — MEASUREMENT ONLY
- **Definition**: Per-tumor geometric property (at what resolution does it vanish?)
- **Validation**: Survived 500 permutation trials, subject-clustered bootstrap CI excludes zero
- **Status**: Predicts tumor difficulty beyond plain size, but no trained algorithm yet cleared the bar

### Bottleneck Size-Dependency (E48) — CAUSAL FINDING
- **Discovery**: Small lesions depend MORE on bottleneck, opposite the hypothesis (ρ=–0.454, p<0.001)
- **Validation**: Re-verified fresh (E85/E86, ρ=–0.3834, same direction/magnitude)
- **Status**: Motivated E49–E51 mechanisms (all failed or tied D4-only)

---

## The Failure Structure (What We Tried & Killed)

| Domain | Attempts | Status | Best Result |
|--------|----------|--------|------------|
| **Loss geometry** | EGGO-M, SC-TAM (16 phases) | 8+ diagnostics each, verified-but-null | 0.9030 (below baseline) |
| **Architecture** | D4+D8, attention gate, CCABA, IECG, combined (E44–E51) | 6 completed, 3 multi-seed checked | 0.9095 (tied D4-only) |
| **Objective** | ASR (E52–E53) | Smoke-test killed or incomplete | 0.8969 (below baseline) |
| **Resolution** | 96³, 128³, DTC, α_c training (E29–E34) | Naive resolution worse; DTC artifact; α_c not yet trained | No improvement |
| **Preprocessing** | Grid-entropy, transformation-error, bottleneck localization (E37–E43) | 7+ mechanistic tests, all permutation-controlled nulls | No improvement |
| **Latent steering** | E15 readout decomposition, local recoverability, reachability (E188) | Proven readout-mediated (34–55% energy, 5.2e−5 effect) | 0.8936 (below baseline) |
| **Representation space** | Rank, routing, commutator, capacity, compression (E113–E176) | All 8 novelty axes occupied, search terminated | No improvement |

**Result**: **Zero mechanisms cleared +1pp.** Three mechanisms (E49, E50, E51) properly multi-seed-checked all tie or trail D4-only.

---

## Three Methodological Contributions

1. **Multi-seed variance calibration** (E49→E50→E51)  
   - Single-seed headlines are favorable noise (~+0.20pp average overestimate)
   - Mandatory policy: ≥3 seeds for any Dice claim

2. **Pooled-vs-per-subject Dice correction** (E56)  
   - Systematic 1.91pp inflation in pooled Dice
   - Canonical baseline should be 0.8872 per-subject, not 0.9063 pooled
   - Corrected +1pp target: 0.8942, not 0.9163

3. **Causal-diagnostic methodology** (E43→E47→E48)  
   - Genuine causal nulls using trained-model interventions (not correlational proxies)
   - Caught and fixed sign-convention bugs inline (E25)
   - Permutation safeguards catch artifacts other tests miss (E30–E34, E32)

---

## Why We Stopped

### Representation-Space Interventions: 8 Axes All Occupied
- **E169–E176**: Systematic novelty search across architecture, objective, representation, adaptive-compute, capacity, compression, cross-domain, evaluation
- **Result**: Every axis found to be addressed by 2025–2026 literature
- **Decision**: Search terminated; no differentiated principle found

### Latent Steering: Proven Readout-Mediated (E188)
- **E15 readout decomposition**: seg_head is rank-3; projecting onto full row space, D(z+d_parallel) reproduces D(z+d_E) to max 7e−5 while D(z+d_perp) = 5.2e−5 (NOTHING), despite d_perp carrying **34–55% of energy**
- **Finding**: At dec1, "behavioral quotient of z is Wz"; everything in ker(W) is invisible
- **Decision**: Latent-steering line closed. Next frontier (if any) requires prior-art audit on "upstream → nonlinear decoder → altered reachable output, not reducible to readout steering" BEFORE code

### Multi-Seed Variance Killed Single-Seed Headlines
- **E49**: Single seed +0.51pp, 3-seed mean +0.31pp (tied D4-only, CI crosses zero)
- **Retroactive**: E45 (+0.47pp) and E46 (+0.39pp) were single-seed flukes
- **Decision**: Only properly multi-seed-validated results count; none beat D4-only

---

## Standing Recommendation
**STOP mechanism search. WRITE UP:**
1. Causal-diagnostic methodology chain (E43→E47→E48: genuine causal nulls + counterintuitive small-lesion finding)
2. Multi-seed variance correction methodology (E49→E50→E51: legitimate contribution)

**If continued**: Requires prior-art audit before code, not after.

---

## Validated Reusable Findings (No Method)

| Measurement | Source | Evidence | Use |
|-------------|--------|----------|-----|
| D4 mechanism | E36 | Size-graded breakdown: 1–50 voxel +40% Dice, vanishes >1000 voxels | Baseline for future work |
| t1c modality causality | E142 | Modality ablation: t1c→0 means ET Dice 0.8433→0.0015 | Explains E137/E140/E141 nulls |
| Stage-specific rank | E169b/E174 | R²=0.518 enc3 vs 0.004 dec1 (size-controlled) | Identifies structural asymmetry, no exploitation found |
| Pooled-vs-per-subject gap | E56 | 1.91pp systematic inflation | Correction to all future Dice claims |

---

## Code Artifacts

**Active**:
- `/neuroscan_3d_v*.py` (v1–v9 architecture lineage, git-tracked)
- `/experiments/exp_e12_eggo_m/` (main hub: e100–e198+ subdirs with cached JSON results)

**Frozen Results** (e.g., `E192_Z_cache.json`, `E193_Z_TC_cache.json`):
- Representation snapshots, cross-model audits
- No re-training needed for audits

**Orphaned** (do NOT cite):
- `E151_dec3_sweep.json`, `E152_resolution_recovery.json`, `E153_responsibility.json` — no generating script, definitions unrecoverable

---

## Timeline

- **E1–E43** (~4 months): EGGO-M margin-loss family + deep supervision discovery
- **E44–E51** (2 weeks): Architecture search post-pivot (CCABA, IECG, attention, combined)
- **E52–E97** (1+ month): Objective-level and bottleneck diagnostics
- **E98–E129** (2+ weeks): Search within capacity/rank space
- **E138–E153** (2+ weeks): Readout frontier, contrast interventions
- **E169–E188** (2 weeks): Prior-art search + latent-steering closure

**Total**: 188+ phases, 2+ years elapsed

---

## If Asked "Why Didn't It Work?"

The project is NOT a failure — it's an exceptionally detailed **empirical map of one model's failure structure**. Key findings:

1. **Small tumors are the bottleneck**, but they're attacked at three wrong levels:
   - **Preprocessing**: Median native tumor vanishes at 64³ (E29) — information loss pre-model
   - **Representation**: D4 improves their quality, but not enough (+0.33pp)
   - **Readout**: Terminal steering is mediated by fixed row space (E188-F) — no hidden capacity

2. **Causal mechanisms are fragile**: E49's +0.51pp single-seed headline → +0.31pp true effect (20pp overestimate from noise)

3. **Representation-space levers are saturated**: All 8 novelty axes occupied (E169–E176), no differentiated principle found in 2025–2026 literature

4. **The +1pp bar is hard**: D4 (+0.33pp) is the only validated improvement across 8 architecture families, 2 loss formulations, and 3 objective-level levers

---

## What's Different About This Project

- **Causal rigor**: Trained-model interventions, not correlational diagnostics
- **Bug catching**: Sign-convention errors, shared-tensor reuse, precision artifacts caught and fixed inline (not silently)
- **Permutation safeguards**: Randomization tests catch mathematical artifacts (E30–E34, E32)
- **Honest reporting**: Contradictory results investigated (E18-followup refutes E18's own causal claim); single-seed overestimates disclosed (E49→E50→E51)
- **Audit discipline**: Comprehensive project audit (E27), code integrity checks (E21.5), measurement audits (E56)

---

**This project succeeded at understanding why it failed. That is a contribution.**
