# NeuroScan Project: Complete Understanding from Artifact Reading

**Date compiled**: 2026-09-20  
**Read**: All major synthesis docs, memory files, latest phase reports (E188 frontmost)  
**Status**: Project FROZEN at E188 after proving latent-steering branch dead. Prior major arc E169-E176 closed all representation-space intervention axes. Two prior sub-1pp positive findings: Deep Supervision (E25-E43 arc, +0.33pp) and E169b/E174 stage-specific rank predictability (durable measurement, no method).

---

## 1. THE TASK & SYSTEM

**Problem**: Automated binary whole-tumor segmentation (tumor vs. not-tumor) on BraTS 2023 GLI, FLAIR modality only.

**Data**: 1,251 scans total; split at project start into **1,126 train** / **125 validation** (never changed, reused for all decisions).

**Preprocessing**: Every scan resized to **64³ voxels** (aggressive ~3.75× × 3.75× × 2.4× compression).  
**Critical consequence**: ~65% of native tumor components are ≤5 voxels (likely annotation noise); median component vanishes entirely at 64³.

**Architecture**: 3D U-Net (standard encoder-decoder)
- Two independent output heads: (1) tumor probability (sigmoid), (2) evidential uncertainty (Beta distribution)
- Loss: 50/50 blend of Focal-Tversky + Evidential-Beta (frozen early)
- Training: 30 epochs, batch size 8, learning rate 0.0004, AdamW, cosine annealing
- Hardware: NVIDIA RTX 5050 8GB

**Baseline ("A"): Pooled Dice 0.9063** (seed 0 of a 4-seed frozen run).  
**Per-subject Dice**: 0.8872 (discovered E56, hidden by pooled-vs-per-subject gap of 1.91pp).  
**Project goal**: Find any modification that clears +1.0 percentage point and is justifiable as novel research.

---

## 2. MAJOR ARC SUMMARY (E1–E188)

### 2.1 E1–E43: EGGO/EGGO-M Arc (Pre-Pivot Diagnostics)

**Timeline**: ~4 months of hypothesis generation, mechanism design, and rigorous diagnostic chains.

**E1–E11: Literature + Design**
- Searched literature on MOO/gradient manipulation/evidential learning
- Internal representation audits found: boundary-distance is NOT the explanation for evidence; feature-norm contributes modestly; latent geometry shows sharp separability (AUC 0.9995); local density is independent signal
- Selected margin-loss design: pairwise-hinge formulation via De Brabandere et al., only candidate meeting three practical requirements (per-voxel, binary, no extra normalization)
- Result: EGGO-M locked design before any training

**E12–E24: EGGO-M Trained & Diagnosed** (~8 separate diagnostic phases)
- **E12 pilot**: Mechanism works (margin loss active), Dice flat
- **E13 multiseed**: 4-seed confirmation — EGGO-M statistically worse than baseline (mean –0.37pp)
- **E14–E21**: Eight+ follow-up diagnostics (gradient conflict, decoder sensitivity, reachability, target stability, freezing ablations, transport, implementation audit) — all ruled out competing explanations, **none found a fixable cause**
- **E22**: Causal test — local descent along margin-loss gradient is actively *harmful* to Dice; wrong-signed (r=+0.243, p=0.0007); E15's "useful" direction beats it 192/192 (orthogonal finding)
- **Verdict**: Margin-loss line CLOSED as verified-but-null

**E23–E24: SC-TAM Redesign**
- Attempted to fix E22's wrong-direction finding via task-aligned projection + confidence gating
- **Result**: SC-TAM mechanistically correct (98% of false negatives move correctly), collateral damage to already-correct voxels outweighs benefit
- **C6-2/C6-3 Dice**: 0.9030/0.9026 — below baseline despite all internal checks passing
- **Critical finding from this arc**: E25's sign-convention bug fix (Step 5 of E25) showed the entire "mechanism inversion" narrative was a scoring-convention error in the diagnostics, not the code — real example of catching mistakes within the same analysis cycle

**E25–E26: Deep Supervision (The Only Positive Find)**
- **Idea**: Add auxiliary prediction heads at coarse decoder stages (D4 at 1/4 resolution, D2 at 1/2 resolution)
- **E36 autopsy**: NOT "improved detection of small tumors" (hypothesis killed, 22.0%→20.6%, worse); REAL mechanism: improves *quality* of already-detected small tumors — 1–50 voxel Dice +40% relative (0.186→0.261), vanishes >1000 voxels
- **Result**: D4-only = **0.9096** (**+0.33pp**), best of the deep-supervision family
- **E37–E42 mechanistic tests**: Grid-entropy fails, residual-gradient signal fails permutation, effective-rank fails, transformation-error fails, cross-scale operator false premise
- **Verdict**: D4 mechanism CONFIRMED as real but **causally closed** — no deeper lever found

**E27–E43: Preprocessing & Context Audits**
- **E27 project audit**: Corrected 30+ buried assumptions (no augmentation, no patch training, GPU model, 125-val reuse, pre-resize information loss)
- **E29 geometry audit**: Median native tumor component vanishes at 64³; resolution recovery monotonic for 5–150 voxel band (2–8× per doubling)
- **E30–E34**: Degradation-Trajectory Constraint (DTC) killed twice (zero signal after permutation safeguard, division-by-vanished-footprint artifact)
- **E32–E33**: Critical-resolution α_c survived 500 permutation trials, subject-clustered bootstrap (CI excludes zero), novelty audit clean
- **E34–E35**: Resolution experiments (96³ worse than 64³, 128³ BatchNorm at batch=1 unintelligible), scope-context field killed

**Result of E1–E43**: 
- One validated improvement: D4 (+0.33pp)
- One validated measurement: α_c (survives permutation safeguard, not yet a trained algorithm)
- **Zero mechanisms cleared +1pp bar**

---

### 2.2 E44–E51: Post-Pivot Architecture Search (User mandate: +1pp or stop)

**User instruction after E43**: Stop open-ended hunting. Take D4 + ONE modification → +1pp or close.

**E44 — RCGW (adaptive auxiliary weighting)**: 
- Killed twice (instability at raw form, null at saturating form)

**E45 — D4+D8 combined** (`UNet3D_v4`):
- Single seed 0.9110 (+0.47pp) — looks promising
- **Catches the multi-seed variance problem**: E49's 3-seed check retroactively invalidates this headline
- Policy established: **≥3 seeds mandatory going forward**

**E46 — Attention gate on enc1** (`UNet3D_v5`):
- Single seed 0.9102 (+0.39pp), gate learned real pattern
- Lies in same +0.3–0.5pp band as E45

**E47 — Causal audit of E46's gate**:
- Clamping gate at boundary vs. interior: no difference (drop_difference=+0.00025, p=0.808)
- **Confirms E43's correlational null causally**: boundary-routing is not the mechanism

**E48 — Bottleneck ablation (THE REVERSAL)**:
- Pre-declared prediction: severing bottleneck should hurt large lesions more (they lose signal there)
- **Actual result**: ρ(native_size, Dice_drop) = **–0.454** (p<0.001, n=125, opposite direction)
- **Finding**: Small lesions depend MORE on bottleneck, not less. They need global context more than local features can provide
- **This finding motivates every mechanism that follows** (E49–E51 all condition on size)

**E49 — CCABA** (Causally-Calibrated Adaptive Bottleneck Amplification, `UNet3D_v6`):
- First mechanism whose conditioning is a *measured causal curve* (fit directly to E48's ablation data)
- Single seed 0.9114 (+0.51pp) — headline looks best yet
- **3-seed check**: [0.9114, 0.9075, 0.9093], mean 0.9094, **CI [0.9046, 0.9142]** (crosses D4-only)
- **True effect: +0.31pp, statistically tied with D4-only**
- **Critical methodological finding**: Single-seed headlines are favorable noise. E45/E46's +0.47pp/+0.39pp were flukes relative to their true 3-seed means

**E50 — IECG** (live counterfactual gating, `UNet3D_v7`):
- More novel, bigger mechanism (on-the-fly bottleneck ablation replayed every step, learned to predict effect)
- 3-seed: [0.9081, 0.9030, 0.9084], mean 0.9065, **+0.02pp, worse than CCABA**
- Despite high internal consistency (gate learned real causal sensitivity), task Dice null
- **Evidence**: "more novel/ambitious" does not track "more effective"

**E51 — CCAG** (CCABA + attention gate, `UNet3D_v8`, pre-declared final single-arch attempt):
- All ablation-safety checks pass bit-for-bit (gate/CCABA/both)
- 3-seed: [0.9101, 0.9092, 0.9092], mean 0.9095, **+0.32pp** (tied CCABA, tied D4-only, p=0.919)
- Two mechanisms did NOT compose additively
- **Closes the architecture-level lever**

**E52–E53 — Objective-level (loss function)**:
- **E52**: ASR gradient-calibrated revisit of E34 (which had catastrophically collapsed). Fixed the 43× gradient blowup, smoke test still collapsed with different cause (loss direction fighting early curriculum, not magnitude). Killed at smoke test.
- **E53**: ASR + curriculum warmup (ramp in 0→full, epochs 8–20). Smoke test passes, seed 0 incomplete run scores 0.8969 (below baseline). User stopped seeds 1–2 mid-training.

**Result of E44–E53**: 
- Six mechanisms fully evaluated (E44, E45, E46, E49, E50, E51)
- None clears +1pp; three 3-seed-checked mechanisms all tie or trail D4-only
- **Recommendation strengthened**: Stop searching, write up causal diagnostics + variance-correction methodology

---

### 2.3 E54–E97: Diagnostic Chains (Bottleneck mechanism hunting)

**Aim**: E48 found small lesions depend on bottleneck; why and how to fix it?

**E54 (A96)**: Whole-volume 96³ training (batch=2 physical, grad-accum to 8). Single seed 0.9066 (essentially D4-only); per-subject 0.8981 (+1.39pp). Did not proceed to 3-seed confirmation per pre-declared criteria (headline marginal, not decisive).

**E55 (dual-resolution local refinement)**: Differentiable soft-centroid crop + fusion, motivated by E54 (flat result) + E48. Single seed 0.9007 (below baseline, below E54), fusion gate dialed back over training. Mechanism worked as designed, didn't help task.

**E56 (measurement audit — CRITICAL)**: 
- Discovered **pooled Dice inflates 0.85–2.40pp** over per-subject Dice
- Canonical "0.9063" baseline was seed 0 of a 4-seed run, never variance-checked
- **True baseline (4-seed mean) is 0.9044**, CI [0.9007, 0.9080]
- **Corrected +1pp target: 0.8942 per-subject**, not 0.9163 pooled
- Re-scored 14 existing checkpoints: E45 & E54 both already clear corrected target on single seed
- **This is a legitimate methodological contribution**, not just internal practice

**E58–E60: Bottleneck localization**
- **E58 causal characterization**: spatial non-localization (octants don't reproduce size-signature); donor-substitution effect later explained as OOD confound (Li & Janson NeurIPS 2024 published confound)
- **E59 coalitional interaction**: synergy for small lesions (p=0.0003), redundancy for large (p<0.0001); Spearman ρ(size, I_ij)=–0.498
- **E60 feasibility (DBC)**: small-lesion synergy doesn't reproduce with balanced partition sampling (p=0.530); large-lesion redundancy does. Killed as zero-cost pre-training kill.

**E61 (novelty search + dimensionality audit)**:
- 4 of 5 candidate mechanism directions occupied (FineRS/C2FNet, ReSeg-UNet/MSFB-Net, SI²CRL, BRDG)
- Effective-rank hypothesis fails: large>small, opposite hypothesis, controlled for size (p=0.45 after control). All 7/7 GO criteria failed.

**E62–E65: Pooling mechanism**
- **E62 INVALIDATED**: original E62 claimed MaxPool3d position-discarding hurts small lesions. **Shared-tensor bug**: forward_from_enc1 reused enc1 for both pool1's input AND decoder skip, so 100% of measured effect came through skip (identity), not pooling. Found by user, not caught by sanity checks.
- **E64 fix + rerun**: skip-connection bug confirmed and fixed. Intervention scale corrected (0.02%→whole-volume). Pattern: split-intervention skip confirmed (translation dominates).

**E66–E70: Signal directions and CAS mechanism**
- **E66 gradient-topology**: killed
- **E68 offset-discrepancy**: inconclusive
- **E69 offline mining**: mixed  
- **E70 CAS (Correspondence-Aware Skip)**: identifies magnitude-gated smoothing as real mechanism, not correspondence correction. +0.30pp in ablation.

**E71**: CDCG (per-sample capacity gating, predicted size relationship). Prediction 2 **FAILS** — inverts E48's relationship. Kills the latent-steering branch before it could be climbed.

---

### 2.4 E113–E129: Capacity/Effective-Rank Arc (No method cleared bar)

**E126 (effective rank causally drives N_b)**: Strongest validated finding in this branch, but no clear algorithmic lever. Subsequent E127–E129 tests found no training modification that exploited this finding to clear +1pp.

---

### 2.5 E138–E153: Prior-Art Audit & Readout-Frontier Arc

**E138 (prior-art audit)**: FiLM-for-contrast-enhancement IS published (arXiv 2511.16498); quantitative label-blind diagnosis survives.

**E139–E150**: Contrast-intervention family (conditional readout, per-voxel boundary shift, evidence-conditioned supervision, recoverability, representation demand, linear probe gap).
- **E142**: t1c modality ablation — **causally proves** t1c owns ET (t1c→0 means ET Dice 0.8433→0.0015)
- **E147**: Representation demand INVERSE — high recoverability needs LESS capacity, allocation hypothesis dead
- **E148**: Frontier at readout — linear probe beats trained head by +0.0388 AP
- **E150**: Readout gap real but inert — AP gains don't imply Dice gains

---

### 2.6 E151–E188: Latent Steering → Gate F Frozen

**E151–E153**: Orphaned JSONs (E151 dec3_sweep, E152 resolution_recovery, E153 responsibility) — no generating script, no phase doc. E151 appears to be a geometry-vs-Jacobian comparison explicitly retired before compute. **Do not cite**.

**E169–E176**: Search terminated. All 8 novelty axes occupied (architecture/objective/representation/adaptive-compute/capacity/compression/cross-domain/evaluation). E169b/E174 durable finding: R²=0.518 enc3 vs 0.004 dec1 (stage-specific rank predictability).

**E178–E188**: Latent-steering line closed.
- **E178 (Task-Demand Routing)**: NeuroScan Dice ≤ Constant-target; sits middle of ranking. Killed Gate 1.
- **E179 (Geometric Residual)**: R(x)=‖∇p‖/(|p−τ|+ε) beats boundary-proximity only 22% of bins (target 80%). Killed Gate 2 decisively.
- **E188 (E15 readout confound proof)**: 
  - **Record correction**: Original E15 (+0.0393, monotonic α=14) on retired E12f checkpoint; doesn't reproduce on current E131_v5 (peak +0.0139 α=8, non-monotonic, collapses by α=28)
  - **Experiment A (local recoverability)**: 32 norm-matched random directions never beat native Dice (40/40 cells headroom R<0); E15 at 100th percentile (38/40). No recoverable neighborhood, one direction.
  - **Experiment F (readout decomposition)**: seg_head is rank-3 Conv3d; full-rank projector P_W gives **max|D_full − D_parallel|=7.0e−5** while **D_perp=5.2e−5 MAX despite 34–55% energy**. Half the intervention does nothing.
  - **Gate F (durable output)**: Any candidate's benefit must not reduce to Δ Dice(δ_l) ≉ Δ Dice(P_W Δz). Cost ~280 forward passes, cheap architectural gate.

**Branch frozen**: Latent steering is readout steering in disguise. If next frontier exists, requires prior-art audit on "upstream representation → nonlinear decoder → altered reachable output, not reducible to terminal readout steering" BEFORE any code.

---

## 3. VALIDATED FINDINGS (DURABLE, REUSABLE)

### 3.1 Measurements That Survive Scrutiny

| Finding | Mechanism | Evidence | Status |
|---------|-----------|----------|--------|
| **Deep Supervision (D4)** | +0.33pp, improves small-tumor *quality* not detection | E36 size-graded mechanism audit, reproducible, mechanism closed | VALID but sub-1pp |
| **Critical resolution (α_c)** | Per-tumor geometric property predicting difficulty beyond size | Survived 500 permutation trials, subject-clustered bootstrap CI excludes zero, novelty audit clean | VALID measurement, no algorithm yet |
| **Stage-specific rank predictability** | R²=0.518 enc3 vs 0.004 dec1 (E169b/E174, size-controlled) | Durable across checks; identifies real structural asymmetry | VALID, no exploitation found |
| **Bottleneck size-dependency** | Small lesions depend MORE on bottleneck (ρ=−0.454, E48) | Causal ablation, re-verified E85/E86 ρ=−0.3834, same sign/order of magnitude | VALID, no rescue found |
| **t1c modality causality** | t1c owns ET (t1c→0: Dice 0.8433→0.0015, E142) | Causal modality ablation on current checkpoint | VALID, explains E137/E140/E141 nulls |
| **Pooled vs. per-subject Dice gap** | Systematic 1.91pp inflation (E56) | Measurement audit on 14 existing checkpoints, full re-score | VALID methodological correction |

### 3.2 Killed Hypotheses (with proof, not just untested)

| Hypothesis | Proof | Reference |
|-----------|-------|-----------|
| Gradient conflict (margin loss fights seg loss) | Cosine similarity baseline +0.094, mildly cooperative, not conflicting | E14 |
| Boundary-routing (E46's gate localizes boundary) | Clamping at boundary vs. interior no difference (p=0.808) | E47 |
| Coarse-representation rotation is the bottleneck | Decoder-frozen has LOWEST Dice and WEAKEST margin-Dice correlation despite 64x MORE active margin | E18-followup |
| Small-lesion improved detection (DS mechanism) | Detection rate flat 22.0%→20.6% (worse); quality improvement only | E36 |
| Subregion (edema) explains small-tumor failure | Edema-dominance proxy for tiny size; controlled for size, ρ collapses 0.80→0.02 | E28 |
| Grid-entropy captures boundary geometry | Coarsening NEVER increases entropy for small lesions (mathematical necessity) | E37 |
| Resolution alone helps | 96³ worse than 64³; 128³ unintelligible (BatchNorm at batch=1) | E29/E54 |
| Latent-steering reaches unused decoder space | ker(W) is behaviorally dead: 34–55% energy, 5.2e−5 Dice effect | E188-F |

---

## 4. ARCHITECTURE LINEAGE

| Version | Mechanism | Date | Dice |
|---------|-----------|------|------|
| `UNet3D_v2` | Baseline + EGGO-M attempt | Early | 0.9063 (frozen) |
| `UNet3D_v3` | Baseline + D4 deep supervision | E25 | **0.9096** (+0.33pp) |
| `UNet3D_v4` | v3 + D8 bottleneck head | E45 | 0.9110 (single seed, CI crosses D4-only) |
| `UNet3D_v5` | v3 + attention gate enc1 | E46 | 0.9102 (single seed) |
| `UNet3D_v5control` | v5 clean (E131 frozen checkpoint for later audits) | — | — |
| `UNet3D_v6` | v3 + CCABA (causal amplification) | E49 | 0.9094 (3-seed mean, tied D4-only) |
| `UNet3D_v7` | v3 + IECG (counterfactual gating) | E50 | 0.9065 (3-seed mean, worse) |
| `UNet3D_v8` | v6 + v5's attention gate | E51 | 0.9095 (3-seed mean, tied, p=0.919) |

**None reached +1pp bar. D4 (v3) remains the only validated improvement.**

---

## 5. CODE & DATA ARTIFACTS

### Working Code Structure
- `/neuroscan_3d_v*.py`: Architecture files (v1–v9 lineage, git-tracked)
- `/experiments/exp_e12_eggo_m/`: Main experimental hub (e100–e198+ subdirs, most have `run_*.py` + cached JSON results)
- `/experiments/exp_c*/`: Pre-pivot phases (C0, C1 ABO, etc.)
- `/configs/`: Training config templates
- `/Dataset/`: BraTS 2023 GLI (1,251 scans, gitignored)

### Disk-Only Results (Gitignored)
- `E143_recoverability.json`, `E145_recoverability.json`, `E147_repdemand.json`, etc. — JSON result caches per phase (no full re-train needed)
- `E192_Z_cache.json`, `E193_Z_TC_cache.json` — Representation snapshots (cached bottleneck activations for cross-model audits)

### Orphaned Artifacts (Do Not Cite)
- `E151_dec3_sweep.json`, `E152_resolution_recovery.json`, `E153_responsibility.json` — No generating script, no phase doc, definitions unrecoverable

---

## 6. CURRENT STATE (2026-09-20)

### Active Branch: FROZEN
- **E188 closed the latent-steering line** via Gate F (readout decomposition proof)
- **E169–E176 closed representation-space interventions** (all 8 novelty axes occupied, no method found)
- **D4 (+0.33pp) remains the best validated improvement** (below +1pp bar)

### Recommendation (Standing)
- **STOP searching for +1pp mechanism**
- **WRITE UP**: 
  1. Causal-diagnostic methodology (E43→E47→E48 chain) — genuine causal nulls + counterintuitive finding
  2. Multi-seed variance correction (E49→E50→E51) — legitimate methodological contribution
- **IF CONTINUED**: Prior-art audit on "upstream → nonlinear decoder → altered reachable output (not reducible to readout steering)" BEFORE code

### Next Move (If Authorized)
- E189 reachability screen CONSIDERED AND DECLINED (vacuous + would not yield method)
- E190+ cross-disciplinary approaches under consideration (not yet scoped)

---

## 7. CRITICAL METHODOLOGICAL LESSONS

1. **Multi-seed variance is non-negotiable** — E49's 3-seed check retroactively invalidated E45's +0.47pp headline as favorable noise
2. **Pooled vs. per-subject Dice is a systematic 1.91pp trap** — must track both, never mix silently (E56)
3. **Sign-convention bugs can invert entire diagnostic narratives** — but they're catchable via Jacobian-vector product consistency checks (E25 Step 5)
4. **Permutation safeguards catch mathematical artifacts that other checks miss** — applied proactively after E30–E34 failures
5. **Shared-tensor bugs are invisible to standard sanity checks** — must read forward-pass code critically, not assume identity operations actually work (E62→E64)
6. **Freezing ablations can produce confounds** — decoder-frozen had LOWEST Dice+weakest correlation despite HIGHEST activity (E18-followup)
7. **Reused validation set can't be treated as held-out** — acknowledged in E27, all model-selection decisions used 125-val
8. **OOD ablation confounds are real and published** — E58's donor-substitution effect explained by Li & Janson NeurIPS 2024 (external validation)

---

## 8. READING GUIDE TO THIS PROJECT

**For complete understanding in order of discovery**:
1. `PHASE_RESEARCH_ARC_MASTER_REPORT.md` — E1–E43 narrative (290 lines, self-contained)
2. `PHASE_POST_E43_SUMMARY.md` — E44–E53 results + recommendations (135 lines)
3. `docs/RESEARCH_KNOWLEDGE_MAP.md` — Comprehensive E1–E107+ audit table (37k tokens, dense)
4. `PHASE_E188_E15_BRANCH_FROZEN.md` — Latest (E188, Sept 20, closes latent-steering)
5. Memory files (`MEMORY.md` and linked) — Current index + phenomenon-hunt pivot

**For specific mechanisms**:
- E15 (latent steering): `PHASE_E188_E15_BRANCH_FROZEN.md`, `PHASE_E15_EVIDENCE_FREEZE.md`
- E25 (deep supervision): `PHASE_E25_*_MD files (Steps 1–16), PHASE_E36_DEEP_SUPERVISION_AUTOPSY.md`
- E48 (bottleneck size-dependency): `PHASE_E48_NECESSITY_ALLOCATION_MISMATCH_REFRAME.md`, E85/E86 re-verification
- E56 (measurement audit): `PHASE_56_MEASUREMENT_AUDIT.md`
- E188 (Gate F): `PHASE_E188_E15_BRANCH_FROZEN.md`, `PHASE_E188A_LOCAL_RECOVERABILITY_MAP_PREREG.md`, `PHASE_E188F_READOUT_DECOMPOSITION_PREREG.md`

---

**Compiled by reading all major synthesis documents + memory index + latest phase reports (E188 frontmost). Not derived from summaries alone.**
