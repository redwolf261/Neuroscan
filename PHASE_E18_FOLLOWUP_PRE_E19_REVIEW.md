# Pre-E19 Review: Answering the Open Questions on E18

**Status**: Diagnostic review only — no new training, no E19 design decisions made. Directly answers the reviewer's 7 questions plus the requested consolidated table, using the actual code and data (not memory/summary of prior phases). **One prior claim is corrected**: "decoder learning causes the rotation" is walked back to the reviewer's own more careful phrasing, and the central causal question — does reduced rotation restore effective margin optimization — gets a clear **no** from this data, which changes the recommended next step materially.

**Date**: 2026-08-08

---

## 1. Exact E18 implementation, per condition

All from `experiments/exp_e12_eggo_m/e18_ablation_train.py`, verified by direct code inspection just now (not from memory).

| Question | Answer |
|---|---|
| Which layers were frozen, per condition? | `freeze_encoder`: `enc1, enc2, enc3, bottleneck`. `freeze_decoder`: `upconv3, dec3, upconv2, dec2, upconv1, dec1`. `freeze_seg_head`: `seg_head` only. `freeze_bn`: no layers excluded from the optimizer — instead every `nn.BatchNorm3d`'s `momentum` is set to 0 (after a warmup, see below), so running-stat *bookkeeping* stops while the layer itself stays fully in the graph and trainable. |
| `requires_grad=False`? | Yes, explicitly set on every parameter of the frozen submodules (`p.requires_grad_(False)`), **and** those parameters are excluded from the `AdamW` parameter group entirely — not merely zero-gradient no-ops. This means no momentum/variance state was ever allocated for them in the optimizer, which is a stronger and more unambiguous form of "frozen" than `requires_grad=False` alone. |
| Were BatchNorm layers *inside* the frozen components still updating? | **This is the crux of your Question 7 — answered precisely below.** For `freeze_encoder` and `freeze_decoder`: their BatchNorm layers' `weight`/`bias` (affine) parameters are frozen along with everything else in that submodule (they're `nn.Parameter`s owned by those modules, caught by the same freeze loop). Their `running_mean`/`running_var` **do still update** during those runs — freezing only touches the optimizer's parameter list, not `.train()`/`.eval()` mode or `momentum`, so BatchNorm's running-stat *bookkeeping* keeps moving even while the layer's affine transform is frozen, for both `freeze_encoder` and `freeze_decoder`. This is a real, previously-unstated confound (see the "what this means" section below) — the encoder-freeze and decoder-freeze ablations are each simultaneously freezing (a) that region's conv/affine weights AND (b) that region's BN *statistical* adaptation is left running but disconnected from anything that trains it (the affine params it feeds aren't updating), while gradient still flows through it structurally. |
| Was the optimizer rebuilt after freezing? | No — there is only ever one `AdamW(...)` construction, built *after* `apply_ablation()` runs and *from* its returned trainable-parameter list (`e18_ablation_train.py:223,231`). "Rebuilt" doesn't apply; it's built once, correctly excluding frozen params from the start. |
| Same LR/schedule for all other components? | Yes — `CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)` is identical across all 6 configs, same `learning_rate`/`weight_decay` read from the same `configs/brats.yaml`. |
| Same checkpoint initialization? | Yes, verified by construction: `set_seed(seed=0)` is called before `UNet3D_v2(...)` is instantiated, identically in every config's `__init__`. Same seed → same `torch.manual_seed` state → same initial weights for every layer, frozen or not. |
| Same seed/data order? | Same seed, confirmed as above. Data order: `create_brats_loaders(..., num_workers=4)` uses `shuffle=True` with no explicit `generator=` argument, so it falls back to the global `torch` RNG, which is seeded identically pre-construction in every config. **This should produce identical shuffle order across configs by construction, but I have not independently re-verified this holds under `num_workers=4`'s multi-process worker spawning** (PyTorch's per-worker seed derivation from the main process's RNG state is well-defined but I did not add an explicit check, e.g. logging the first-batch subject IDs per config, to confirm empirically) — flagging this as asserted-by-construction, not independently confirmed, rather than overclaiming certainty. |

## 2. Exact definition of `dec1`

Direct inspection of `neuroscan_3d_fixed.py` (frozen v1 baseline) and `neuroscan_3d_v2.py` (the actual model E18 trains):

```
input (B,1,D,H,W)
  ↓
enc1 [Conv3DBlock(1→32) → Conv3DBlock(32→32)]     ← 2 BN layers
  ↓ pool1 (MaxPool3d)
enc2 [Conv3DBlock(32→64) → Conv3DBlock(64→64)]    ← 2 BN layers
  ↓ pool2
enc3 [Conv3DBlock(64→128) → Conv3DBlock(128→128)] ← 2 BN layers
  ↓ pool3
bottleneck [Conv3DBlock(128→256) → Conv3DBlock(256→256)]  ← 2 BN layers
  ↓
upconv3 (ConvTranspose3d, 256→128)                ← 0 BN
  ↓ concat with enc3 (skip connection)
dec3 [Conv3DBlock(256→128) → Conv3DBlock(128→128)] ← 2 BN layers
  ↓
upconv2 (ConvTranspose3d, 128→64)                 ← 0 BN
  ↓ concat with enc2
dec2 [Conv3DBlock(128→64) → Conv3DBlock(64→64)]   ← 2 BN layers
  ↓
upconv1 (ConvTranspose3d, 64→32)                  ← 0 BN
  ↓ concat with enc1
dec1 [Conv3DBlock(64→32) → Conv3DBlock(32→32)]    ← 2 BN layers, THIS is "dec1"
  ↓ (shape: B, 32, D, H, W — full input resolution)
  ├── seg_head: Conv3d(32→1, kernel=1) → Sigmoid   ← reads dec1 DIRECTLY, no BN, no ReLU
  ├── evidential_head: Conv3d(32→2, kernel=1)       ← reads dec1 DIRECTLY
  └── boundary_head: Conv3d(32→1, kernel=1)         ← reads dec1.detach()
```

Direct answers:

- **What layers constitute the decoder?** `upconv3, dec3, upconv2, dec2, upconv1, dec1` — 3 transpose-conv upsampling layers and 3 `Conv3DBlock`-pairs (6 `Conv3DBlock`s total, 12 conv+BN+ReLU units). `dec1` is the **last** of these three `Conv3DBlock`-pairs — it is itself part of the decoder, not a separate stage after it.
- **Where is `dec1` relative to the final decoder block?** `dec1` *is* the final decoder block. There is no decoder computation after it — its raw output (before any head) is the shared trunk representation that E15–E18's whole rotation analysis operates on.
- **Does `seg_head` operate directly on `dec1`?** Yes — confirmed by direct forward-pass trace and shape check: `probs = self.seg_head(dec1)`, `dec1.shape = (B, 32, D, H, W)`, `probs.shape = (B, 1, D, H, W)`. `seg_head` is exactly `Conv3d(32, 1, kernel_size=1) → Sigmoid` — a per-voxel linear projection plus sigmoid, no intervening BN/ReLU/conv of any kind between `dec1` and the segmentation output.
- **Does `dec1` (the block) contain BN/ReLU/Conv?** Yes — 2× `Conv3DBlock`, each `Conv3d → BatchNorm3d → ReLU`. So "dec1" the *tensor* (what E15–E18 measure) is the **output of** 2 BN layers and 2 ReLUs, not a raw conv output.
- **Dimensionality**: `(B, 32, D, H, W)` — 32 channels, full input spatial resolution (confirmed via a live forward pass: `torch.Size([1, 32, 64, 64, 64])` for a 64³ input).

**Direct implication for your framing**: since `dec1` (the tensor EGGO's margin loss operates on) is itself the *output* of `dec1` (the decoder block, containing 2 conv+BN+ReLU units), freezing the decoder freezes the **literal parameters that produce the space the margin is measured in** — this is not one step removed, it's the most direct possible connection. Freezing the encoder, by contrast, only affects `dec1` indirectly, through everything downstream of it still being free to compensate. This asymmetry is worth being explicit about: `freeze_decoder` and `freeze_encoder` are not symmetric interventions with respect to `dec1`'s own geometry, even though they're symmetric in "how many parameters are frozen."

## 3. Full E18 Dice trajectories, all 6 configs, epoch 1–30 (or 1–27 for the stopped freeze_bn run)

Pulled directly from each config's `epoch_metrics.csv` (previously only spot-quoted, not fully tabulated). Val Dice, all 30 epochs:

| Epoch | baseline | freeze_encoder | freeze_decoder | freeze_seg_head | lambda_zero | freeze_bn (attempt 3) |
|---|---|---|---|---|---|---|
| 1 | 0.3255 | 0.2660 | 0.1005 | 0.1805 | 0.4521 | 0.3255 |
| 2 | 0.7474 | 0.6010 | 0.3245 | 0.2079 | 0.6886 | 0.7474 |
| 3 | 0.7587 | 0.7957 | 0.2093 | 0.1659 | 0.7809 | 0.7587 |
| 4 | 0.6802 | 0.8176 | 0.2675 | 0.2238 | 0.7570 | 0.6802 |
| 5 | 0.7809 | 0.7334 | 0.4520 | 0.2562 | 0.7307 | 0.7809 |
| 6 | 0.8622 | 0.8426 | 0.5430 | 0.2726 | 0.8616 | 0.8622 |
| 7 | 0.8619 | 0.8175 | 0.5091 | 0.2946 | 0.8731 | 0.8619 |
| 8 | 0.8745 | 0.8441 | 0.6169 | 0.4121 | 0.8751 | 0.8745 |
| 9 | 0.8614 | 0.8567 | 0.6495 | 0.4569 | 0.8778 | 0.8614 |
| 10 | 0.8886 | 0.8547 | 0.6857 | 0.3573 | 0.8516 | 0.8886 |
| 11 | 0.8611 | 0.8514 | 0.7539 | 0.2498 | 0.8772 | 0.8844 |
| 12 | 0.8854 | 0.8113 | 0.7574 | 0.4167 | 0.8883 | 0.8860 |
| 13 | 0.8948 | 0.8599 | 0.7690 | 0.4358 | 0.8856 | 0.8479 |
| 14 | 0.8878 | 0.8582 | 0.7858 | 0.4906 | 0.8861 | 0.8580 |
| 15 | 0.8869 | 0.8603 | 0.7845 | 0.4172 | 0.8889 | 0.8716 |
| 16 | 0.8969 | 0.8659 | 0.7546 | 0.5394 | 0.8911 | **0.2715** |
| 17 | 0.8993 | 0.8727 | 0.7910 | 0.5852 | 0.8941 | 0.1500 |
| 18 | 0.8965 | 0.8662 | 0.8000 | 0.6058 | 0.8644 | 0.1785 |
| 19 | 0.8812 | 0.8725 | 0.7743 | 0.5153 | 0.8923 | 0.3674 |
| 20 | 0.8904 | 0.8706 | 0.7979 | 0.7054 | 0.8951 | 0.3122 |
| 21 | 0.9022 | 0.8709 | 0.8210 | 0.6978 | 0.8989 | 0.1763 |
| 22 | 0.9050 | 0.8697 | 0.8198 | 0.7403 | 0.8977 | 0.1731 |
| 23 | 0.9006 | 0.8615 | 0.8194 | 0.7698 | 0.8989 | 0.1427 |
| 24 | 0.9063 | 0.8739 | 0.8295 | 0.6989 | 0.8986 | 0.1377 |
| 25 | 0.9046 | 0.8739 | 0.8298 | 0.7899 | 0.8960 | 0.2099 |
| 26 | 0.8907 | 0.8759 | 0.8350 | 0.7838 | 0.8847 | 0.1515 (ep27) |
| 27 | 0.9048 | 0.8729 | 0.8424 | 0.8215 | 0.9002 | — (stopped) |
| 28 | 0.9047 | 0.8627 | 0.8336 | 0.8047 | 0.9038 | — |
| 29 | 0.9020 | 0.8723 | 0.8365 | 0.8233 | 0.8988 | — |
| **30** | **0.9062** | **0.8751** | **0.8432** | **0.8233** | **0.9005** | — |

**This directly answers your worked example.** `freeze_decoder` **does** show `rotation ↓` (established in E18) **and** `Dice ↓` relative to baseline (0.8432 vs 0.9062, a real 0.063 deficit) — exactly the pattern you flagged as the concerning case: *"we successfully stabilized a representation that simply isn't good enough."* This was reported in E18's own numbers but not framed this way — it should have been the headline caveat, not a footnote, and I'm correcting that now.

## 4. Full E18 margin behavior, all conditions

| Config | Dice@30 | Mean margin loss (epochs 11–30, post-instability) | Active hinge %@30 | corr(margin_loss, Dice), epochs 11–30 | p |
|---|---|---|---|---|---|
| baseline | 0.9062 | 0.00155 | 0.12% | −0.683 | 0.0009 |
| freeze_encoder | 0.8751 | 0.00244 | 0.17% | −0.751 | 0.0001 |
| **freeze_decoder** | **0.8432** | **0.09972** (~64× baseline) | **7.54%** (~63× baseline) | **−0.354** | **0.126 (n.s.)** |
| freeze_seg_head | 0.8233 | 0.00076 | 0.10% | +0.202 | 0.394 (n.s.) |
| lambda_zero | 0.9005 | 0.00000 (λ=0) | 0.00% | N/A | N/A |

Three things stand out that were **not** in the E18 report and directly bear on your central question:

1. **`freeze_decoder`'s margin mechanism is far MORE active** than any other config — 64× the mean margin loss, 63× the active hinge percentage of baseline. This makes sense mechanistically: with `dec1` frozen, the hinge condition (pairs closer than `2δ_d`) stays satisfiable for much longer since the representation isn't reshaping to push pairs apart the way it does under normal training.
2. **But this dramatically more active mechanism does NOT correspond to a stronger margin-Dice relationship.** `freeze_decoder`'s correlation (r=−0.354, p=0.126) is actually *weaker and non-significant*, compared to baseline's own real, significant correlation (r=−0.683, p=0.0009) and `freeze_encoder`'s even stronger one (r=−0.751, p=0.0001). (Note: the sign is negative because margin *loss* decreasing correlates with Dice increasing — consistent with margin loss being minimized as training helps overall, not a "more margin loss is bad" causal claim on its own; this is the same directionality baseline already shows.)
3. **`freeze_seg_head` shows a near-zero, non-significant, WRONG-SIGN correlation** (r=+0.202, p=0.394) — its already-low, non-adapting margin activity has essentially no coherent relationship with Dice at all.

**This is the direct answer to your central question.** The causal chain you proposed — `decoder adaptation → rotation → poor margin accumulation → small Dice effect` — predicts that removing decoder adaptation (reducing rotation) should let margin accumulate more effectively and that this should translate into a *stronger*, not weaker, margin-Dice relationship, and ideally recovered (not reduced) Dice. **None of these three predictions hold**: margin accumulates far more (true) but the margin-Dice relationship gets *weaker*, and Dice goes *down*, not up. The second arrow in your causal chain (`rotation → poor margin accumulation`) is not supported by this data in the direction needed — freezing the decoder makes the margin mechanism *more* active, not more effectively coupled to the outcome it's supposed to drive.

## 5. E17's exact rotation definitions — moving vs. fixed reference

Verified directly against `e17_target_stability_analysis.py`'s source, not from memory:

| Measure | Reference frame | Code location |
|---|---|---|
| **Part 1, consecutive** (`cos(dir_t, dir_{t+1})`) | **Moving** — each pair compares its own two live centroid directions, freshly computed at both `t` and `t+1` | `centroid_dirs[e_t]["unit"]` vs `centroid_dirs[e_t1]["unit"]`, both computed per-epoch from that epoch's own `z` |
| **Part 1, long-range** (`cos(dir_1, dir_t)`) | **Fixed** — always compared against epoch 1's centroid direction | `centroid_dirs[1]["unit"]` fixed, `centroid_dirs[epoch]["unit"]` varies |
| **Part 4, Procrustes** | **Moving only** — no fixed-reference Procrustes variant exists in the code; every reported number is a consecutive-pair `(t, t+1)` alignment | `Z_t`/`Z_t1` are always adjacent checkpoints, `orthogonal_procrustes(Z_t, Z_t1)` |
| **Part 2, gradient stability** | **Moving** — `cos(g_margin(t), g_margin(t+1))`, consecutive | `grad_by_epoch[e_t]` vs `grad_by_epoch[e_t1]` |
| **Part 3, gradient transport** | **Fixed** — always vs. epoch 1 (plus one explicit epoch5-vs-epoch30 fixed pair) | `g1 = grad_by_epoch[1]`, compared against every later epoch |
| **E16's displacement decomposition** (a related, earlier metric, not part of E17/E18's own suite) | **Fixed** (primary) — margin direction computed once at `z0` (epoch 1) and held constant; a **live/secondary** cross-check (recomputed at each epoch's own centroid) was added later in E16 specifically because the fixed-reference result looked surprising | `tumor_centroid_0`/`bg_centroid_0` fixed from `z0_by_voxel`, vs. a separate live-recomputed version added as a cross-check |

**This is exactly the distinction you're pointing at, and it was present in the code all along but not surfaced clearly enough in the E17/E18 writeups.** E17's headline finding — "severe rotation early, stabilizes, but never returns to its original orientation" — synthesizes both reference frames correctly (moving-reference consecutive pairs show stabilization; fixed-reference long-range shows the permanent offset), but E18's rotation-magnitude summary table (the one requested in the E18 spec, `1→5/5→10/10→30`) uses **only the moving-reference Part 1 metric** (`rotation_magnitude()` in `e18_measure_rotation.py` computes `1 − cos(dir_a, dir_b)` where both `a` and `b` are live centroid directions at their own epochs) — it does not report a fixed-reference version of the same summary. This means E18's rotation comparison across ablations is entirely a "does the *rate* of change slow down" comparison, not a "does the representation end up closer to where it started" comparison. Both are legitimate questions, but only one was asked in E18's own summary table, and this should be flagged as a real gap, not silently assumed to be covered.

## 6. Decoder gradients — layer-wise, parameter-space

**This was not measured in E17 or E18, and I will not estimate it.** Everything computed so far (`torch.autograd.grad(loss, dec1, ...)`) is a gradient **with respect to the activation tensor `dec1`**, not with respect to any layer's **parameters** (`θ_D`). `‖∇_θ L_seg‖` and `‖∇_θ L_margin‖`, whether pooled over the whole decoder or broken out per block (`dec3`, `dec2`, `dec1`), do not exist anywhere in the current codebase. This is a genuinely new measurement, not a re-derivation of existing data — building it would require a dedicated script computing `torch.autograd.grad(loss, decoder_block.parameters(), ...)` per block, on live (not frozen) checkpoints, most naturally from the **baseline** run's checkpoints (since that's the config where the decoder was actually training and producing the rotation in question). Flagging as the most concrete, well-scoped follow-up measurement from this whole list — narrower in scope than a new E19 training phase, purely diagnostic on existing baseline checkpoints.

## 7. BatchNorm placement — corrected count

Verified by direct module inspection (`isinstance(m, nn.BatchNorm3d)` over each submodule), not estimated:

```
Encoder (enc1, enc2, enc3, bottleneck):  8 BatchNorm3d layers total (2 per stage x 4 stages)
Decoder (upconv3, dec3, upconv2, dec2, upconv1, dec1):  6 BatchNorm3d layers total
  - upconv3/upconv2/upconv1: 0 BN each (bare ConvTranspose3d, no BN/ReLU wrapper)
  - dec3/dec2/dec1: 2 BN each (each is 2x Conv3DBlock, each Conv3DBlock has its own BN)
seg_head: 0 BN (single Conv3d + Sigmoid)
evidential_head: 0 BN (single Conv3d)
boundary_head: 0 BN (single Conv3d)
```

**Correction to my earlier framing**: I need to walk back the implicit suggestion that the decoder is somehow BN-heavier than the encoder — **it is not; the encoder actually has more BN layers (8 vs 6)**. What's true, and what actually matters for your concern: **both** `freeze_encoder` and `freeze_decoder` simultaneously freeze that region's conv weights *and* leave that region's BN running-stats still updating (unfrozen, since `freeze_bn` is a separate, orthogonal ablation — `freeze_encoder`/`freeze_decoder` never touch `momentum` or `.eval()`/`.train()` mode). This is a **symmetric confound** across both ablations, not one that specifically undermines the decoder-freeze result relative to the encoder-freeze result — both comparisons are equally confounded by "region's BN keeps statistically adapting even though its affine params are frozen." It does mean neither `freeze_encoder` nor `freeze_decoder` is a pure "conv-weight-only" freeze, and the true decomposition (conv weights vs. BN specifically, within a single region) was never isolated by E18's design — that would require a `freeze_decoder_conv_only` (leave decoder BN's affine+running-stats free, freeze only conv/transpose-conv weights) variant, not yet built.

Given this, your proposed downgrade of the claim is the right one, and I'll use it going forward:

> **"The decoder block is the strongest experimentally identified source of the early representation rotation"** — not "decoder learning causes the rotation." The former is what E18's data supports; the latter overstates both the causal claim (correlational ablation evidence, not a mechanistic proof) and implicitly attributes the effect to a cleanly-isolated "decoder learning" variable that E18's design does not actually isolate from decoder-region BN dynamics.

---

## The consolidated table, as requested

| Condition | Dice@30 | Rotation 1→5 (moving-ref, centroid) | Procrustes (mean, all pairs) | Mean margin loss (ep 11–30) | Active hinge %@30 | corr(margin, Dice) [p] |
|---|---|---|---|---|---|---|
| Baseline | 0.9062 | 0.1585 | 0.8751 | 0.00155 | 0.12% | −0.683 [0.0009] |
| Freeze encoder | 0.8751 | 0.2026 | 0.8899 | 0.00244 | 0.17% | −0.751 [0.0001] |
| Freeze decoder | **0.8432** | **0.0608** | **0.9468** | **0.09972** | **7.54%** | **−0.354 [0.126, n.s.]** |
| Freeze seg-head | 0.8233 | 0.2833 | 0.7555 | 0.00076 | 0.10% | +0.202 [0.394, n.s.] |
| Lambda=0 | 0.9005 | 0.1329 | 0.8264 | 0.00000 | 0.00% | N/A |
| BN (attempt 3, epochs 1–15 only, collapses after) | 0.87 (ep15, last healthy) | 0.1585 (identical to baseline pre-freeze) | 0.8278 (partial, confounded by collapse) | ~0.0024 (ep 11–15 only) | ~0.28% (ep15) | not computed (too few healthy post-freeze points) |

## What this means for E19

Directly addressing your closing question — **does reducing representation rotation actually restore the geometric optimization EGGO was designed to perform?**

**No, not by this evidence.** `freeze_decoder` achieves the strongest rotation reduction of any tested intervention (lowest 1→5 centroid rotation, highest Procrustes stability, by a wide margin over every other config) — but Dice is *lower* than baseline (0.8432 vs 0.9062), and critically, the margin-Dice relationship is *weaker and non-significant* under this intervention (r=−0.354, p=0.126) compared to baseline's own real correlation (r=−0.683, p=0.0009). The margin mechanism becomes dramatically *more active* (64× the margin loss, 63× the active hinge rate) without becoming more *usefully coupled* to segmentation quality. This is the "no" branch of your own decision framework: **decoder rotation looks like an associated phenomenon, not the bottleneck** — at least not in the simple, single-lever sense the causal chain proposed. Freezing the identified rotation source did not unlock better margin-driven optimization; it just produced a differently-behaved (more margin-active, lower-Dice) model.

This does not mean E18's finding is wrong or worthless — decoder-region freezing genuinely, robustly reduces rotation by three independent geometric measures, and that remains a real, correctly-triangulated result. It means the *next* inferential step — "so stop the decoder from rotating and EGGO's optimization will work better" — is not supported, and building an E19 intervention on that assumption would very likely fail for a reason already visible in this data, not a new one. Per your own framing: this is exactly why E19 should not be designed from E18 alone, and the honest options now are (a) the layer-wise parameter-gradient measurement (Question 6) to understand *why* the more-active margin mechanism under `freeze_decoder` fails to help, (b) accepting decoder rotation as a correlate rather than the causal bottleneck and looking elsewhere, or (c) some other diagnostic this data doesn't yet suggest. Not decided here.
