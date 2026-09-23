# Recoverability frontier — reproducible pipeline

Promoted from scratchpad 2026-09-20. This directory holds every script behind the
E143 → E199 result chain. Run in numerical order from the **repo root**.

## Provenance warning (why this directory exists)

Scripts 01 and 02 originally lived only in a session scratchpad and **were deleted from the
temp directory** before promotion. They were reconstructed from a full read made earlier the
same day, then **verified by re-execution**: script 02 reproduces the original run log
bit-identically —

```
Z alone 0.1243 | Z+O_i 0.6874 | dR2 +0.5631 | perm p 0.0000 | Regime III n=0/90
```

This is exactly the failure mode E147 suffered permanently (its `Rstar` generator was never
committed and cannot be recovered — see `PHASE_E147_REPRESENTATIONAL_DEMAND_MINING.md`).
**Nothing scientific should live only in a scratchpad.**

## Scripts

| # | script | produces | ~cost |
|---|---|---|---|
| 01 | `01_e143_recoverability_ET.py` | `E143_recoverability.json` — cross-fitted $O_i$, 4 observers, 5-fold, n=90 | ~20 min GPU |
| 02 | `02_e144_delta_r2_ET.py` | ΔR² vs 15-var difficulty model + permutation null + Regime III check | ~5 min |
| 03 | `03_e192_crossmodel_ET.py` | `E192_crossmodel.json` — the same test across **6 trained models** | ~10 min |
| 04 | `04_e193_recoverability_TC.py` | `E193_recoverability_TC.json` — frozen methodology, TC instead of ET | ~20 min GPU |
| 05 | `05_e193b_delta_r2_TC.py` | `E193_crossmodel_TC.json` — TC ΔR² across 6 models (**the negative**) | ~5 min |
| 06 | `06_e197_tail_inferability.py` | `E197_inferability.json` — in-subject oracle on the 15-subject tail | ~15 min GPU |
| 07 | `07_e198_tail_transfer.py` | `E198_transfer.json` — leave-one-out transfer within the tail (**the closure**) | ~30 min GPU |

Outputs land in `experiments/exp_e12_eggo_m/`. Scripts 02/03 require 01; 05 requires 04;
07 is independent of 06 but is only interpretable beside it.

## Headline results

| result | value |
|---|---|
| observer independence (ET) | pairwise Spearman **0.786 – 0.974** (all 6 pairs) |
| ΔR², ET, 6 models | **+0.451 … +0.563**, permutation p = 0.0000 for all |
| ΔR², TC, 6 models | **−0.046 … +0.099**, 3 of 6 negative → **phenomenon is ET-specific** |
| Regime III (high $O_i$, poor Dice) | **n = 0 of 90** |
| tail: in-subject oracle vs honest transfer | **0.741 → 0.193** (leakage), `global_good` 0.352 |

## Reading discipline

- **Script 06 is deliberately leaky** (trains and tests on the same subject). Its output is an
  inferability upper bound and **must never be quoted as a performance figure**. Its only
  legitimate use is the contrast against script 07.
- $O_i$ is **model-independent** but **not label-free**: ground truth trains the observers and
  restricts voxels to `seg>0`. It is blind to the *target subject's* labels via cross-fitting.
  Write "cross-fitted", never "label-free".
- Per E194, no number enters the paper unless it has been recomputed from a stored artifact.

## Environment

Python 3.11 (`venv_gpu`), PyTorch + CUDA, `nibabel`, `scipy`, `scikit-learn`.
Data: `Dataset/Training/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData` (1,251 subjects).
Model evals: `experiments/exp_e12_eggo_m/e130/E130_full_eval_*_per_subject.json` (6 models).
Seeds fixed throughout (`np.random.default_rng(0)`, `torch.manual_seed(0)`).

## Related documents

`PHASE_E191` (decision) · `PHASE_E192` (cross-model) · `PHASE_E193` (TC negative) ·
`PHASE_E194` (kill-list audit) · `PHASE_E195` (oracle arithmetic) · `PHASE_E196` (headroom gate) ·
`PHASE_E197` (tail inferability) · `PHASE_E198` (transfer closure) · `PHASE_E199` (prior art)
