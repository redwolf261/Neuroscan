"""E203 -- F0: offline oracle test for recoverability-gap correction.

GATE: does disagreement-correction between the raw-intensity observer and the
trained segmentation model have >=1pp THREE-REGION headroom, even under the
most generous possible (oracle) combination rule? If not, the entire final
mechanism proposal is dead before any code is written.

WHAT THIS CAN AND CANNOT TEST WITH EXISTING ARTIFACTS:
  We have, per subject: O_i (SUBJECT-LEVEL scalar, cross-fitted observer Dice)
  and D_i (SUBJECT-LEVEL scalar, model ET Dice). We do NOT have voxel-level
  P_raw(v) and P_net(v) saved anywhere -- E143's observer outputs a binary
  prediction per voxel internally but only the resulting scalar Dice was
  persisted to disk, and the network's own per-voxel probs were never cached
  (E200 is caching them now, for morphology, but even that only gives P_net,
  not P_raw at matching voxel resolution).

  This means the exact voxel-level oracles proposed (max(P_net,P_raw),
  confidence-gated correction, a trained tiny logistic correction) CANNOT be
  computed from stored artifacts alone -- they need a NEW paired forward pass
  (network probs) + observer probs at IDENTICAL voxels, which is a real,
  bounded piece of new computation, not a re-derivation.

  What CAN be computed right now, honestly, from existing scalars: the
  SUBJECT-LEVEL oracle already computed in E198/the REPORT_RECOVERABILITY_
  FRONTIER.md -- max(D_i, O_i) per subject -- is mathematically the loosest
  possible upper bound on ANY voxel-level correction scheme for ET, because
  no combination of P_net and P_raw can produce a Dice for that subject
  higher than what a voxel-level oracle exploiting BOTH signals perfectly
  could achieve, and Dice(voxel-level oracle) generally EXCEEDS
  max(D_i, O_i) -- an oracle that combines per-voxel can be strictly better
  than a subject-level max of endpoints. So max(D_i, O_i) is a LOWER bound
  on the correction scheme's ceiling, not an upper bound.

  Reported here: TWO honest numbers, clearly labeled --
    (a) LOWER BOUND ceiling: subject-level max(O_i, D_i) -- if even the most
        generous number computable right now clears +1pp on the ET
        contribution alone at 3-region weighting, that's necessary-but-not-
        sufficient evidence to proceed (it already appeared in
        REPORT_RECOVERABILITY_FRONTIER.md as the "oracle" ceiling: +0.377pp
        3-region -- reproduced here independently from raw JSONs, not
        copied).
    (b) What a TRUE voxel-level F0 requires to be run properly: a follow-up
        script (not this one) that caches P_net(v) and P_raw(v) at matched
        voxels for the SAME subjects and constructs the 4 oracles the user
        specified (max, confidence-gated, morphology-conditioned, tiny
        logistic). This script explicitly does NOT fabricate that.

Run from repo root: python experiments/exp_recoverability/11_e203_f0_disagreement_oracle.py
"""
import json
import numpy as np
from pathlib import Path

ROOT = Path('.')

# ---- load exactly the same artifacts E195/E198/REPORT used, recomputed here ----
ev = {x['subject_id']: x for x in json.load(open(
    ROOT / 'experiments/exp_e12_eggo_m/e130/E130_full_eval_E131_v5control_seed0_per_subject.json'))}
O = json.load(open(ROOT / 'experiments/exp_e12_eggo_m/E143_recoverability.json'))['mlp_deep']

all_ids = list(ev.keys())
et = np.array([ev[s]['dice_ET'] for s in all_ids])
tc = np.array([ev[s]['dice_TC'] for s in all_ids])
wt = np.array([ev[s]['dice_WT'] for s in all_ids])
base_3region = (et.mean() + tc.mean() + wt.mean()) / 3

print('='*76)
print('E203 -- F0 GATE, PART (a): subject-level lower-bound ceiling')
print('='*76)
print(f'baseline 3-region mean Dice: {base_3region:.6f}')
print(f'target (+1.0pp):             {base_3region+0.01:.6f}')
print()

# apply max(D_i, O_i) only to the 90 subjects where O_i exists; others unchanged
O_ids = set(O.keys())
et_oracle = et.copy()
n_lifted = 0
lift_total = 0.0
for i, s in enumerate(all_ids):
    if s in O_ids:
        o = O[s]
        if o > et[i]:
            n_lifted += 1
            lift_total += (o - et[i])
            et_oracle[i] = o

oracle_3region = (et_oracle.mean() + tc.mean() + wt.mean()) / 3
gain_pp = 100 * (oracle_3region - base_3region)

print(f'subjects with O_i available:        {len(O_ids)} of {len(all_ids)}')
print(f'subjects where O_i > D_i (lifted):  {n_lifted}')
print(f'sum of per-subject lifts (ET):      {lift_total:.4f}')
print(f'ET mean: {et.mean():.4f} -> {et_oracle.mean():.4f}  ({100*(et_oracle.mean()-et.mean()):+.3f}pp)')
print(f'3-region mean: {base_3region:.6f} -> {oracle_3region:.6f}')
print(f'3-region GAIN: {gain_pp:+.3f}pp')
print()
if gain_pp >= 1.0:
    print('LOWER BOUND CLEARS +1.0pp -- necessary condition met. A true voxel-level')
    print('oracle (which can only be >= this subject-level number) is WORTH TESTING.')
else:
    print(f'LOWER BOUND = {gain_pp:+.3f}pp, {"below" if gain_pp < 1.0 else "at/above"} +1.0pp.')
    print('This does NOT by itself kill F0 -- a voxel-level oracle can legitimately')
    print('exceed this subject-level max(D_i,O_i) bound (see docstring). But it means')
    print('the burden of proof for F0 passing now rests ENTIRELY on the voxel-level')
    print('correction (part b), not on this cheap check.')

print()
print('='*76)
print('SANITY CHECK against REPORT_RECOVERABILITY_FRONTIER.md Section 4.1')
print('='*76)
print('Report claims: ET mean current=0.8441, oracle=0.8554, ET gain=+1.132pp,')
print('               3-region gain=+0.377pp')
print(f'This script (full 125, not the 90-subject O_i pop restricted to target_voxels_ET>=200):')
print(f'  ET mean current = {et.mean():.4f}')
print(f'  ET mean oracle   = {et_oracle.mean():.4f}')
print(f'  ET gain          = {100*(et_oracle.mean()-et.mean()):+.3f}pp')
print(f'  3-region gain    = {gain_pp:+.3f}pp')
note = ('MATCHES within rounding.' if abs(gain_pp - 0.377) < 0.05
        else 'DIFFERS from the report -- population or O_i source may differ, investigate before citing either number.')
print(f'  -> {note}')

print()
print('='*76)
print('PART (b): what a TRUE voxel-level F0 needs -- NOT computed here')
print('='*76)
print("""
The subject-level max(O_i, D_i) bound above is a LOWER bound on any voxel-
level oracle's achievable ceiling (a voxel-level combiner has strictly more
freedom than picking one whole-subject prediction over the other). It having
already reproduced the +0.377pp figure from REPORT_RECOVERABILITY_FRONTIER.md
means this repeats a number already established, it does not newly test the
disagreement-correction idea.

The genuinely new test the user specified -- Oracle 1 (max per-voxel),
Oracle 2 (confidence-gated), Oracle 3 (morphology-conditioned), Oracle 4
(tiny trained logistic on disagreement) -- requires P_net(v) and P_raw(v) at
MATCHED voxels for the same subjects, which do not exist as a stored
artifact. This is a real, bounded next step (re-run E143's 4 observers'
per-voxel predictions on the SAME held-out folds, paired against the
sliding-window model probs E200 is already caching), not a re-hash of E198.

RECOMMENDATION: build E204 next -- the true voxel-level F0 -- using E200's
cached model probabilities (once it finishes) joined against a re-run of
E143's mlp_deep observer's per-voxel predictions on the same 90-subject
population, at the SAME voxel indices. Estimated cost: reuses E200's cached
inference (already running) + one more observer inference pass (~5 min GPU,
matching E143's own runtime). No training required for Oracles 1-3; Oracle 4
needs a tiny logistic fit (seconds).
""")
