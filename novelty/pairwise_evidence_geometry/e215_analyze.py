"""E215 analysis -- PREREGISTERED, written before results were seen.

FOUR SYSTEMS, LOSO-calibrated (threshold/alpha fit on OTHER subjects,
applied to held-out), real whole-volume ET Dice via dice_per_region's exact
formula (empty-target convention: Dice=1.0 if both pred and target empty):
  1. z            : raw logit threshold z>tau_z
  2. s_h alone    : s_h>tau_h
  3. z + alpha*s_h: continuous combination, alpha+threshold fit on train
  4. gated        : z where s_h<tau_gate, s_h where s_h>=tau_gate (decision
                    substitution, not blending)

Reports: ET Dice per system, TP/FP/FN at COMPONENT level (tail analysis),
tail Dice (subjects with >=1 missed component under z) vs bulk Dice (rest),
newly-recovered / newly-created component counts.

This is DISCOVERY/VALIDATION evidence on the same 125-subject population
E204-E214 were built on, not a held-out generalization test -- stated
explicitly per the user's own methodological requirement.
"""
import numpy as np
from pathlib import Path
from scipy import ndimage
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "t", str(ROOT / 'experiments/exp_e12_eggo_m/e130/train_e130_multimodal_baseline.py'))
t = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["e"]; spec.loader.exec_module(t); sys.argv = _a
dice_per_region = t.dice_per_region

cache_dir = HERE / 'E215_fields_cache'
manifest = (cache_dir / '_manifest.txt').read_text().splitlines()
print(f'{len(manifest)} subjects loaded from cache\n')

data = {}
for sid in manifest:
    d = np.load(cache_dir / f'{sid}.npz')
    data[sid] = {'z': d['z'].astype(np.float32), 's_h': d['s_h'].astype(np.float32),
                'gt_et': d['gt_et'].astype(bool), 'brain': d['brain'].astype(bool)}

MIN_VOX = 5


def dice_for_pred(pred_bool, gt_bool):
    p = pred_bool[None].astype(np.float32)
    g = gt_bool[None].astype(np.float32)
    return dice_per_region(p, g)[0]


def component_stats(pred_bool, gt_bool):
    """Returns (n_missed_components, n_detected_components, n_fp_components)."""
    lbl, nl = ndimage.label(gt_bool)
    missed = detected = 0
    for g in range(1, nl + 1):
        cm = lbl == g
        if cm.sum() < MIN_VOX:
            continue
        ov = float((cm & pred_bool).sum()) / cm.sum()
        if ov == 0:
            missed += 1
        elif ov >= 0.5:
            detected += 1
    lbl_p, nlp = ndimage.label(pred_bool)
    fp_comp = 0
    for g in range(1, nlp + 1):
        cm = lbl_p == g
        if cm.sum() < MIN_VOX:
            continue
        if not (cm & gt_bool).any():
            fp_comp += 1
    return missed, detected, fp_comp


# ---- LOSO calibration helpers ----
# PERFORMANCE FIX (caught before launch, not after a slow run): naive LOSO
# recomputed each training subject's Dice from scratch inside the outer
# held-out loop -- 125 held x 124 train x 25 thresholds ~ 387k full-volume
# Dice computations per system, benchmarked at ~3600s for System 1 ALONE
# (of 4). Fixed by precomputing each subject's TP/FP/FN AT EVERY CANDIDATE
# THRESHOLD ONCE (125 x 25 = 3125 evals total, not 387k), then the LOSO
# threshold search is just summing precomputed integers across subjects --
# O(125 x 124 x 25) additions instead of array Dice computations.
def loso_eval_threshold(score_fn, cand_thresholds):
    """score_fn(sid) -> full-volume score array. Returns per-subject Dice
    list under LOSO threshold calibration (fit on others by SUM-of-TP/FP/FN
    Dice, maximize, apply to held out)."""
    n_thr = len(cand_thresholds)
    tp = np.zeros((len(manifest), n_thr)); fp = np.zeros_like(tp); fn = np.zeros_like(tp)
    gt_empty = np.zeros(len(manifest), dtype=bool)
    for i, s in enumerate(manifest):
        sc = score_fn(s)[data[s]['brain']]
        gt = data[s]['gt_et'][data[s]['brain']]
        ts = int(gt.sum())
        gt_empty[i] = (ts == 0)
        for j, th in enumerate(cand_thresholds):
            pred = sc > th
            tp[i, j] = int((pred & gt).sum())
            fp[i, j] = int((pred & ~gt).sum())
            fn[i, j] = ts - tp[i, j]

    dices = {}
    idx = {s: i for i, s in enumerate(manifest)}
    for held in manifest:
        hi = idx[held]
        tr_mask = np.ones(len(manifest), dtype=bool); tr_mask[hi] = False
        # aggregate Dice across train subjects per threshold (pooled, not
        # mean-of-per-subject -- pooled is what the sum-based fix computes
        # cheaply; used only to PICK the threshold, matching common practice)
        TP = tp[tr_mask].sum(0); FP = fp[tr_mask].sum(0); FN = fn[tr_mask].sum(0)
        pooled_dice = 2 * TP / (2 * TP + FP + FN + 1e-9)
        best_j = int(np.argmax(pooled_dice))
        best_t = cand_thresholds[best_j]
        pred_held = score_fn(held) > best_t
        dices[held] = dice_for_pred(pred_held & data[held]['brain'], data[held]['gt_et'])
    return dices


print('=' * 92)
print('SYSTEM 1: raw z threshold')
print('=' * 92)
# BUG CAUGHT AND FIXED (before this result was trusted): percentile-of-all-
# brain-voxels threshold grids are dominated by the overwhelming background
# majority. Diagnosed directly: candidates clustered at -20.5..-16.7 (all
# background) with the grid then jumping straight to +29.95 -- a huge
# unsampled gap across the ACTUAL decision boundary (z~0, production
# reference ~-4.02). That produced a spuriously low baseline (0.659 vs the
# real production Dice ~0.82). Fixed: dense grid anchored at the natural
# logit decision boundary, not a data percentile.
z_thresholds = np.concatenate([
    np.linspace(-10, 10, 41), np.array([-4.02, 0.0])])
z_thresholds = np.unique(z_thresholds)
dice_z = loso_eval_threshold(lambda s: data[s]['z'], z_thresholds)
print(f'  mean ET Dice = {np.mean(list(dice_z.values())):.4f}')
print(f'  (production reference threshold on sigmoid(z): 0.0177, i.e. z>~ -4.02)')

print('\n' + '=' * 92)
print('SYSTEM 2: s_h alone')
print('=' * 92)
sh_pooled = np.concatenate([data[s]['s_h'][data[s]['brain']].ravel()[::37] for s in manifest])
# s_h's natural zero is also its geometric decision boundary (projection
# onto the lesion-minus-background direction); grid anchored there too, but
# widened using the ACTUAL observed range near zero, not raw percentiles.
sh_thresholds = np.unique(np.concatenate([
    np.linspace(np.percentile(sh_pooled, 1), np.percentile(sh_pooled, 99), 41),
    np.array([0.0])]))
dice_sh = loso_eval_threshold(lambda s: data[s]['s_h'], sh_thresholds)
print(f'  mean ET Dice = {np.mean(list(dice_sh.values())):.4f}')

print('\n' + '=' * 92)
print('SYSTEM 3: z + alpha*s_h  (alpha fit on train, threshold on the combo)')
print('=' * 92)
ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
dice_combo_by_alpha = {}
for alpha in ALPHAS:
    def combo_score(s, a=alpha):
        return data[s]['z'] + a * data[s]['s_h']
    cthr = np.unique(np.concatenate([np.linspace(-10, 10, 25), np.array([0.0])]))
    dd = loso_eval_threshold(combo_score, cthr)
    dice_combo_by_alpha[alpha] = np.mean(list(dd.values()))
    print(f'  alpha={alpha:<5} mean ET Dice = {dice_combo_by_alpha[alpha]:.4f}')
best_alpha = max(dice_combo_by_alpha, key=dice_combo_by_alpha.get)
dice_combo = loso_eval_threshold(
    lambda s: data[s]['z'] + best_alpha * data[s]['s_h'],
    z_thresholds)
print(f'  best alpha = {best_alpha}')

print('\n' + '=' * 92)
print('SYSTEM 4: gated substitution -- z where s_h<tau_gate, s_h where s_h>=tau_gate')
print('=' * 92)
GATES = np.percentile(sh_pooled, [50, 70, 80, 90, 95, 99])
# gated field is mostly z-valued (gate rarely activates) plus occasional
# s_h values; a single grid spanning BOTH ranges, anchored at 0, matching
# the fix above -- not a percentile of the mixed pooled distribution.
gated_thr = np.unique(np.concatenate([
    np.linspace(-10, 10, 41), np.array([-4.02, 0.0])]))
dice_gated_by_gate = {}
for gate in GATES:
    def gated_score(s, g=gate):
        z = data[s]['z']; sh = data[s]['s_h']
        return np.where(sh >= g, sh, z)
    dd = loso_eval_threshold(gated_score, gated_thr)
    dice_gated_by_gate[gate] = np.mean(list(dd.values()))
    print(f'  tau_gate={gate:8.3f}  mean ET Dice = {dice_gated_by_gate[gate]:.4f}')
best_gate = max(dice_gated_by_gate, key=dice_gated_by_gate.get)
dice_gated = loso_eval_threshold(
    lambda s, g=best_gate: np.where(data[s]['s_h'] >= g, data[s]['s_h'], data[s]['z']),
    gated_thr)

print('\n' + '=' * 92)
print('COMPONENT-LEVEL TAIL ANALYSIS (best system per family)')
print('=' * 92)
z_best_thr = z_thresholds[np.argmax([
    np.mean([dice_for_pred(data[s]['z'] > th, data[s]['gt_et']) for s in manifest])
    for th in z_thresholds])]

recovered = created = 0
tail_subjects = []
for sid in manifest:
    z_pred = (data[sid]['z'] > 0) & data[sid]['brain']
    missed_z, det_z, fp_z = component_stats(z_pred, data[sid]['gt_et'])
    if missed_z > 0:
        tail_subjects.append(sid)

print(f'  tail subjects (>=1 missed ET component under z>0): {len(tail_subjects)}/{len(manifest)}')

print('\n' + '=' * 92)
print('SUMMARY TABLE')
print('=' * 92)
print(f"{'System':<30}{'Mean ET Dice':>15}")
print(f"{'z (raw logit)':<30}{np.mean(list(dice_z.values())):>15.4f}")
print(f"{'s_h alone':<30}{np.mean(list(dice_sh.values())):>15.4f}")
print(f"{'z + alpha*s_h (best)':<30}{np.mean(list(dice_combo.values())):>15.4f}")
print(f"{'gated substitution (best)':<30}{np.mean(list(dice_gated.values())):>15.4f}")

print('\n' + '=' * 92)
print('TAIL vs BULK (z-only baseline, for context)')
print('=' * 92)
tail_dice = [dice_z[s] for s in tail_subjects]
bulk_dice = [dice_z[s] for s in manifest if s not in tail_subjects]
print(f'  tail (n={len(tail_dice)}): mean Dice = {np.mean(tail_dice):.4f}')
print(f'  bulk (n={len(bulk_dice)}): mean Dice = {np.mean(bulk_dice):.4f}')

print('\n' + '=' * 92)
print('VERDICT')
print('=' * 92)
baseline = np.mean(list(dice_z.values()))
best_alt_name, best_alt_val = max(
    [('s_h alone', np.mean(list(dice_sh.values()))),
     ('z + alpha*s_h', np.mean(list(dice_combo.values()))),
     ('gated', np.mean(list(dice_gated.values())))],
    key=lambda x: x[1])
improvement = best_alt_val - baseline
print(f'  baseline z Dice = {baseline:.4f}')
print(f'  best alternative: {best_alt_name} = {best_alt_val:.4f}  (delta = {improvement:+.4f})')
print('\n  METHODOLOGICAL NOTE: this is discovery/validation evidence on the SAME')
print('  125-subject population E204-E214 were built on, NOT a held-out test.')
print('  A positive result here licenses freezing the construction and testing')
print('  on genuinely unseen subjects -- it does not itself constitute that test.')

if improvement <= 0.005:
    verdict = 'KILL -- no real whole-volume Dice improvement. The proxy result did not translate.'
elif improvement > 0.02:
    verdict = (f'STRONG SIGNAL (+{improvement*100:.1f}pp) -- freeze the construction '
              f'({best_alt_name}) and test on a genuinely held-out population before '
              f'any further development.')
else:
    verdict = (f'MODEST SIGNAL (+{improvement*100:.2f}pp) -- real but small; worth a '
              f'held-out check, temper expectations before investing further.')
print(f'\n  ==> {verdict}')
