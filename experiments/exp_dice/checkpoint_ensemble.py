"""Checkpoint ensembling on top of the fixed TTA pipeline.

Reports, per the stopping rule:
  1. each checkpoint individually (LOSO tau)
  2. ensemble at default tau=0.5
  3. ensemble at the CURRENT TTA-optimised thresholds (no re-fit)
  4. ensemble with thresholds RE-OPTIMISED by LOSO   <-- the decisive number

Logits are averaged (not masks), consistent with the campaign.

NOTE: epochs 40/45/50 of this run are POST-COLLAPSE (val Dice 0.52-0.57 vs
0.88-0.89) -- the documented late-training BatchNorm collapse. They are
excluded; including them would poison the ensemble. Usable healthy
checkpoints: best(ep34), epoch_030, epoch_035.
"""
import sys, json, itertools
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / 'cache'
REGIONS = ('ET', 'TC', 'WT')
# EXTENDED AGAIN: TC pinned at the 1e-4 floor, so the search was truncated,
# not converged. Extend to 1e-8. (Cheap: cached logits, no re-inference.)
GRID = np.unique(np.round(np.concatenate([
    np.logspace(-8, -1.05, 40),
    np.arange(0.10, 0.96, 0.05)]), 10))
REF = 0.8595193
TTA_TAU = (0.0177, 0.0004, 0.90)      # current operating point from the TTA run


def load(tag):
    subs, L, G = [], [], []
    for f in sorted(CACHE.glob(f'*__{tag}.npz')):
        d = np.load(f)
        shp = tuple(d['gt_shape'])
        gt = np.unpackbits(d['gt'])[:int(np.prod(shp))].reshape(shp).astype(bool)
        subs.append(f.name.split('__')[0])
        L.append(d['logit'].astype(np.float32)); G.append(gt)
    return subs, L, G


def dice_bin(p, y):
    ps, ts = float(p.sum()), float(y.sum())
    if ts == 0:
        return 1.0 if ps == 0 else 0.0
    return float(2.0 * float((p & y).sum()) / (ps + ts))


def dice_cube(L, G):
    n = len(L)
    lt = np.log(GRID / (1 - GRID))
    D = np.zeros((n, 3, len(GRID)))
    for i in range(n):
        for r in range(3):
            lg, gt = L[i][r], G[i][r]
            for k, tl in enumerate(lt):
                D[i, r, k] = dice_bin(lg > tl, gt)
    return D


def nk(tau):
    return int(np.argmin(np.abs(GRID - tau)))


def loso_from_cube(D):
    n = D.shape[0]
    held = np.zeros((n, 3)); ch = []
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        ks = [int(np.argmax(D[m, r, :].mean(0))) for r in range(3)]
        ch.append([GRID[k] for k in ks])
        for r in range(3):
            held[i, r] = D[i, r, ks[r]]
    return held, np.array(ch)


def main():
    tags = sys.argv[1:] or ['tta', 'tta_e30', 'tta_e35']
    print(f'reference {REF:.7f}  target {REF+0.01:.7f}')
    print(f'fixed TTA tau: ET={TTA_TAU[0]} TC={TTA_TAU[1]} WT={TTA_TAU[2]}\n')

    data = {}
    for tg in tags:
        s, L, G = load(tg)
        if not s:
            print(f'  {tg}: NO CACHE'); continue
        data[tg] = (s, L, G)
    if not data:
        return
    ref_subs = data[tags[0]][0]

    # ---- 1. individual checkpoints ----
    print('=' * 92)
    print('1. INDIVIDUAL CHECKPOINTS (all with TTA)')
    print('=' * 92)
    cubes = {}
    for tg, (s, L, G) in data.items():
        D = dice_cube(L, G); cubes[tg] = D
        held, ch = loso_from_cube(D)
        print(f"  {tg:<10} tau=0.5 {D[:,:,nk(0.5)].mean():.6f} | LOSO {held.mean():.6f} "
              f"({100*(held.mean()-REF):+.3f} pp)  tau median "
              f"{[round(float(np.median(ch[:,i])),4) for i in range(3)]}")

    # ---- ensembles over every subset of size >= 2 ----
    print('\n' + '=' * 92)
    print('2-4. LOGIT-AVERAGED ENSEMBLES')
    print('=' * 92)
    print(f"{'members':<28}{'tau=0.5':>11}{'fixed TTA tau':>15}{'LOSO tau':>11}{'vs ref':>10}")
    best = None
    results = {}
    names = list(data.keys())
    for rsize in range(2, len(names) + 1):
        for combo in itertools.combinations(names, rsize):
            # verify subject alignment
            assert all(data[c][0] == ref_subs for c in combo), 'subject mismatch'
            n = len(ref_subs)
            Lens = []
            for i in range(n):
                Lens.append(np.mean([data[c][1][i] for c in combo], axis=0))
            G = data[combo[0]][2]
            D = dice_cube(Lens, G)
            a = D[:, :, nk(0.5)].mean()
            kk = [nk(t) for t in TTA_TAU]
            b = np.mean([D[:, r, kk[r]].mean() for r in range(3)])
            held, ch = loso_from_cube(D)
            c = held.mean()
            lbl = '+'.join(x.replace('tta_', '').replace('tta', 'best') for x in combo)
            print(f'  {lbl:<26}{a:>11.6f}{b:>15.6f}{c:>11.6f}{100*(c-REF):>+10.3f}')
            results[lbl] = dict(tau05=float(a), fixed=float(b), loso=float(c),
                                tau=[float(np.median(ch[:, i])) for i in range(3)])
            if best is None or c > best[1]:
                best = (lbl, c, held, ch)

    lbl, c, held, ch = best
    print(f'\n  BEST: {lbl}  LOSO {c:.7f}  ({100*(c-REF):+.4f} pp vs reference)')
    print(f'    per-region: ET {held[:,0].mean():.6f} TC {held[:,1].mean():.6f} WT {held[:,2].mean():.6f}')
    print(f'    tau median: ' + ', '.join(
        f'{REGIONS[i]} {np.median(ch[:,i]):.4f} [{ch[:,i].min():.4f},{ch[:,i].max():.4f}]' for i in range(3)))

    # paired bootstrap vs single-best-checkpoint LOSO
    single_held, _ = loso_from_cube(cubes[tags[0]])
    d = held.mean(1) - single_held.mean(1)
    rng = np.random.default_rng(0)
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(10000)])
    print(f'\n  vs single best checkpoint (both LOSO): {100*d.mean():+.4f} pp  '
          f'95% CI [{100*np.percentile(bs,2.5):+.4f}, {100*np.percentile(bs,97.5):+.4f}]  '
          f'improved {int((d>0).sum())}/{len(d)}')
    print(f'\n  TARGET CHECK: {c:.7f} vs {REF+0.01:.7f}  -> '
          f'{"REACHED" if c >= REF+0.01 else f"short by {100*(REF+0.01-c):.3f} pp"}')

    json.dump(results, open(HERE / 'CKPT_ensemble.json', 'w'), indent=2)
    print('\nwrote CKPT_ensemble.json')


if __name__ == '__main__':
    main()
