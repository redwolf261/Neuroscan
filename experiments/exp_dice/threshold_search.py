"""Per-region decision-threshold optimisation on the frozen baseline.

Reports TWO numbers, deliberately:
  IN-SAMPLE : thresholds chosen on all 125 subjects, scored on the same 125.
              This is what a naive report would show. It is fitting on the
              evaluation set and is an OPTIMISTIC BOUND, not an improvement.
  LOSO      : for each subject, thresholds are chosen on the OTHER 124 and
              applied to the held-out one. This is what actually generalises.

If the two diverge, the gap IS the overfitting.
"""
import sys, json, csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / 'cache'
REGIONS = ('ET', 'TC', 'WT')
TAG = sys.argv[1] if len(sys.argv) > 1 else 'base'
# extended: ET/TC hit the 0.05 floor on the coarse grid, so the search was
# truncated rather than converged. Go much lower, log-spaced at the bottom.
# EXTENDED AGAIN: TC pinned at the 1e-4 floor, so the search was truncated,
# not converged. Extend to 1e-8. (Cheap: cached logits, no re-inference.)
GRID = np.unique(np.round(np.concatenate([
    np.logspace(-8, -1.05, 40),
    np.arange(0.10, 0.96, 0.05)]), 10))


def load(tag):
    subs, L, G = [], [], []
    for f in sorted(CACHE.glob(f'*__{tag}.npz')):
        d = np.load(f)
        shp = tuple(d['gt_shape'])
        gt = np.unpackbits(d['gt'])[:int(np.prod(shp))].reshape(shp).astype(bool)
        subs.append(f.name.split('__')[0])
        L.append(d['logit'].astype(np.float32))
        G.append(gt)
    return subs, L, G


def dice_bin(p, y):
    ps, ts = float(p.sum()), float(y.sum())
    if ts == 0:
        return 1.0 if ps == 0 else 0.0
    return float(2.0 * float((p & y).sum()) / (ps + ts))


def main():
    subs, L, G = load(TAG)
    n = len(subs)
    print(f'tag={TAG}  subjects={n}')

    # precompute Dice for every (subject, region, threshold) once
    lg_thr = np.log(GRID / (1 - GRID))          # threshold in logit space
    D = np.zeros((n, 3, len(GRID)), dtype=np.float64)
    for i in range(n):
        for r in range(3):
            lg = L[i][r]; gt = G[i][r]
            for k, tl in enumerate(lg_thr):
                D[i, r, k] = dice_bin(lg > tl, gt)
    base_k = int(np.argmin(np.abs(GRID - 0.5)))
    base = D[:, :, base_k].mean(0)
    base3 = base.mean()
    print(f'\nBASELINE (tau=0.5 everywhere): ET {base[0]:.6f} TC {base[1]:.6f} '
          f'WT {base[2]:.6f} | 3-region {base3:.7f}')
    print(f'TARGET: {base3+0.01:.7f}\n')

    # ---------- IN-SAMPLE (optimistic bound) ----------
    print('=' * 78)
    print('IN-SAMPLE threshold search (FITS ON THE EVALUATION SET -- optimistic bound)')
    print('=' * 78)
    best_k = [int(np.argmax(D[:, r, :].mean(0))) for r in range(3)]
    ins = np.array([D[:, r, best_k[r]].mean() for r in range(3)])
    ins3 = ins.mean()
    print(f'  best tau: ' + ', '.join(f'{REGIONS[r]}={GRID[best_k[r]]:.2f}' for r in range(3)))
    print(f'  per-region Dice: ET {ins[0]:.6f} TC {ins[1]:.6f} WT {ins[2]:.6f}')
    print(f'  3-region: {ins3:.7f}   delta {100*(ins3-base3):+.4f} pp')

    # per-region sweep for visibility
    print('\n  sweep (mean Dice per region at each tau):')
    print('   tau   ' + ''.join(f'{REGIONS[r]:>10}' for r in range(3)))
    for k, g in enumerate(GRID):
        if k % 2:
            continue
        print(f'   {g:.2f}  ' + ''.join(f'{D[:,r,k].mean():>10.5f}' for r in range(3)))

    # ---------- LOSO (honest) ----------
    print('\n' + '=' * 78)
    print('LOSO threshold search (thresholds fit on other 124, applied to held-out)')
    print('=' * 78)
    held = np.zeros((n, 3))
    chosen = []
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        ks = [int(np.argmax(D[m, r, :].mean(0))) for r in range(3)]
        chosen.append([GRID[k] for k in ks])
        for r in range(3):
            held[i, r] = D[i, r, ks[r]]
    lo = held.mean(0); lo3 = lo.mean()
    print(f'  per-region Dice: ET {lo[0]:.6f} TC {lo[1]:.6f} WT {lo[2]:.6f}')
    print(f'  3-region: {lo3:.7f}   delta {100*(lo3-base3):+.4f} pp')
    ch = np.array(chosen)
    print(f'  chosen tau stability: ' +
          ', '.join(f'{REGIONS[r]} median {np.median(ch[:,r]):.2f} '
                    f'(range {ch[:,r].min():.2f}-{ch[:,r].max():.2f})' for r in range(3)))

    print(f'\n  OVERFIT GAP (in-sample minus LOSO): {100*(ins3-lo3):+.4f} pp')

    # ---------- paired bootstrap on the LOSO gain ----------
    d = held.mean(1) - D[:, :, base_k].mean(1)
    rng = np.random.default_rng(0)
    bs = np.array([rng.choice(d, n, replace=True).mean() for _ in range(10000)])
    print(f'  LOSO per-subject gain: mean {100*d.mean():+.4f} pp, '
          f'95% CI [{100*np.percentile(bs,2.5):+.4f}, {100*np.percentile(bs,97.5):+.4f}] pp')
    print(f'  subjects improved: {int((d>0).sum())}/{n}')

    json.dump({'tag': TAG, 'base3': base3, 'insample3': ins3, 'loso3': lo3,
               'best_tau': [float(GRID[k]) for k in best_k],
               'loso_gain_pp': float(100 * d.mean())},
              open(HERE / f'THRESH_{TAG}.json', 'w'), indent=2)
    print(f'\nwrote THRESH_{TAG}.json')


if __name__ == '__main__':
    main()
