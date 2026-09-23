"""Three-way attribution across inference configurations.

For every cached config (tag) report:
  (a) fixed tau=0.5            -- raw effect of the inference change
  (b) fixed OLD tau from base  -- pure inference gain at the current operating point
  (c) LOSO re-optimised tau    -- combined gain

(b) is the one that prevents attributing threshold-calibration gains to TTA.
Reference baseline for the whole campaign stays 0.859519 (base, tau=0.5).
"""
import sys, json
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
REF = 0.8595193          # campaign reference: base config, tau=0.5


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
    """D[i,r,k] for all thresholds."""
    n = len(L)
    lg_thr = np.log(GRID / (1 - GRID))
    D = np.zeros((n, 3, len(GRID)))
    for i in range(n):
        for r in range(3):
            lg, gt = L[i][r], G[i][r]
            for k, tl in enumerate(lg_thr):
                D[i, r, k] = dice_bin(lg > tl, gt)
    return D


def nearest_k(tau):
    return int(np.argmin(np.abs(GRID - tau)))


def evaluate(tag, old_tau):
    subs, L, G = load(tag)
    if not subs:
        return None
    D = dice_cube(L, G)
    n = len(subs)
    k05 = nearest_k(0.5)
    a = D[:, :, k05].mean(0)
    ko = [nearest_k(t) for t in old_tau]
    b = np.array([D[:, r, ko[r]].mean() for r in range(3)])
    # LOSO re-optimised
    held = np.zeros((n, 3)); chosen = []
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        ks = [int(np.argmax(D[m, r, :].mean(0))) for r in range(3)]
        chosen.append([GRID[k] for k in ks])
        for r in range(3):
            held[i, r] = D[i, r, ks[r]]
    c = held.mean(0)
    ins_k = [int(np.argmax(D[:, r, :].mean(0))) for r in range(3)]
    ins = np.array([D[:, r, ins_k[r]].mean() for r in range(3)])
    return dict(n=n, tau05=a, tau_old=b, loso=c, insample=ins,
                loso_tau=np.array(chosen), ins_tau=[GRID[k] for k in ins_k],
                held=held, D=D, k05=k05)


def main():
    tags = sys.argv[1:] or ['base', 'tta']
    old_tau = (0.02, 0.005, 0.70)
    print(f'campaign reference (base, tau=0.5): {REF:.7f}   target {REF+0.01:.7f}')
    print(f'old tau (from base LOSO): ET={old_tau[0]} TC={old_tau[1]} WT={old_tau[2]}\n')
    print('=' * 100)
    print(f"{'config':<10}{'n':>4}{'(a) tau=.5':>13}{'(b) old tau':>13}{'(c) LOSO tau':>14}"
          f"{'(a)-ref':>10}{'(b)-ref':>10}{'(c)-ref':>10}")
    print('=' * 100)
    res = {}
    for tg in tags:
        r = evaluate(tg, old_tau)
        if r is None:
            print(f'{tg:<10}  (no cache)')
            continue
        res[tg] = r
        a3, b3, c3 = r['tau05'].mean(), r['tau_old'].mean(), r['loso'].mean()
        print(f"{tg:<10}{r['n']:>4}{a3:>13.6f}{b3:>13.6f}{c3:>14.6f}"
              f"{100*(a3-REF):>+10.3f}{100*(b3-REF):>+10.3f}{100*(c3-REF):>+10.3f}")
    print()
    for tg, r in res.items():
        print(f"  {tg}: per-region @LOSO  ET {r['loso'][0]:.6f}  TC {r['loso'][1]:.6f}  "
              f"WT {r['loso'][2]:.6f}   | in-sample tau {[round(float(x),4) for x in r['ins_tau']]}")
        ch = r['loso_tau']
        print(f"       tau stability: " + ', '.join(
            f"{REGIONS[i]} {np.median(ch[:,i]):.4f} [{ch[:,i].min():.4f},{ch[:,i].max():.4f}]"
            for i in range(3)))

    # paired comparison base vs tta at matched decision rule
    if 'base' in res and 'tta' in res:
        print('\n' + '=' * 100)
        print('PAIRED: TTA vs BASE at the SAME decision rule (isolates the inference change)')
        print('=' * 100)
        for nm, key in [('tau=0.5', 'tau05'), ('old tau', 'tau_old')]:
            kk = res['base']['k05'] if key == 'tau05' else None
            if key == 'tau05':
                db = res['base']['D'][:, :, res['base']['k05']].mean(1)
                dt = res['tta']['D'][:, :, res['tta']['k05']].mean(1)
            else:
                ko = [nearest_k(t) for t in old_tau]
                db = np.stack([res['base']['D'][:, r, ko[r]] for r in range(3)], 1).mean(1)
                dt = np.stack([res['tta']['D'][:, r, ko[r]] for r in range(3)], 1).mean(1)
            d = dt - db
            rng = np.random.default_rng(0)
            bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(10000)])
            print(f'  {nm:<10} mean {100*d.mean():+.4f} pp  '
                  f'95% CI [{100*np.percentile(bs,2.5):+.4f}, {100*np.percentile(bs,97.5):+.4f}]  '
                  f'improved {int((d>0).sum())}/{len(d)}')
        # LOSO-vs-LOSO
        d = res['tta']['held'].mean(1) - res['base']['held'].mean(1)
        rng = np.random.default_rng(0)
        bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(10000)])
        print(f'  {"LOSO tau":<10} mean {100*d.mean():+.4f} pp  '
              f'95% CI [{100*np.percentile(bs,2.5):+.4f}, {100*np.percentile(bs,97.5):+.4f}]  '
              f'improved {int((d>0).sum())}/{len(d)}')

    json.dump({t: {'tau05': float(r['tau05'].mean()), 'tau_old': float(r['tau_old'].mean()),
                   'loso': float(r['loso'].mean())} for t, r in res.items()},
              open(HERE / 'CONFIG_compare.json', 'w'), indent=2)
    print('\nwrote CONFIG_compare.json')


if __name__ == '__main__':
    main()
