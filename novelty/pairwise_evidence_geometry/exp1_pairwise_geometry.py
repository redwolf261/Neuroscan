"""EXPERIMENT 1 -- Pairwise Evidence Geometry.

Question: does the frozen segmentation head preserve a recoverable RELATIONAL
signal inside components it absolutely rejects?

Populations (all matched, per the forensics protocol):
  L : missed ET/TC GT components (zero predicted overlap)
  A : adjacent 1-6 voxel background shell, confirmed non-GT
  H : hard negative -- the component's OWN mask translated to the max-logit
      background location, so size AND shape are identical by construction

Statistics, all computed on the FROZEN logit field z(x):
  A. Pi(C)    = P[ z(x) > z(b) ]                      pairwise ordering
  B. M(C)     = mean sigma( (z(x)-z(b)) / tau )       margin-weighted
  C. Pi_q(C)  = P[ z(x) > Q_q(z_B) ]  for q in .5 .75 .9 .95   tail ordering

Four tests:
  1. L vs H  -- is relational evidence lesion-specific?
  2. L vs A  -- is the effect local?
  3. LOSO    -- does it generalise across subjects?
  4. D vs R  -- does relational evidence scale with absolute failure?
                D(C) = -median(z_C);  R(C) = median(z_C) - median(z_B)

Controls: partial correlations against size / mean logit / location, and a
subject-preserving permutation null.

NO TRAINING. GT used only to locate regions and to label rows.
"""
import sys, csv, json, time
from pathlib import Path
import numpy as np
from scipy import ndimage, stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / 'experiments' / 'exp_dice' / 'cache'
FORENSIC = ROOT / 'experiments' / 'exp_forensics' / 'FORENSIC_per_subject.csv'
REGIONS = ('ET', 'TC', 'WT')
MIN_VOX, MAX_COMP = 5, 6
QS = (0.5, 0.75, 0.9, 0.95)
TAUS = (1.0, 2.0)
MAX_PAIR = 4000      # subsample cap per side for the pairwise stats


def load_subject(sid, tag='base'):
    """Cached logits + GT, plus a REAL brain mask from the t1c volume.
    (A logit-derived mask is useless: cached logits are dense, so |z|>0 is
    identically true everywhere.)"""
    f = CACHE / f'{sid}__{tag}.npz'
    if not f.exists():
        return None
    d = np.load(f)
    shp = tuple(d['gt_shape'])
    gt = np.unpackbits(d['gt'])[:int(np.prod(shp))].reshape(shp).astype(bool)
    brain = None
    try:
        import nibabel as nib
        dd = ROOT / 'Dataset/Training/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData' / sid
        v = nib.load(str(dd / f'{sid}-t1c.nii.gz')).get_fdata()
        # cached volumes are brain-bbox cropped; match by cropping the mask the
        # same way the cache builder did (nonzero bounding box).
        nz = np.argwhere(v > 0)
        lo = nz.min(0); hi = nz.max(0) + 1
        brain = (v > 0)[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        if brain.shape != tuple(shp)[1:]:      # shp is 4D (3,D,H,W); mask is 3D
            brain = None
    except Exception:
        brain = None
    if brain is None:
        brain = np.ones(tuple(shp)[1:], bool)
    return d['logit'].astype(np.float32), gt, brain


def pairwise_stats(zc, zb, rng):
    """Pi, margin-weighted M, and tail Pi_q. Subsampled for tractability."""
    if len(zc) > MAX_PAIR:
        zc = rng.choice(zc, MAX_PAIR, replace=False)
    if len(zb) > MAX_PAIR:
        zb = rng.choice(zb, MAX_PAIR, replace=False)
    out = {}
    # Pi via rank identity (exact, O(n log n)) -- equals mean 1[z_x > z_b]
    n1, n0 = len(zc), len(zb)
    allv = np.concatenate([zc, zb])
    r = stats.rankdata(allv)
    out['Pi'] = float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    # margin-weighted (explicit, on the subsample)
    diff = zc[:, None] - zb[None, :]
    for tau in TAUS:
        out[f'M_tau{tau}'] = float((1.0 / (1.0 + np.exp(-diff / tau))).mean())
    # tail ordering
    for q in QS:
        out[f'Pi_q{q}'] = float((zc > np.quantile(zb, q)).mean())
    return out


def main():
    t0 = time.time()
    fr = list(csv.DictReader(open(FORENSIC)))
    tail = [r['subject_id'] for r in fr
            if float(r['ET_dice']) < 0.5 or float(r['TC_dice']) < 0.5]
    print(f'tail subjects: {len(tail)}')

    rows = []
    for si, sid in enumerate(tail):
        got = load_subject(sid)
        if got is None:
            print(f'  !! no cache for {sid}'); continue
        LOG, Y, brain = got
        # the frozen decision at the campaign baseline rule
        P = LOG > 0.0        # logit > 0 == prob > 0.5
        for ri, rn in [(0, 'ET'), (1, 'TC')]:
            lg = LOG[ri]
            lbl, nl = ndimage.label(Y[ri])
            miss = sorted([(int((lbl == g).sum()), g) for g in range(1, nl + 1)
                           if (lbl == g).sum() >= MIN_VOX and not ((lbl == g) & P[ri]).any()],
                          reverse=True)[:MAX_COMP]
            for sz, g in miss:
                cm = lbl == g
                anygt = Y[0] | Y[1] | Y[2]
                dist = ndimage.distance_transform_edt(~cm)
                adj = (dist > 1) & (dist <= 6) & ~anygt & brain
                if adj.sum() < MIN_VOX:
                    continue
                # hard negative: translate the component's own mask to max-logit bg
                rng = np.random.default_rng(abs(hash((sid, rn, g))) % (2**31))
                forbid = ndimage.binary_dilation(anygt, iterations=6)
                idx = np.argwhere(cm); lo = idx.min(0); ext = idx.max(0) - lo + 1
                rel = idx - lo; dims = np.array(cm.shape)
                best, bestv = None, -np.inf
                for _ in range(300):
                    hi = dims - ext
                    if (hi <= 0).any():
                        break
                    off = np.array([rng.integers(0, h + 1) for h in hi])
                    p = rel + off
                    cand = np.zeros_like(cm); cand[tuple(p.T)] = True
                    if (cand & forbid).any() or not brain[cand].all():
                        continue
                    v = float(lg[cand].mean())
                    if v > bestv:
                        bestv, best = v, cand
                if best is None or best.sum() < MIN_VOX:
                    continue
                # shared background reference for every population: the shell
                zb = lg[adj].astype(np.float64)
                # hard-negative's own shell (for its relational stat)
                dh = ndimage.distance_transform_edt(~best)
                adjh = (dh > 1) & (dh <= 6) & ~anygt & brain
                if adjh.sum() < MIN_VOX:
                    continue
                zbh = lg[adjh].astype(np.float64)
                for pop, mask, bg in [('LESION', cm, zb), ('ADJ', adj, zb),
                                      ('HARDNEG', best, zbh)]:
                    zc = lg[mask].astype(np.float64)
                    if len(zc) < MIN_VOX or len(bg) < MIN_VOX:
                        continue
                    st = pairwise_stats(zc, bg, rng)
                    st.update({
                        'subject_id': sid, 'region': rn, 'comp_id': int(g),
                        'population': pop, 'n_vox': int(mask.sum()),
                        'comp_vox': int(cm.sum()),
                        'median_z': float(np.median(zc)),
                        'mean_z': float(zc.mean()),
                        'median_zb': float(np.median(bg)),
                        'D': float(-np.median(zc)),
                        'R': float(np.median(zc) - np.median(bg)),
                        'centroid_z': float(ndimage.center_of_mass(mask)[0]),
                    })
                    rows.append(st)
        if (si + 1) % 4 == 0:
            print(f'  {si+1}/{len(tail)} subj, {len(rows)} rows ({time.time()-t0:.0f}s)', flush=True)

    keys = sorted({k for r in rows for k in r})
    out = HERE / 'EXP1_pairwise.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'\nwrote {out.name} ({len(rows)} rows) in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
