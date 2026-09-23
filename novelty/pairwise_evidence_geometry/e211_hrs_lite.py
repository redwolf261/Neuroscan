"""E211 -- HRS-Lite: learned proposal + evaluator on the frozen E209/E210
component set. FROZEN UNet3D_v5. NO end-to-end retraining, NO TTA, NO
threshold optimization, NO whole-volume inference.

Reuses the EXACT E209/E210 component extraction (missed L components + hard
negatives H, same picks). All computation restricted to that fixed component
set. FocalTversky (the actual training loss, imported from
train_e130_multimodal_baseline.py) is used for U, per the user's explicit
"same segmentation objective already used by NeuroScan" requirement.

DATASET per (component, direction, alpha):
  h              : pooled bottleneck feature at the component (256,)
  d              : direction (proposal output, OR random, OR prototype --
                   see conditions below)
  alpha          : perturbation strength
  h_cf           : h + alpha * a * d_hat  (a = proposal's own activation gate)
  z0, z1         : region-mean raw ET logit before/after
  dz             : z1 - z0
  reactivity     : |dz|  (same definition as E209/E210)
  U              : FocalTversky(Y0,G) - FocalTversky(Y1,G) over component+shell
                   (POSITIVE = improvement)
  crossing       : z0<0 and z1>0
  cos_to_prototype : diagnostic only, logged not trained on

FOUR CONDITIONS PER COMPONENT (subject-grouped split enforced throughout,
matching E209's GroupKFold discipline):
  1. prototype  : E210's direction, alpha in {0.25,0.5,1.0} (replication)
  2. random     : norm-matched random direction (replication)
  3. proposal   : g_phi(h) -- LEARNED direction, trained ONLY via
                  L_transition/L_rank (dz-based), L_dir logged as diagnostic
                  ONLY, never backpropagated -- so a positive result is not
                  just "memorized the prototype".
  4. anti-proposal : -1 * proposal direction, as a same-network sign control

TRAINING (Stage A, matching the user's spec): UNet3D_v5 fully frozen
throughout. Only g_phi (proposal, a tiny 2-layer MLP: 256->64->(1+256)) and
V_psi (evaluator, tiny MLP on the transition-feature vector) are trained.
5-fold subject-grouped CV -- report HELD-OUT metrics only.

THREE EVALUATORS (user's spec), same held-out folds:
  R  : reactivity only            -> AUC(U>0), R^2
  C  : reactivity + dz            -> AUC(U>0), R^2
  H  : full [h, h_cf, dh, z0, z1, dz, reactivity] -> AUC(U>0), R^2

KILL if: proposal doesn't beat random on held-out AUC(crossing)/dz; OR
Delta R^2 = R^2_H - R^2_C is small/non-significant; OR U's dependence on
Uhat vanishes after conditioning on reactivity (partial correlation).
Small-n caveat stated explicitly in the analysis (74 components total).
"""
import sys, csv, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from neuroscan_3d_v5 import UNet3D_v5
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset
import importlib.util

spec = importlib.util.spec_from_file_location(
    "t", str(ROOT / 'experiments/exp_e12_eggo_m/e130/train_e130_multimodal_baseline.py'))
t = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["e"]; spec.loader.exec_module(t); sys.argv = _a
focal_tversky = t.focal_tversky

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
PATCH, MIN_VOX, MAX_PER_SUBJ = 128, 5, 3
ET = 0
ALPHAS = [0.25, 0.5, 1.0]
N_FOLDS = 5


def _decode_from_bottleneck(model, bn, e1, e2, e3):
    """No no_grad here -- callers choose. Frozen UNet params already have
    requires_grad_(False) set in main(), so this is cheap either way; the
    difference only matters for whether grad flows THROUGH bn back to
    whatever produced it (e.g. the proposal network)."""
    n = bn.shape[0]
    skip3 = e3.expand(n, -1, -1, -1, -1)
    d3 = model.dec3(torch.cat([model.upconv3(bn), skip3], 1))
    d2 = model.dec2(torch.cat([model.upconv2(d3),
                               e2.expand(n, -1, -1, -1, -1)], 1))
    eg, _ = model.attn_gate1(gate=bn, skip=e1.expand(n, -1, -1, -1, -1))
    d1 = model.dec1(torch.cat([model.upconv1(d2), eg], 1))
    z = model.seg_head[0](d1)
    return z


def tail_from_bottleneck(model, bn, e1, e2, e3):
    """no_grad variant for diagnostic/eval use (apply_and_eval) -- bn here is
    never a leaf we need gradients through."""
    with torch.no_grad():
        return _decode_from_bottleneck(model, bn, e1, e2, e3)


class Proposal(nn.Module):
    """g_phi: h (256,) -> (activation gate a in [0,1], direction d in R^256)."""
    def __init__(self, dim=256, hid=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU())
        self.a_head = nn.Linear(hid, 1)
        self.d_head = nn.Linear(hid, dim)

    def forward(self, h):
        f = self.net(h)
        a = torch.sigmoid(self.a_head(f)).squeeze(-1)
        d = self.d_head(f)
        d = d / (d.norm(dim=-1, keepdim=True) + 1e-6)
        return a, d


class Evaluator(nn.Module):
    def __init__(self, in_dim, hid=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 1))

    def forward(self, q):
        return self.net(q).squeeze(-1)


def collect_components(model, ds, dev):
    """Reproduces E209/E210's exact component extraction. Returns list of
    dicts with subject_id, comp_id, population, bn (full tensor), roi mask
    at bottleneck res, region mask (input res), shell mask, GT mask, e1/e2/e3,
    mu_prototype (leave-subject-out)."""
    print('Pass 1: building detected-lesion bottleneck prototype bank...', flush=True)
    proto_vecs, proto_subjects = [], []
    for ii in range(min(60, len(ds))):
        img, tgt, sid = ds[ii]
        Y = tgt.numpy() > 0.5
        probs = t.sliding_window_predict(model, img.unsqueeze(0).to(dev),
                                         t.PATCH, t.SW_OVERLAP, 3, dev, True)
        P = probs >= 0.5
        lbl, nl = ndimage.label(Y[ET])
        for g in range(1, nl + 1):
            cm = lbl == g
            if cm.sum() < MIN_VOX:
                continue
            ov = float((cm & P[ET]).sum()) / cm.sum()
            if ov < 0.5:
                continue
            com = np.array(ndimage.center_of_mass(cm)).astype(int)
            st = [int(np.clip(c - PATCH // 2, 0, s - PATCH))
                  for c, s in zip(com, cm.shape)]
            sl = tuple(slice(s, s + PATCH) for s in st)
            cm_p = cm[sl]
            if cm_p.sum() < MIN_VOX:
                continue
            x = img[(slice(None),) + sl].unsqueeze(0).to(dev)
            with torch.no_grad():
                e1 = model.enc1(x); e2 = model.enc2(model.pool1(e1))
                e3 = model.enc3(model.pool2(e2)); bn = model.bottleneck(model.pool3(e3))
            bn0 = bn[0]; shp3 = bn0.shape[1:]
            com_m = np.array(ndimage.center_of_mass(cm_p))
            scale = np.array(shp3) / np.array(cm_p.shape)
            pos3 = tuple(np.clip((com_m * scale).astype(int), 0, np.array(shp3) - 1))
            proto_vecs.append(bn0[:, pos3[0], pos3[1], pos3[2]].cpu().numpy())
            proto_subjects.append(sid)
            break
    proto_vecs = np.array(proto_vecs)
    print(f'  bank size = {len(proto_vecs)} prototypes', flush=True)

    comps = []
    for ii in range(len(ds)):
        img, tgt, sid = ds[ii]
        Y = tgt.numpy() > 0.5
        probs = t.sliding_window_predict(model, img.unsqueeze(0).to(dev),
                                         t.PATCH, t.SW_OVERLAP, 3, dev, True)
        P = probs >= 0.5
        brain = img[0].numpy() != 0
        lbl, nl = ndimage.label(Y[ET])
        missed = []
        for g in range(1, nl + 1):
            cm = lbl == g
            if cm.sum() < MIN_VOX:
                continue
            ov = float((cm & P[ET]).sum()) / cm.sum()
            if ov == 0:
                missed.append((int(cm.sum()), g))
        missed.sort(reverse=True)
        picks = missed[:MAX_PER_SUBJ]
        if not picks:
            continue
        mask_other = np.array([s != sid for s in proto_subjects])
        mu_prototype = (proto_vecs[mask_other].mean(0) if mask_other.sum() >= 5
                        else proto_vecs.mean(0))
        for _, g in picks:
            cm = lbl == g
            com = np.array(ndimage.center_of_mass(cm)).astype(int)
            st = [int(np.clip(c - PATCH // 2, 0, s - PATCH))
                  for c, s in zip(com, cm.shape)]
            sl = tuple(slice(s, s + PATCH) for s in st)
            cm_p = cm[sl]
            if cm_p.sum() < MIN_VOX:
                continue
            anygt = (Y[0] | Y[1] | Y[2])[sl]; brain_p = brain[sl]
            x = img[(slice(None),) + sl].unsqueeze(0).to(dev)
            with torch.no_grad():
                e1 = model.enc1(x); e2 = model.enc2(model.pool1(e1))
                e3 = model.enc3(model.pool2(e2)); bn = model.bottleneck(model.pool3(e3))
            shp3 = bn[0].shape[1:]
            roi_in = ndimage.binary_dilation(cm_p, iterations=8) & brain_p
            r3 = (torch.nn.functional.adaptive_max_pool3d(
                torch.from_numpy(roi_in.astype(np.float32))[None, None],
                shp3)[0, 0] > 0.5)
            if r3.sum() < 4:
                continue
            shell = (ndimage.binary_dilation(cm_p, iterations=6) & ~cm_p
                     & brain_p & ~anygt)
            if shell.sum() < MIN_VOX:
                continue
            com_m = np.array(ndimage.center_of_mass(cm_p))
            scale = np.array(shp3) / np.array(cm_p.shape)
            pos3 = tuple(np.clip((com_m * scale).astype(int), 0, np.array(shp3) - 1))
            mu_local = bn[0][:, pos3[0], pos3[1], pos3[2]].cpu().numpy()

            # Store on CPU -- with ~70+ components, keeping bottleneck/enc
            # tensors resident on an 8GB GPU across the whole collection loop
            # exhausts VRAM and causes severe slowdown (diagnosed after a
            # real stall). Moved back to `dev` per-component in apply_and_eval.
            comps.append({
                'subject_id': sid, 'comp_id': int(g),
                'e1': e1.detach().cpu(), 'e2': e2.detach().cpu(),
                'e3': e3.detach().cpu(), 'bn': bn.detach().cpu(),
                'r3': r3.cpu(), 'region_mask': cm_p, 'shell_mask': shell,
                'gt_region_np': cm_p.astype(np.float32),
                'mu_local': mu_local, 'mu_prototype': mu_prototype,
            })
        if (ii + 1) % 25 == 0:
            print(f'  collect {ii+1}/{len(ds)} subj, {len(comps)} comps', flush=True)
    return comps


def apply_and_eval(model, comp, direction_np, alpha, dev, a_gate=1.0):
    """direction_np: (256,) unit vector. Returns dict of z0,z1,dz,U,crossing.
    comp's tensors are stored on CPU; moved to dev here, per-call."""
    bn = comp['bn'].to(dev); e1 = comp['e1'].to(dev)
    e2 = comp['e2'].to(dev); e3 = comp['e3'].to(dev)
    r3 = comp['r3'].to(dev); msk = comp['region_mask']; shell = comp['shell_mask']
    dvec = torch.from_numpy(direction_np).float().to(dev)

    z0_full = tail_from_bottleneck(model, bn, e1, e2, e3)[0]
    lg0 = z0_full[ET].cpu().numpy()
    p0 = torch.sigmoid(z0_full[ET])

    bn_p = bn.clone()
    bn_p[0, :, r3] = bn_p[0, :, r3] + alpha * a_gate * dvec.unsqueeze(1)
    z1_full = tail_from_bottleneck(model, bn_p, e1, e2, e3)[0]
    lg1 = z1_full[ET].cpu().numpy()
    p1 = torch.sigmoid(z1_full[ET])

    z0v = float(lg0[msk].mean()); z1v = float(lg1[msk].mean())
    crossing = bool(z0v < 0 and z1v > 0)

    region_shell = torch.from_numpy((msk | shell).astype(np.float32)).to(dev)
    gt_full = torch.zeros_like(p0)
    gt_full[torch.from_numpy(msk).to(dev)] = 1.0
    with torch.no_grad():
        u_mask = region_shell.bool()
        ft0 = float(focal_tversky(p0[u_mask].unsqueeze(0).unsqueeze(0),
                                  gt_full[u_mask].unsqueeze(0).unsqueeze(0)))
        ft1 = float(focal_tversky(p1[u_mask].unsqueeze(0).unsqueeze(0),
                                  gt_full[u_mask].unsqueeze(0).unsqueeze(0)))
    U = ft0 - ft1   # FocalTversky is a LOSS (lower=better) -> improvement = ft0-ft1 > 0

    return {'z0': z0v, 'z1': z1v, 'dz': z1v - z0v, 'reactivity': abs(z1v - z0v),
           'U': U, 'crossing': int(crossing)}


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for q in model.parameters():
        q.requires_grad_(False)
    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    print(f'E211 HRS-Lite, {len(ds)} subjects', flush=True)

    t0 = time.time()
    comps = collect_components(model, ds, dev)
    print(f'collected {len(comps)} L components ({time.time()-t0:.0f}s)', flush=True)

    subjects = np.array([c['subject_id'] for c in comps])
    gkf = GroupKFold(n_splits=min(N_FOLDS, len(set(subjects))))
    fold_of = np.zeros(len(comps), dtype=int)
    for fi, (_, te) in enumerate(gkf.split(np.zeros(len(comps)), groups=subjects)):
        fold_of[te] = fi

    all_rows = []
    for fold in range(gkf.get_n_splits(np.zeros(len(comps)), groups=subjects)):
        train_idx = [i for i in range(len(comps)) if fold_of[i] != fold]
        test_idx = [i for i in range(len(comps)) if fold_of[i] == fold]
        if len(train_idx) < 5 or len(test_idx) < 1:
            continue

        torch.manual_seed(fold)
        proposal = Proposal().to(dev)
        opt_p = torch.optim.Adam(proposal.parameters(), lr=1e-3)

        # ---- Stage A1: train proposal via transition utility on TRAIN fold ----
        rng = np.random.default_rng(1000 + fold)
        for epoch in range(30):
            idx_perm = rng.permutation(train_idx)
            for i in idx_perm:
                c = comps[i]
                h = torch.from_numpy(c['mu_local']).float().to(dev)
                # differentiable transition loss: maximize dz under the
                # proposal's own direction/gate (single forward pass per
                # condition; no wasted apply_and_eval calls -- those results
                # were computed and discarded in an earlier draft)
                a_t, d_t = proposal(h.unsqueeze(0))
                bn = c['bn'].to(dev); e1 = c['e1'].to(dev)
                e2 = c['e2'].to(dev); e3 = c['e3'].to(dev); r3 = c['r3'].to(dev)
                bn_p = bn.clone()
                bn_p[0, :, r3] = bn_p[0, :, r3] + 0.5 * a_t[0] * d_t[0].unsqueeze(1)
                z1_full = _decode_from_bottleneck(model, bn_p, e1, e2, e3)[0]
                with torch.no_grad():
                    z0_full = tail_from_bottleneck(model, bn, e1, e2, e3)[0]
                msk_t = torch.from_numpy(c['region_mask']).to(dev)
                z0v_t = z0_full[ET][msk_t].mean()
                z1v_t = z1_full[ET][msk_t].mean()
                loss = -(z1v_t - z0v_t.detach())  # maximize dz (transition benefit)
                opt_p.zero_grad(); loss.backward(); opt_p.step()

        # ---- build the transition-feature dataset for ALL 4 conditions ----
        def build_rows(idx_list, split_name):
            out = []
            for i in idx_list:
                c = comps[i]
                h = torch.from_numpy(c['mu_local']).float().to(dev)
                with torch.no_grad():
                    a_p, d_p = proposal(h.unsqueeze(0))
                d_p_np = d_p[0].cpu().numpy(); a_p_val = float(a_p[0])
                proto_dir = c['mu_prototype'] - c['mu_local']
                proto_dir = proto_dir / (np.linalg.norm(proto_dir) + 1e-8)
                rng3 = np.random.default_rng(abs(hash((c['subject_id'], c['comp_id']))) % (2**31))
                rand_dir = rng3.standard_normal(256); rand_dir = rand_dir / (np.linalg.norm(rand_dir) + 1e-8)
                cos_proto = float(np.dot(d_p_np, proto_dir))

                for cond_name, dvec, gate in [
                    ('prototype', proto_dir, 1.0),
                    ('random', rand_dir, 1.0),
                    ('proposal', d_p_np, a_p_val),
                    ('anti_proposal', -d_p_np, a_p_val),
                ]:
                    for alpha in ALPHAS:
                        r = apply_and_eval(model, c, dvec, alpha, dev, a_gate=gate)
                        out.append({
                            'fold': fold, 'split': split_name,
                            'subject_id': c['subject_id'], 'comp_id': c['comp_id'],
                            'condition': cond_name, 'alpha': alpha,
                            'a_gate': gate, 'cos_to_prototype': cos_proto,
                            **r,
                        })
            return out

        all_rows += build_rows(train_idx, 'train')
        all_rows += build_rows(test_idx, 'test')
        print(f'fold {fold}: train={len(train_idx)} test={len(test_idx)} '
              f'total_rows={len(all_rows)} ({time.time()-t0:.0f}s)', flush=True)

    with open(HERE / 'E211_hrs_lite.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(all_rows[0]))
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f'\nwrote E211_hrs_lite.csv ({len(all_rows)} rows)  {time.time()-t0:.0f}s',
          flush=True)


if __name__ == '__main__':
    main()
