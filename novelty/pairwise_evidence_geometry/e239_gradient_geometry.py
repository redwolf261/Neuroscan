"""E239 -- gradient-geometry decomposition, testing H7 (lesion/background
gradient cancellation), per the user's exact spec (pasted_content,
2026-09-29). MEASUREMENT ONLY -- model weights are never updated, only
per-region gradients of the FULL production loss are computed and
compared.

H7's premise, motivated directly by E238's own two findings (loss cares
about missed lesions, D1 gradient is NOT suppressed) plus E232 (confident
absence): "gradient exists != gradient survives aggregation." The lesion
gradient g_L may be real and undiminished on its own, yet be neutralized
or opposed by the aggregate gradient from everything else in the same
training step.

LOSS DECOMPOSED (per explicit user answer to this turn's disambiguation
question): the FULL production per-step loss, all three terms, exactly
as train_e130/e228a compute it --
    loss = seg + MU * boundary + LAMBDA_DS3 * aux3
  where seg = 0.5*focal_tversky(probs,targets) + 0.5*evidential_beta(...)
  (full-res, ET channel), boundary = BCEWithLogitsLoss(boundary_logit,
  targets) (full-res), aux3 = focal_tversky(aux_probs3, D4-downsampled
  targets) (K=4 downsampled grid). MU=0.1, LAMBDA_DS3=0.9927, K=4 --
  all copied verbatim from e228a_conflict_suppression.py (already
  verified byte-identical to production there).

SPATIAL REGIONS (per explicit user answers to this turn's questions):
  - L (lesion): the tracked ET component's own voxels.
  - B_near: 12-voxel binary dilation ring around the lesion, excluding
    the lesion itself and excluding any OTHER ET lesion's voxels (a
    wider "local competing anatomy" zone, deliberately larger than
    E233/E234's 3-voxel contrast-measurement shell -- a different
    purpose here).
  - B_far: everything else in the 128^3 patch NOT in L, B_near, or any
    other ET lesion (R).
  - R (rest): every OTHER ET connected component in the same patch (per
    explicit user answer -- tests cross-LESION competition, not just
    generic background).
  Each region gets its own binary mask (full-res AND its own D4-
  downsampled version, via the SAME d4_cell_mask_for_lesion-style cell
  assignment used in e228a, reused not reimplemented, generalized here
  to arbitrary region masks not just a single lesion).

PER-REGION LOSS: for a region's mask M (full-res) and its D4-cell mask
M_D4, the region's own scalar loss is the SAME 3-term formula, but with
every tensor (probs, targets, alpha, beta, boundary_logit/targets,
aux_probs3/t_d4) restricted (masked, non-lesion-region voxels zeroed
contribution) to that region only -- i.e. each region gets its own
independent forward-loss-restricted-to-that-region, then
backward() w.r.t. ALL shared model parameters theta (full model, not
just D1 -- THIS is the key difference from E238, which only needed the
D1 leaf). Four separate backward passes per lesion (L, B_near, B_far,
R), each on a FRESH gradient buffer (model.zero_grad between each).

METRICS, per lesion:
  cos(g_L, g_B) where g_B = g_B_near + g_B_far (pooled background, matching
    the "critical prediction" formula) -- and SEPARATELY cos(g_L,
    g_B_near) vs cos(g_L, g_B_far) for the spatial control.
  g_total = g_L + g_B (pooled, matching the survival-condition formula)
  rho_i = cos(g_total, g_L)
  P_i = (g_total . g_L) / ||g_L||^2  (projection: >0 helps, ~0 cancelled,
    <0 actively harmful)
  ||g_L|| for the "gradient not diminished on its own" check (already
  measured differently in E238 at D1 only; here at full-theta scale).

POPULATION (four groups, per explicit user spec):
  1. persistent_missed / 2. matched_detected: E234's 117 tightly-matched
     pairs (verified equalized on size/t1c, Wilcoxon p=0.40 both).
  2. high_contrast_missed / ordinary_missed: e235's build_groups() G2/
     matched_missed split (t1c-median-based).
Some overlap between (1)/(2) and (3)/(4) is expected and fine -- these
are two different groupings of the same underlying missed-lesion
population, reported as separate rows/analyses, not merged.
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e235_linear_probe_layerwise import build_groups  # noqa: E402

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
ET = 0
PATCH = (128, 128, 128)
K = 4  # D4 downsample factor, matches production/e228a
MU = 0.1  # boundary weight, matches production
LAMBDA_DS3 = 0.9927  # D4 aux weight, matches production
NEAR_DILATION = 12  # voxels, per explicit user choice (wider than E233's 3-vox shell)
MIN_VOX = 5


def focal_tversky(probs, target, alpha=0.5, beta=0.5, gamma=4.0 / 3.0, eps=1e-6):
    """Byte-identical copy of train_e130's own function."""
    dims = tuple(range(2, probs.dim()))
    tp = (probs * target).sum(dims)
    fp = (probs * (1 - target)).sum(dims)
    fn = ((1 - probs) * target).sum(dims)
    ti = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return ((1 - ti) ** gamma).mean()


def evidential_beta(alpha, beta, target, weight=0.5, eps=1e-6):
    """Byte-identical copy of train_e130's own function."""
    s = alpha + beta
    p = alpha / (s + eps)
    nll = (target - p) ** 2 + p * (1 - p) / (s + 1.0)
    return weight * nll.mean()


def load_patch(ds, sid_to_idx, sid):
    image, target, _ = ds._load_subject(ds.subject_dirs[sid_to_idx[sid]])
    D, H, W = image.shape[1:]
    pd_, ph_, pw_ = PATCH
    cd, ch, cw = D // 2, H // 2, W // 2
    sd_, ed_ = max(0, cd - pd_//2), min(D, cd + pd_//2)
    sh_, eh_ = max(0, ch - ph_//2), min(H, ch + ph_//2)
    sw_, ew_ = max(0, cw - pw_//2), min(W, cw + pw_//2)
    img_c = image[:, sd_:ed_, sh_:eh_, sw_:ew_]
    tgt_c = target[:, sd_:ed_, sh_:eh_, sw_:ew_]
    if img_c.shape[1:] != (pd_, ph_, pw_):
        ip = np.zeros((4, pd_, ph_, pw_), dtype=np.float32)
        tp_ = np.zeros((3, pd_, ph_, pw_), dtype=np.float32)
        ip[:, :img_c.shape[1], :img_c.shape[2], :img_c.shape[3]] = img_c
        tp_[:, :tgt_c.shape[1], :tgt_c.shape[2], :tgt_c.shape[3]] = tgt_c
        img_c, tgt_c = ip, tp_
    return img_c, tgt_c


def build_region_masks(tgt_c, comp_id):
    """Returns dict of region -> full-res boolean mask (D,H,W), or None if
    the target lesion is unusable. Regions: L (this lesion), B_near (12-vox
    dilation ring, excluding L and R), B_far (everything else, excluding
    L/B_near/R), R (every OTHER ET component in this patch)."""
    et_lbl, et_n = ndimage.label(tgt_c[ET] > 0.5)
    if comp_id < 1 or comp_id > et_n:
        return None
    L = et_lbl == comp_id
    if L.sum() < MIN_VOX:
        return None
    R = (et_lbl > 0) & (~L)  # all other ET components
    dil = ndimage.binary_dilation(L, iterations=NEAR_DILATION)
    B_near = dil & (~L) & (~R)
    B_far = (~L) & (~B_near) & (~R)
    return {'L': L, 'B_near': B_near, 'B_far': B_far, 'R': R}


def mask_to_d4(mask_np, K):
    """(D,H,W) bool -> (1,1,D/K,H/K,W/K) float mask, cell=1 if ANY voxel
    in that cell belongs to the region (matches d4_cell_mask_for_lesion's
    own 'any'-style occupancy convention from e228a, generalized to
    arbitrary masks)."""
    D, H, W = mask_np.shape
    grid = D // K
    m = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool3d(m, kernel_size=K, stride=K)
    return pooled  # (1,1,grid,grid,grid)


def region_loss(out, tgt_t, mask_t, mask_d4_t, boundary_criterion):
    """Computes the FULL 3-term production loss, restricted to `mask_t`
    (full-res) for the seg+boundary terms and `mask_d4_t` (D4-grid) for
    the aux3 term. Restriction = select only the region's voxels/cells
    before computing each term (not a multiplicative zero-out, which
    would distort focal_tversky's FP/FN accounting) -- same
    'restrict-then-compute' convention as e238_h4's forward_loss."""
    probs, alpha_ev, beta_ev = out['probs'], out['alpha'], out['beta']
    boundary_logit = out['boundary_logit']
    aux_probs3 = out['aux_probs3']

    def restrict(x, m):
        C = x.shape[1]
        xf = x[0].reshape(C, -1)
        mf = m.reshape(-1)
        sel = xf[:, mf]
        if sel.shape[1] == 0:
            return None
        return sel.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    mt = mask_t.reshape(-1)
    if mt.sum() == 0:
        return None
    probs_r = restrict(probs, mask_t)[:, ET:ET+1]
    tgt_r = restrict(tgt_t, mask_t)[:, ET:ET+1]
    alpha_r = restrict(alpha_ev, mask_t)[:, ET:ET+1]
    beta_r = restrict(beta_ev, mask_t)[:, ET:ET+1]
    seg = 0.5 * focal_tversky(probs_r, tgt_r) + 0.5 * evidential_beta(alpha_r, beta_r, tgt_r)

    bnd_logit_r = restrict(boundary_logit, mask_t)[:, ET:ET+1]
    bnd_tgt_r = restrict(tgt_t, mask_t)[:, ET:ET+1]
    bnd = boundary_criterion(bnd_logit_r, bnd_tgt_r)

    md4 = mask_d4_t.reshape(-1) > 0.5
    if md4.sum() == 0:
        aux3 = torch.zeros((), device=probs.device)
    else:
        t_d4 = F.avg_pool3d(tgt_t, kernel_size=K, stride=K)
        aux_r = restrict(aux_probs3, mask_d4_t > 0.5)[:, ET:ET+1]
        t_d4_r = restrict(t_d4, mask_d4_t > 0.5)[:, ET:ET+1]
        aux3 = focal_tversky(aux_r, t_d4_r)

    return seg + MU * bnd + LAMBDA_DS3 * aux3


def flat_grad(model):
    """Concatenates all parameter .grad into one flat CPU tensor (moved
    off GPU immediately -- 5.6M params * 4 bytes = 22MB/vector, keeping
    several on GPU simultaneously across regions would be wasteful)."""
    parts = []
    for p in model.parameters():
        if p.grad is None:
            parts.append(torch.zeros(p.numel()))
        else:
            parts.append(p.grad.detach().reshape(-1).cpu())
    return torch.cat(parts)


def measure_lesion(model, ds, sid_to_idx, sid, comp_id, dev, boundary_criterion):
    if sid not in sid_to_idx:
        return None
    img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
    regions = build_region_masks(tgt_c, comp_id)
    if regions is None:
        return None
    sz = int(regions['L'].sum())

    img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
    tgt_t = torch.from_numpy(tgt_c).unsqueeze(0).float().to(dev)

    region_masks_t = {}
    region_masks_d4_t = {}
    for name, m in regions.items():
        region_masks_t[name] = torch.from_numpy(m).to(dev)
        region_masks_d4_t[name] = mask_to_d4(m, K).to(dev)

    grads = {}
    for name in ['L', 'B_near', 'B_far', 'R']:
        if region_masks_t[name].sum().item() == 0:
            grads[name] = None
            continue
        model.zero_grad(set_to_none=True)
        out = model(img_t)
        loss = region_loss(out, tgt_t, region_masks_t[name], region_masks_d4_t[name], boundary_criterion)
        if loss is None:
            grads[name] = None
            continue
        loss.backward()
        grads[name] = flat_grad(model)

    if grads['L'] is None:
        return None

    g_L = grads['L']
    g_Bn = grads['B_near'] if grads['B_near'] is not None else torch.zeros_like(g_L)
    g_Bf = grads['B_far'] if grads['B_far'] is not None else torch.zeros_like(g_L)
    g_R = grads['R'] if grads['R'] is not None else torch.zeros_like(g_L)
    g_B = g_Bn + g_Bf

    def cos(a, b):
        na, nb = a.norm().item(), b.norm().item()
        if na < 1e-12 or nb < 1e-12:
            return float('nan')
        return float((a @ b).item() / (na * nb))

    g_total = g_L + g_B
    norm_gtotal = g_total.norm().item()
    norm_gL = g_L.norm().item()
    rho = cos(g_total, g_L)
    P = float((g_total @ g_L).item() / (norm_gL ** 2 + 1e-12))

    return {
        'sid': sid, 'comp_id': comp_id, 'size': sz,
        'norm_gL': norm_gL, 'norm_gBnear': g_Bn.norm().item(), 'norm_gBfar': g_Bf.norm().item(),
        'norm_gR': g_R.norm().item(),
        'cos_L_B': cos(g_L, g_B), 'cos_L_Bnear': cos(g_L, g_Bn), 'cos_L_Bfar': cos(g_L, g_Bf),
        'cos_L_R': cos(g_L, g_R),
        'rho': rho, 'P': P,
    }


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev)
    model.load_state_dict(ck['model_state'])
    model.eval()  # eval mode (no dropout/BN-update), but params DO require grad
    for p in model.parameters():
        p.requires_grad_(True)
    boundary_criterion = nn.BCEWithLogitsLoss()

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}

    matches = list(csv.DictReader(open(HERE / 'E234_matches.csv')))
    det, high_contrast_missed, matched_missed, t1c_median = build_groups()

    if smoke:
        matches = matches[:6]
        high_contrast_missed = high_contrast_missed[:6]
        matched_missed = matched_missed[:6]

    out_path = HERE / ('E239_smoke.csv' if smoke else 'E239_gradient_geometry.csv')
    fh = open(out_path, 'w', newline='')
    fieldnames = ['group', 'subject_id', 'comp_id', 'size',
                 'norm_gL', 'norm_gBnear', 'norm_gBfar', 'norm_gR',
                 'cos_L_B', 'cos_L_Bnear', 'cos_L_Bfar', 'cos_L_R', 'rho', 'P']
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    w.writeheader()

    t0 = time.time()
    n_done = 0

    print('Group 1/2: E234 matched persistent_missed / matched_detected pairs '
          f'(n={len(matches)} pairs)', flush=True)
    for m in matches:
        res_miss = measure_lesion(model, ds, sid_to_idx, m['missed_subject'], int(m['missed_comp']), dev, boundary_criterion)
        res_det = measure_lesion(model, ds, sid_to_idx, m['detected_subject'], int(m['detected_comp']), dev, boundary_criterion)
        for grp, res in [('persistent_missed', res_miss), ('matched_detected', res_det)]:
            if res is None:
                continue
            row = {'group': grp, 'subject_id': res['sid'], 'comp_id': res['comp_id'], 'size': res['size']}
            row.update({k: res[k] for k in fieldnames if k not in ('group', 'subject_id', 'comp_id', 'size')})
            w.writerow(row)
        n_done += 1
        if n_done % 5 == 0 or smoke:
            print(f'  {n_done}/{len(matches)} pairs ({time.time()-t0:.0f}s)', flush=True)
    fh.flush()

    print(f'\nGroup 3: high_contrast_missed (n={len(high_contrast_missed)})', flush=True)
    n2 = 0
    for r in high_contrast_missed:
        res = measure_lesion(model, ds, sid_to_idx, r['subject_id'], int(r['comp_id']), dev, boundary_criterion)
        if res is None:
            continue
        row = {'group': 'high_contrast_missed', 'subject_id': res['sid'], 'comp_id': res['comp_id'], 'size': res['size']}
        row.update({k: res[k] for k in fieldnames if k not in ('group', 'subject_id', 'comp_id', 'size')})
        w.writerow(row)
        n2 += 1
        if n2 % 5 == 0 or smoke:
            print(f'  {n2}/{len(high_contrast_missed)} ({time.time()-t0:.0f}s)', flush=True)
    fh.flush()

    print(f'\nGroup 4: ordinary_missed / matched_missed (n={len(matched_missed)})', flush=True)
    n3 = 0
    for r in matched_missed:
        res = measure_lesion(model, ds, sid_to_idx, r['subject_id'], int(r['comp_id']), dev, boundary_criterion)
        if res is None:
            continue
        row = {'group': 'ordinary_missed', 'subject_id': res['sid'], 'comp_id': res['comp_id'], 'size': res['size']}
        row.update({k: res[k] for k in fieldnames if k not in ('group', 'subject_id', 'comp_id', 'size')})
        w.writerow(row)
        n3 += 1
        if n3 % 5 == 0 or smoke:
            print(f'  {n3}/{len(matched_missed)} ({time.time()-t0:.0f}s)', flush=True)
        if smoke and n3 >= 6:
            break

    fh.close()
    print(f'\nE239 complete ({time.time()-t0:.0f}s). wrote {out_path.name}', flush=True)


if __name__ == '__main__':
    main()
