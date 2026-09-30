"""E238-H4 -- objective-suppression signature test, per the user's exact
spec (pasted_content, 2026-09-29). MEASUREMENT ONLY: no loss/architecture
change is made anywhere in this script. Frozen E131 checkpoint throughout.

QUESTION: "Does the training objective actually create a systematic
incentive to ignore these lesions, independent of their raw information
content?"

Two SEPARATE, EXPLICITLY COMPARED measurements per the user's own
disambiguation (asked and answered via AskUserQuestion this turn):

  1. FORWARD LOSS VALUE -- the production loss function (focal Tversky +
     evidential Beta NLL, byte-identical copies of train_e130's own
     functions, reused verbatim from e228a_conflict_suppression.py, NOT
     reimplemented) evaluated per-lesion, restricted to that lesion's own
     voxels, using the model's CURRENT frozen output. Tests whether the
     objective itself already discounts these lesions' cost, independent
     of any gradient-flow question.

  2. BACKWARD GRADIENT MAGNITUDE AT D1 (dec1, pre-seg_head) -- d(lesion-
     restricted loss)/d(dec1), via a fresh backward pass with dec1 as a
     leaf requiring grad (everything upstream of dec1 stays in the normal
     frozen no-grad forward, matching e237's own resumed_forward
     convention; only the small subgraph from dec1 -> seg_head -> loss
     needs autograd here). THIS IS DELIBERATELY NOT A REPEAT OF E227/
     E228-A: those measured the D4 AUX head's gradient only. This
     measures the MAIN seg_head/D1 gradient, which was never directly
     probed. If this also vanishes, it's a new, independent confirmation
     at a different layer -- not the same finding restated.

POPULATION: E234's own tightly-matched missed-vs-detected pairs
(E234_matches.csv, 117 pairs, VERIFIED equalized on size/t1c-contrast,
Wilcoxon p=0.40 both -- built specifically so H1 cannot explain any
remaining gap). This IS (per e235's build_groups reasoning) drawn from
lesions that survived a match against a detected-lesion's raw evidence
profile, i.e. skews toward the "high-contrast missed" population this
whole H4/H6 discriminator is about -- reusing this existing, already-
verified-tight matched population rather than rebuilding G2-specific
matching from scratch.

H4 SIGNATURE (per the user's framework, pre-registered):
  - If forward loss for missed IS comparably large/larger than matched
    detected (objective treats them as costly) BUT D1-gradient magnitude
    is significantly SUPPRESSED relative to matched detected -> genuine,
    NEW suppression signature (gradient dies between D1 and the loss,
    independent of D4) -> H4 gets a measurable signature, worth pursuing.
  - If forward loss is ALSO small for missed (objective already
    "satisfied") -> not suppression, a different and more specific
    story (the objective itself doesn't see these as costly) -> also
    worth reporting distinctly, but does NOT support the "suppression"
    framing.
  - If gradient magnitude is NOT significantly different from matched
    detected -> H4 (this specific mechanism) dies.
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage, stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
ET = 0
PATCH = (128, 128, 128)
MIN_VOX = 5


def focal_tversky(probs, target, alpha=0.5, beta=0.5, gamma=4.0 / 3.0, eps=1e-6):
    """Byte-identical copy of train_e130's own function (verbatim, via
    e228a_conflict_suppression.py's own verbatim copy)."""
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


def component_mask(tgt_c, comp_id):
    et_lbl, et_n = ndimage.label(tgt_c[ET] > 0.5)
    if comp_id < 1 or comp_id > et_n:
        return None
    return et_lbl == comp_id


def measure_lesion(model, ds, sid_to_idx, sid, comp_id, dev):
    """Returns dict with forward_loss (lesion-restricted, no_grad) and
    d1_grad_norm (lesion-restricted, WITH grad through dec1->seg_head),
    or None if the lesion is unusable (out of patch, too small, etc.)."""
    if sid not in sid_to_idx:
        return None
    img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
    cm = component_mask(tgt_c, comp_id)
    if cm is None:
        return None
    sz = int(cm.sum())
    if sz < MIN_VOX:
        return None

    img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
    tgt_t = torch.from_numpy(tgt_c).unsqueeze(0).float().to(dev)
    mask_t = torch.from_numpy(cm).to(dev)

    # ---- forward pass up to dec1, everything frozen/no-grad ----
    with torch.no_grad():
        enc1 = model.enc1(img_t); pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1); pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2); pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)
        upconv3 = model.upconv3(bottleneck)
        cat3 = torch.cat([upconv3, enc3], dim=1)
        dec3 = model.dec3(cat3)
        upconv2 = model.upconv2(dec3)
        cat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = model.dec2(cat2)
        upconv1 = model.upconv1(dec2)
        enc1_gated, _ = model.attn_gate1(gate=bottleneck, skip=enc1)
        cat1 = torch.cat([upconv1, enc1_gated], dim=1)
        dec1_frozen = model.dec1(cat1)

    # ---- forward loss value: lesion-restricted, no grad needed.
    # Uses model(img_t) directly (the real forward call) for probs/alpha/
    # beta, rather than re-deriving probs from dec1_frozen separately --
    # guarantees this is EXACTLY production's own output, not a
    # hand-assembled reconstruction that could silently diverge. ----
    with torch.no_grad():
        full_out = model(img_t)
        probs = full_out['probs']
        alpha_ev, beta_ev = full_out['alpha'], full_out['beta']

        def restrict(x):
            # x: (1, C, D, H, W) -> flatten spatial, select lesion voxels
            C = x.shape[1]
            xf = x[0].reshape(C, -1)
            mf = mask_t.reshape(-1)
            return xf[:, mf].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        probs_l = restrict(probs)[:, ET:ET+1]
        tgt_l = restrict(tgt_t)[:, ET:ET+1]
        alpha_l = restrict(alpha_ev)[:, ET:ET+1]
        beta_l = restrict(beta_ev)[:, ET:ET+1]
        seg_loss_val = (0.5 * focal_tversky(probs_l, tgt_l) +
                        0.5 * evidential_beta(alpha_l, beta_l, tgt_l)).item()
        mean_prob = probs[0, ET][mask_t].mean().item()
        max_prob = probs[0, ET][mask_t].max().item()

    # ---- backward gradient at D1: fresh leaf tensor, small subgraph only ----
    dec1_leaf = dec1_frozen.detach().clone().requires_grad_(True)
    probs_g = model.seg_head(dec1_leaf)
    # evidential head: verified against neuroscan_3d_v5.py lines 160-163 --
    # single conv producing 2*out_channels raw logits, chunked into
    # alpha_raw/beta_raw, then softplus(.)+1.0 -- NOT a separate module
    # returning (alpha, beta) directly. Reproduced exactly here (not
    # reimplemented differently) so the gradient measures the SAME
    # quantity that actually appears in the production loss.
    evidential_raw_g = model.evidential_head(dec1_leaf)
    alpha_raw_g, beta_raw_g = torch.chunk(evidential_raw_g, 2, dim=1)
    alpha_g = F.softplus(alpha_raw_g) + 1.0
    beta_g = F.softplus(beta_raw_g) + 1.0

    def restrict_g(x):
        C = x.shape[1]
        xf = x[0].reshape(C, -1)
        mf = mask_t.reshape(-1)
        return xf[:, mf].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    probs_lg = restrict_g(probs_g)[:, ET:ET+1]
    tgt_lg = restrict_g(tgt_t)[:, ET:ET+1]
    if alpha_g is not None:
        alpha_lg = restrict_g(alpha_g)[:, ET:ET+1]
        beta_lg = restrict_g(beta_g)[:, ET:ET+1]
        loss_g = 0.5 * focal_tversky(probs_lg, tgt_lg) + 0.5 * evidential_beta(alpha_lg, beta_lg, tgt_lg)
    else:
        loss_g = focal_tversky(probs_lg, tgt_lg)
    model.zero_grad(set_to_none=True)
    loss_g.backward()
    d1_grad = dec1_leaf.grad
    grad_norm_lesion = d1_grad[0, :, mask_t].norm().item() if d1_grad is not None else float('nan')
    grad_norm_all = d1_grad[0].norm().item() if d1_grad is not None else float('nan')

    return {
        'sid': sid, 'comp_id': comp_id, 'size': sz,
        'forward_loss': seg_loss_val, 'mean_prob': mean_prob, 'max_prob': max_prob,
        'd1_grad_norm_lesion': grad_norm_lesion, 'd1_grad_norm_all': grad_norm_all,
    }


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev)
    model.load_state_dict(ck['model_state'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)  # frozen weights; only dec1 LEAF activation needs grad

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}

    matches = list(csv.DictReader(open(HERE / 'E234_matches.csv')))
    if smoke:
        matches = matches[:10]
    print(f'E238-H4: {len(matches)} matched missed/detected pairs', flush=True)

    out = HERE / ('E238_h4_smoke.csv' if smoke else 'E238_h4_objective.csv')
    fh = open(out, 'w', newline='')
    w = csv.DictWriter(fh, fieldnames=[
        'pair_idx', 'role', 'subject_id', 'comp_id', 'size',
        'forward_loss', 'mean_prob', 'max_prob', 'd1_grad_norm_lesion', 'd1_grad_norm_all'])
    w.writeheader()

    t0 = time.time()
    n_ok = 0
    for i, m in enumerate(matches):
        res_missed = measure_lesion(model, ds, sid_to_idx, m['missed_subject'], int(m['missed_comp']), dev)
        res_det = measure_lesion(model, ds, sid_to_idx, m['detected_subject'], int(m['detected_comp']), dev)
        if res_missed is None or res_det is None:
            continue
        for role, res in [('missed', res_missed), ('detected', res_det)]:
            w.writerow({'pair_idx': i, 'role': role, 'subject_id': res['sid'],
                       'comp_id': res['comp_id'], 'size': res['size'],
                       'forward_loss': res['forward_loss'], 'mean_prob': res['mean_prob'],
                       'max_prob': res['max_prob'],
                       'd1_grad_norm_lesion': res['d1_grad_norm_lesion'],
                       'd1_grad_norm_all': res['d1_grad_norm_all']})
        n_ok += 1
        if n_ok % 10 == 0 or smoke:
            print(f'  {n_ok}/{len(matches)} pairs ({time.time()-t0:.0f}s)', flush=True)
    fh.close()
    print(f'\nE238-H4 complete: {n_ok} valid pairs ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
