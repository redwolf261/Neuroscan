"""E172 step 2 -- HARNESS VALIDATION ONLY. Reproduce E15's oracle at dec1.

Not a candidate arm. This asks one question: does this harness detect the
intervention E15 measured? If not, any negative result from it is a harness
failure, not evidence.

E15 spec, verbatim from PHASE_E15_EVIDENCE_FREEZE.md:
  representation : z = dec1 (32ch, immediately before seg_head)   <-- NOT enc1
  intervention   : z' = z + alpha * (z - mu_opp) / ||z - mu_opp||  <-- PER-VOXEL
  alpha          : {0,1,2,4,8,14,20,28}  ABSOLUTE units, not fractions of ||z||
  expected       : 0.9050 -> 0.9096 -> 0.9138 -> 0.9209 -> 0.9317 -> 0.9443

My first implementation got all three wrong (enc1, global broadcast, fractional
alpha). Those results are void.
"""
import sys, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

root = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(root))
from neuroscan_3d_v5 import UNet3D_v5
from Dataset.brats_multimodal_dataset import create_multimodal_loaders

CKPT = root/"experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth"
ALPHAS = [0, 1, 2, 4, 8, 14, 20, 28]
N_SUBJ = 20          # E15 used 20 fixed validation subjects


def dec1_and_head(m, image):
    """Run to dec1, return (dec1, closure applying frozen seg_head)."""
    cache = {}
    h = m.dec1.register_forward_hook(lambda mo, i, o: cache.__setitem__("z", o))
    with torch.no_grad():
        m(image)
    h.remove()
    return cache["z"].detach()


def dice3(pred, gt):
    out = []
    for r in range(pred.shape[0]):
        p, t = pred[r], gt[r]
        ps, ts = p.sum(), t.sum()
        out.append(1.0 if ts == 0 and ps == 0 else
                   (0.0 if ts == 0 else float(2*(p*t).sum()/(ps+ts))))
    return out


def main():
    dev = torch.device("cuda")
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    assert abs(ck["best_mean_dice"] - 0.8929357248544694) < 1e-9
    print("[Sanity] checkpoint identity PASS")
    m = UNet3D_v5(4, 3).to(dev); m.load_state_dict(ck["model_state"]); m.eval()
    for p in m.parameters(): p.requires_grad_(False)

    _, val = create_multimodal_loaders(root_dir=str(root/"Dataset"/"Training"),
        batch_size=1, num_workers=0, val_split=0.1, patch_size=(128,128,128), seed=0)

    curves = []
    for i in range(N_SUBJ):
        image, target, sid = val.dataset[i]
        D, H, W = image.shape[1:]
        pad = [0,0,0,0,0,0]
        for ax, sz in enumerate((D,H,W)):
            if sz < 128: pad[(2-ax)*2+1] = 128-sz
        if any(pad):
            image = F.pad(image.unsqueeze(0), pad).squeeze(0)
            target = F.pad(target.unsqueeze(0), pad).squeeze(0)
        D,H,W = image.shape[1:]
        z0,y0,x0 = (D-128)//2, (H-128)//2, (W-128)//2
        img = image[:, z0:z0+128, y0:y0+128, x0:x0+128].unsqueeze(0).to(dev)
        tgt = target[:, z0:z0+128, y0:y0+128, x0:x0+128].unsqueeze(0).to(dev)

        z = dec1_and_head(m, img)                      # (1,32,128,128,128)
        C = z.shape[1]
        # E15: mu_opp is the same-volume centroid of the OPPOSITE GT class,
        # per voxel. Binary fg/bg from the union of the 3 GT regions.
        fg = (tgt.sum(1, keepdim=True) > 0).float()    # (1,1,D,H,W)
        zf = z.reshape(1, C, -1); w = fg.reshape(1, 1, -1)
        nf = w.sum().clamp(min=1.0); nb = (1-w).sum().clamp(min=1.0)
        mu_f = (zf*w).sum(2)/nf                        # (1,C) fg centroid
        mu_b = (zf*(1-w)).sum(2)/nb                    # (1,C) bg centroid
        # opposite centroid PER VOXEL: fg voxels get mu_b, bg voxels get mu_f
        mu_opp = (w*mu_b.unsqueeze(2) + (1-w)*mu_f.unsqueeze(2)).reshape_as(z)
        diff = z - mu_opp
        d = diff / diff.norm(dim=1, keepdim=True).clamp(min=1e-8)   # PER-VOXEL unit

        row = {"sid": sid, "dice": {}}
        gt_np = tgt.squeeze(0).cpu().numpy()
        for al in ALPHAS:
            with torch.no_grad():
                out = m.seg_head(z + al*d).float()
                row["dice"][str(al)] = float(np.mean(dice3((out>0.5).float().squeeze(0).cpu().numpy(), gt_np)))
        curves.append(row)
        print(f"[{i+1}/{N_SUBJ}] {sid} " +
              " ".join(f"a{a}={row['dice'][str(a)]:.4f}" for a in [0,4,14,28]), flush=True)

    means = {str(a): float(np.mean([c["dice"][str(a)] for c in curves])) for a in ALPHAS}
    print("\n=== E15 ORACLE REPRODUCTION (dec1, per-voxel, absolute alpha) ===")
    for a in ALPHAS:
        print(f"  alpha={a:3d}  mean Dice={means[str(a)]:.4f}")
    mono = all(means[str(ALPHAS[k+1])] >= means[str(ALPHAS[k])] - 1e-4 for k in range(4))
    gain = means[str(14)] - means[str(0)]
    print(f"\nmonotone over 0..8: {mono}   gain at alpha=14: {gain:+.4f}")
    print("E15 reference gain at alpha=14: +0.0393")
    print("HARNESS VALID" if gain > 0.01 else "HARNESS STILL NOT REPRODUCING E15")
    json.dump({"means": means, "per_subject": curves, "gain_at_14": gain,
               "harness_valid": bool(gain > 0.01)},
              open(Path(__file__).parent/"E172_oracle_smoke.json","w"), indent=1)


if __name__ == "__main__":
    main()
