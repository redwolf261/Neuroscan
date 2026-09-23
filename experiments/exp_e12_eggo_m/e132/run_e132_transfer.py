"""
E132 -- does the causal loss-localisation METHODOLOGY transfer to a
structurally different architecture?

THE CLAIM UNDER TEST. Across E48-E131 this project developed a procedure:
  (1) ablate each downsampling site in turn, measure the Dice drop -> N_k,
      the causal necessity of that site;
  (2) measure, label-blind, the participation ratio (effective rank) of the
      2x2x2 windows entering that site;
  (3) test whether PR predicts N_k across subjects.
On UNet3D_v5 this found a strong coupling at the pool3 site (rho=0.83
exploratory, 0.877 held-out partial, causally confirmed by a dose-response
in E126, surviving 3x capacity in E129).

If that coupling exists only at one hard-coded layer of one network, it is
an artifact of v5. The methodology claim requires it to reappear when the
same procedure is applied to a network built differently.

PRE-REGISTERED PREDICTIONS, recorded BEFORE the diagnostic is run (v15 was
still training when this file was written):

  P1. The ablation will identify a non-uniform necessity profile across
      v15's FOUR downsampling sites -- i.e. the sites are not
      interchangeable. (Weak prediction; mainly a sanity check that the
      procedure produces structure rather than noise.)

  P2. THE REAL TEST. At whichever site the ablation identifies as most
      causally necessary, PR will predict N_k across subjects with
      rho > 0.3 and permutation p < 0.05, surviving a lesion-volume
      control.
      NOTE what this does and does not say. It does NOT predict the
      coupling appears at "the third site" or at any fixed depth. v15 has
      four sites and a deeper bottleneck; if the coupling is a property of
      segmentation networks rather than of v5's layout, it should track
      the causally-identified site wherever that lands.

  P3. HARDEST VERSION. v15 downsamples with LEARNED STRIDED CONVOLUTIONS,
      not MaxPool. Every prior result in this project concerns what a
      fixed winner-take-all operation discards. A strided conv is a
      learnable weighted sum and can in principle preserve any linear
      combination of the 8 values. If the coupling STILL appears, the
      finding is no longer "MaxPool throws information away" but the
      broader "downsampling transitions destroy task-relevant information
      in a way measurable in situ that predicts downstream necessity".

  FALSIFICATION. If P2 fails -- no site shows a significant PR<->N_k
      coupling surviving the volume control -- the methodology does NOT
      transfer, the v5 finding is architecture-specific, and that is the
      result to report. This is a real possibility and it is the reason
      the predictions are written down first.

METHOD. Deliberately identical in construction to E121/E129 so the
comparison is like-for-like:
  - ablation: zero that stage's OUTPUT tensor, let everything downstream
    recompute for real (E121's CORRECTED construction -- E121 showed that
    zeroing the wrong tensor produces an out-of-distribution intervention
    that inflates the effect and flips its size correlation);
  - PR: closed form tr(G)^2/||G||_F^2 over each 2x2x2 window of the tensor
    ENTERING the site (verified against eigvalsh to 2.4e-06; also avoids a
    cusolver batch-limit crash);
  - controls: Spearman with 1000-permutation p, plus partial correlation
    controlling log lesion volume;
  - metric: mean Dice over the three regions, matching the E130/E131 task.
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy import stats

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v15 import UNet3D_v15  # noqa: E402
from neuroscan_3d_v13 import participation_ratio_windows  # noqa: E402
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset, REGIONS  # noqa: E402

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "e130_train", str(project_root / "experiments" / "exp_e12_eggo_m" / "e130"
                      / "train_e130_multimodal_baseline.py"))
_t = importlib.util.module_from_spec(_spec)
_argv = sys.argv
sys.argv = ["x"]
_spec.loader.exec_module(_t)
sys.argv = _argv

OUT_DIR = Path(__file__).parent
N_PERM = 1000
SEED = 0

# v15's four downsampling sites. Each entry: (name, the module whose OUTPUT
# gets zeroed, the tensor that ENTERS it and whose windows PR is measured on)
SITES = ["down1", "down2", "down3", "down4"]


def forward_v15(model, x, ablate=None, want_pr_at=None):
    """One forward pass. If ablate is a site name, that site's OUTPUT is
    zeroed and everything downstream recomputes from the zeros. If
    want_pr_at is a site name, also returns the tensor entering it."""
    feats = {}
    e1 = model.enc1(x)
    if want_pr_at == "down1":
        feats["pre"] = e1
    p1 = model.down1(e1)
    if ablate == "down1":
        p1 = torch.zeros_like(p1)

    e2 = model.enc2(p1)
    if want_pr_at == "down2":
        feats["pre"] = e2
    p2 = model.down2(e2)
    if ablate == "down2":
        p2 = torch.zeros_like(p2)

    e3 = model.enc3(p2)
    if want_pr_at == "down3":
        feats["pre"] = e3
    p3 = model.down3(e3)
    if ablate == "down3":
        p3 = torch.zeros_like(p3)

    e4 = model.enc4(p3)
    if want_pr_at == "down4":
        feats["pre"] = e4
    p4 = model.down4(e4)
    if ablate == "down4":
        p4 = torch.zeros_like(p4)

    b = model.bottleneck(p4)
    d4 = model.dec4(torch.cat([model.up4(b), e4], dim=1))
    d3 = model.dec3(torch.cat([model.up3(d4), e3], dim=1))
    d2 = model.dec2(torch.cat([model.up2(d3), e2], dim=1))
    d1 = model.dec1(torch.cat([model.up1(d2), e1], dim=1))
    return model.seg_head(d1), feats


def mean_dice(pred_bin, target_bin):
    return float(np.mean(_t.dice_per_region(pred_bin, target_bin)))


def perm_corr(x, y, seed=SEED):
    rho, _ = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    null = np.array([stats.spearmanr(x, rng.permutation(y))[0] for _ in range(N_PERM)])
    return float(rho), float((np.abs(null) >= abs(rho)).mean())


def partial_corr(x, y, ctrl, seed=SEED):
    X = np.column_stack([np.ones(len(ctrl)), ctrl])
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    rho, _ = stats.spearmanr(rx, ry)
    rng = np.random.default_rng(seed)
    null = np.array([stats.spearmanr(rx, rng.permutation(ry))[0] for _ in range(N_PERM)])
    return float(rho), float((np.abs(null) >= abs(rho)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(
        project_root / "experiments" / "exp_e12_eggo_m" / "e131" / "runs"
        / "E132_v15_seed0" / "checkpoints" / "best.pth"))
    ap.add_argument("--n_subjects", type=int, default=60,
                    help="patch-based diagnostic; 60 gives adequate power for rho>0.3")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    assert ck.get("arch") == "v15", f"expected v15 checkpoint, got {ck.get('arch')}"
    print(f"checkpoint epoch={ck['epoch']+1} best_mean_dice={ck['best_mean_dice']:.4f}")

    model = UNet3D_v15(4, 3).to(dev)
    model.load_state_dict(ck["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ds = BraTSMultimodalDataset(str(project_root / "Dataset" / "Training"), "val",
                                val_split=0.1, patch_size=(128, 128, 128))
    n = min(a.n_subjects, len(ds))
    print(f"diagnostic on {n} validation subjects, 4 ablation sites\n", flush=True)

    recs = []
    for i in range(n):
        image, target, sid = ds[i]
        # centre 128^3 patch: the diagnostic needs a fixed-size input, and a
        # centred crop is deterministic (no sampling noise between sites).
        c = [s // 2 for s in image.shape[1:]]
        sl = tuple(slice(max(0, c[j] - 64), max(0, c[j] - 64) + 128) for j in range(3))
        x = image[(slice(None),) + sl]
        y = target[(slice(None),) + sl]
        if tuple(x.shape[1:]) != (128, 128, 128):
            continue
        xb = x.unsqueeze(0).to(dev)
        yb = y.numpy()

        with torch.no_grad():
            p_intact, _ = forward_v15(model, xb)
            d_intact = mean_dice((p_intact[0].cpu().numpy() >= 0.5).astype(np.float32), yb)
            r = {"subject_id": sid, "dice_intact": d_intact,
                 "lesion_vox": int(yb[2].sum())}
            for site in SITES:
                p_ab, _ = forward_v15(model, xb, ablate=site)
                r[f"N_{site}"] = d_intact - mean_dice(
                    (p_ab[0].cpu().numpy() >= 0.5).astype(np.float32), yb)
                _, f = forward_v15(model, xb, want_pr_at=site)
                r[f"PR_{site}"] = float(participation_ratio_windows(f["pre"]).mean())
        recs.append(r)
        if (i + 1) % 15 == 0:
            print(f"  {i+1}/{n}", flush=True)

    with open(OUT_DIR / "E132_per_subject.json", "w") as f:
        json.dump(recs, f, indent=2)

    vol = np.log(np.array([r["lesion_vox"] for r in recs]) + 1)
    print(f"\n{'='*72}")
    print(f"E132 TRANSFER TEST -- v15 (5 levels, LEARNED strided-conv downsampling)")
    print(f"n={len(recs)} subjects\n")
    print(f"{'site':>7} {'N_k mean':>10} {'PR mean':>9} {'rho(PR,N)':>11} {'perm p':>9} "
          f"{'partial|vol':>12} {'perm p':>9}")

    out = {"n": len(recs), "checkpoint": a.ckpt, "sites": {}}
    for site in SITES:
        N = np.array([r[f"N_{site}"] for r in recs])
        PR = np.array([r[f"PR_{site}"] for r in recs])
        rho, p = perm_corr(PR, N)
        prho, pp = partial_corr(PR, N, vol)
        print(f"{site:>7} {N.mean():10.4f} {PR.mean():9.4f} {rho:+11.4f} {p:9.4f} "
              f"{prho:+12.4f} {pp:9.4f}")
        out["sites"][site] = {"N_mean": float(N.mean()), "N_std": float(N.std()),
                              "PR_mean": float(PR.mean()),
                              "rho": rho, "perm_p": p,
                              "partial_rho_given_volume": prho, "partial_perm_p": pp}

    # P1: is the necessity profile non-uniform?
    means = [out["sites"][s]["N_mean"] for s in SITES]
    most = SITES[int(np.argmax(means))]
    spread = max(means) - min(means)
    print(f"\nP1 non-uniform necessity: spread={spread:.4f}, most necessary site = {most}")

    # P2: coupling at the causally-identified site
    s = out["sites"][most]
    p2 = (abs(s["rho"]) > 0.3 and s["perm_p"] < 0.05
          and abs(s["partial_rho_given_volume"]) > 0.3 and s["partial_perm_p"] < 0.05)
    print(f"P2 coupling at {most}: rho={s['rho']:+.4f} (p={s['perm_p']:.4f}), "
          f"partial={s['partial_rho_given_volume']:+.4f} (p={s['partial_perm_p']:.4f}) "
          f"-> {'PASS' if p2 else 'FAIL'}")

    any_site = any(abs(out["sites"][k]["rho"]) > 0.3 and out["sites"][k]["perm_p"] < 0.05
                   and abs(out["sites"][k]["partial_rho_given_volume"]) > 0.3
                   and out["sites"][k]["partial_perm_p"] < 0.05 for k in SITES)
    print(f"   (any site passing the same bar: {any_site})")

    if p2:
        verdict = "TRANSFERS"
        detail = ("The coupling reappears at the causally-identified site in a network "
                  "with an extra level and LEARNED strided-conv downsampling. The finding "
                  "is not specific to MaxPool or to v5's layout.")
    elif any_site:
        verdict = "PARTIAL"
        detail = ("A site shows the coupling, but not the one the ablation identifies as "
                  "most necessary. Report exactly that -- do not relabel the prediction "
                  "after the fact.")
    else:
        verdict = "DOES_NOT_TRANSFER"
        detail = ("No site shows a volume-controlled coupling. The v5 finding is "
                  "architecture-specific. This is the pre-registered falsification and "
                  "is the result to report.")
    print(f"\nVERDICT: {verdict}\n{detail}")

    out.update({"most_necessary_site": most, "necessity_spread": float(spread),
                "P2_pass": bool(p2), "any_site_pass": bool(any_site),
                "verdict": verdict, "detail": detail})
    with open(OUT_DIR / "E132_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved E132_summary.json + per-subject table")


if __name__ == "__main__":
    main()
