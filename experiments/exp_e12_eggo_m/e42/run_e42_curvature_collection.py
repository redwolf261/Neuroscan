"""
Phase E42: Objective Geometry Audit -- curvature data collection.

NO TRAINING. NO NEW MODEL. NO LOSS CHANGE. Loads already-trained checkpoints
(E39's 6-condition lambda sweep, plus e25's D4-only/D2-only/Both for the D2
comparison in Section E) and estimates Hessian-vector products / Lanczos
spectra via inference-only forward+double-backward passes.

VERIFIED before running (see verify_lanczos_and_hvp.py, run separately,
both checks passed): Lanczos-recovered eigenvalues match direct
eigendecomposition on a synthetic matrix to 1e-2; HVP matches finite-
difference directional derivatives on the real model to 3.4e-5 relative error.

BATCH SIZE: 4, not this project's usual diagnostic batch of 8 -- measured
directly before deciding: batch=8 requires ~9.5GB GPU memory for the
double-backward HVP computation, exceeding this 8.15GB GPU and causing a
~15x slowdown (22.8s/HVP vs 1.3s/HVP at batch=4, 4.94GB, safe margin).
Agreed with the user as a disclosed deviation from E36-E41's batch=8
convention, necessary purely for HVP tractability, not a scientific choice.

CHECKPOINTS: epochs {1, 15, 30} only (early/middle/late representatives),
per an explicit tractability decision made with the user before running --
NOT all 7 checkpoints E39C/E40 used, to keep the full Lanczos protocol
(6 conditions x 3 checkpoints x 2 losses x 40 Lanczos iters x 2 probe
repeats = 2880 HVPs) within a ~1 hour budget.

LOSSES: L_0 (main loss: seg_loss + mu*boundary_loss, UNCHANGED definition
from every prior phase) and L_4 (D4 auxiliary loss ALONE, lambda_ds3 * aux3_loss,
matching E38/E39/E40's own convention of measuring the term WITH its lambda
applied, not a lambda-independent unit loss).

Output: E42_curvature_data.json -- one record per (condition, epoch, loss_term,
probe_repeat) containing the Lanczos alphas/betas (from which the full
eigenvalue spectrum, trace estimate, etc. are derived in the analysis script).
"""
import sys
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from lanczos_hvp import lanczos_tridiagonalize, eigs_from_tridiagonal

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
MU = 0.1
EPOCHS = [1, 15, 30]
N_LANCZOS_ITER = 40
N_PROBE_REPEATS = 2
BATCH_SIZE = 4

E39_CONDITIONS = {
    "A_lambda0": (project_root / "experiments" / "exp_e12_eggo_m" / "e39" / "runs" / "A_lambda0", 0.0),
    "lambda_0.125": (project_root / "experiments" / "exp_e12_eggo_m" / "e39" / "runs" / "lambda_0.125", 0.125),
    "lambda_0.25": (project_root / "experiments" / "exp_e12_eggo_m" / "e39" / "runs" / "lambda_0.25", 0.25),
    "lambda_0.5": (project_root / "experiments" / "exp_e12_eggo_m" / "e39" / "runs" / "lambda_0.5", 0.5),
    "lambda_1.0": (project_root / "experiments" / "exp_e12_eggo_m" / "e39" / "runs" / "lambda_1.0", 1.0),
    "lambda_2.0": (project_root / "experiments" / "exp_e12_eggo_m" / "e39" / "runs" / "lambda_2.0", 2.0),
}

# For Section E (D2 comparison): the e25 core conditions, reused from
# E39C/E40's own established convention (D4only/D2only/Both).
D2_COMPARISON_CONDITIONS = {
    "D4only_e25": (project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs" / "DeepSup_D4only_seed0", 0.9927, 0.0),
    "D2only_e25": (project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs" / "DeepSup_D2only_seed0", 0.0, 1.0014),
    "Both_e25": (project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs" / "DeepSup_seed0", 0.9927, 1.0014),
}


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UNet3D_v3(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def make_hvp_fn(model, params):
    def hvp(loss_fn, v_list):
        loss = loss_fn()
        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
        gv = sum(torch.sum(g * v) for g, v in zip(grads, v_list))
        hvp_list = torch.autograd.grad(gv, params, retain_graph=False, allow_unused=True)
        hvp_list = [h if h is not None else torch.zeros_like(p) for h, p in zip(hvp_list, params)]
        return hvp_list, float(loss.detach().item())
    return hvp


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    imgs, masks = [], []
    for i in range(BATCH_SIZE):
        img, mask, _ = val_dataset[i]
        imgs.append(img)
        masks.append(mask)
    images = torch.stack(imgs).to(device)
    masks_t = torch.stack(masks).to(device)
    print(f"Fixed diagnostic batch: {BATCH_SIZE} subjects (SAME first-{BATCH_SIZE} convention as prior phases)", flush=True)

    focal_fn = FocalTverskyLoss()
    evidential_fn = EvidentialBetaLoss(weight=0.5)
    boundary_criterion = torch.nn.BCEWithLogitsLoss()

    all_records = []
    t_start = time.time()

    def process(cond_name, ckpt_path, lam3, lam2, epoch):
        model, ckpt = load_model(ckpt_path, device)
        params = [p for p in model.parameters() if p.requires_grad]
        hvp_fn = make_hvp_fn(model, params)

        def loss_main():
            out = model(images)
            probs, alpha, beta = out["probs"], out["alpha"], out["beta"]
            boundary_logit = out["boundary_logit"]
            seg = 0.5 * focal_fn(probs, masks_t) + 0.5 * evidential_fn(alpha, beta, masks_t)
            bnd = boundary_criterion(boundary_logit, masks_t)
            return seg + MU * bnd

        def loss_d4():
            out = model(images)
            aux_probs3 = out["aux_probs3"]
            mask_d4 = F.avg_pool3d(masks_t, kernel_size=4, stride=4)
            return lam3 * focal_fn(aux_probs3, mask_d4)

        loss_fns = {"L0": loss_main}
        if lam3 > 0:
            loss_fns["L4"] = loss_d4
        # else: L4 is trivially the zero function at this checkpoint (D4 head
        # inactive) -- its Hessian is exactly zero, not computed via Lanczos
        # (would be a degenerate all-zero spectrum, not informative, and
        # wastes compute); recorded explicitly as a zero-curvature marker.

        for term_name, loss_fn in loss_fns.items():
            for probe in range(N_PROBE_REPEATS):
                seed = probe * 1000 + epoch
                alphas, betas = lanczos_tridiagonalize(hvp_fn, loss_fn, params, N_LANCZOS_ITER, seed, device)
                eigs = eigs_from_tridiagonal(alphas, betas)
                trace_est = float(alphas.sum())  # standard Lanczos trace estimator (sum of alphas approximates tr(H) for enough iterations relative to spectral spread)

                record = {
                    "condition": cond_name, "epoch": epoch, "loss_term": term_name,
                    "lambda_ds3": lam3, "lambda_ds2": lam2, "probe_repeat": probe, "seed": seed,
                    "alphas": alphas.tolist(), "betas": betas.tolist(),
                    "eigenvalues": eigs.tolist(), "trace_estimate": trace_est,
                }
                all_records.append(record)
                elapsed = time.time() - t_start
                print(f"  [{cond_name}] epoch={epoch} term={term_name} probe={probe}: "
                      f"lambda_max={eigs[0]:.4f}  lambda_min={eigs[-1]:.4f}  trace_est={trace_est:.4f}  "
                      f"({elapsed:.0f}s elapsed)", flush=True)

        if lam3 == 0.0:
            record = {
                "condition": cond_name, "epoch": epoch, "loss_term": "L4",
                "lambda_ds3": lam3, "lambda_ds2": lam2, "probe_repeat": 0, "seed": None,
                "alphas": [], "betas": [], "eigenvalues": [0.0], "trace_estimate": 0.0,
                "note": "D4 head inactive (lambda_ds3=0) -- L4 is trivially zero, Hessian trivially zero, not computed via Lanczos.",
            }
            all_records.append(record)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n=== E39 lambda-sweep conditions ===", flush=True)
    for cond_name, (run_dir, lam) in E39_CONDITIONS.items():
        for epoch in EPOCHS:
            ckpt_path = run_dir / "checkpoints" / f"epoch_{epoch}.pth"
            if not ckpt_path.exists():
                print(f"  MISSING: {ckpt_path} -- skipping", flush=True)
                continue
            process(cond_name, ckpt_path, lam, 0.0, epoch)
        with open(OUT_DIR / "E42_curvature_data.json", "w") as f:
            json.dump(all_records, f)
        print(f"  [checkpoint] saved {len(all_records)} records so far", flush=True)

    print("\n=== D2-comparison conditions (e25) ===", flush=True)
    for cond_name, (run_dir, lam3, lam2) in D2_COMPARISON_CONDITIONS.items():
        for epoch in EPOCHS:
            ckpt_path = run_dir / "checkpoints" / f"epoch_{epoch}.pth"
            if not ckpt_path.exists():
                print(f"  MISSING: {ckpt_path} -- skipping", flush=True)
                continue
            process(cond_name, ckpt_path, lam3, lam2, epoch)
        with open(OUT_DIR / "E42_curvature_data.json", "w") as f:
            json.dump(all_records, f)
        print(f"  [checkpoint] saved {len(all_records)} records so far", flush=True)

    with open(OUT_DIR / "E42_curvature_data.json", "w") as f:
        json.dump(all_records, f)
    print(f"\nSaved {len(all_records)} total records to E42_curvature_data.json "
          f"(total time {time.time()-t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
