"""
Phase E40-A: D4 Optimization Subspace Audit -- directional Gram-matrix data
collection.

NO TRAINING. NO NEW MODEL. NO LOSS CHANGE. Loads already-trained checkpoints
(D4-only, D2-only, Both from e25; the 6-condition lambda sweep from e39) and
runs INFERENCE-ONLY forward/backward passes to extract raw residual gradient
DIRECTIONS -- a genuine, disclosed scope expansion beyond "already-saved
trajectories only" (agreed with the user before running), because E36/E38/
E39's own saved files contain ONLY scalar norms and pairwise cosines against
the main-loss gradient, never cross-batch/cross-epoch gradient comparisons
or raw directions -- verified directly by inspecting those files before
concluding this expansion was necessary, not assumed.

WHY A GRAM MATRIX, NOT A FULL COVARIANCE MATRIX: the model has ~5.6M
parameters; a full covariance matrix over gradient directions would be
5.6M x 5.6M (infeasible, ~125TB). Per E40's own explicit instruction ("if
full parameter dimensionality makes explicit covariance impossible, use a
mathematically equivalent low-dimensional representation, but verify
numerical equivalence on a smaller parameter subset first"), this uses the
standard exact identity: for T unit vectors u_1..u_T in a d-dimensional
space (d >> T), the nonzero eigenvalues of the d x d covariance
C = (1/T) sum_t u_t u_t^T are IDENTICAL to the eigenvalues of the T x T Gram
matrix G = (1/T) U^T U, where G_ij = u_i . u_j. This was verified directly
on a synthetic d=500, T=12 test before use here: full-covariance eigenvalues
and Gram-matrix eigenvalues matched to 1e-8, and R_eff computed from each
matched to float64 precision. Only pairwise dot products between residual
DIRECTIONS are needed -- never the full d-dimensional vectors stored
simultaneously (each is computed, dotted against previously-stored ones,
then discarded).

RESIDUAL DEFINITION (exactly as specified):
    r_4 = g_4 - (g_4^T g_0 / (|g_0|^2 + eps)) * g_0
    (equivalently for D2: r_2 using g_2 in place of g_4)
    u_t = r_t / (|r_t| + eps)   -- normalized residual direction

CONDITIONS COVERED:
    - D4only, D2only, Both (from e25/deep_sup_runs) -- the core D4-vs-D2
      comparison, at epochs {1,5,10,15,20,25,30} (matching E39C's own
      early=1,5,10 / middle=15,20 / late=25,30 phase convention).
    - The 6 E39 lambda-sweep conditions (A_lambda0, lambda_0.125/0.25/0.5/
      1.0/2.0), at the SAME epoch checkpoints, for the outcome test
      (Section 5 of the E40 prompt) -- these only ever have a D4 head
      active (lambda_ds2=0 throughout E39A/B, verified against
      e39/train_e39_lambda_sweep.py before assuming this), so only r_4 is
      computed for these conditions, not r_2.

DIAGNOSTIC BATCHES: the SAME 15 disjoint batches of 8 validation subjects
E36/E38 already established (subjects 0-119, batches of 8), reused for
direct consistency with prior phases -- not re-chosen.

Output: E40_directional_gram_data.json -- one record per (condition, epoch,
scale) containing the TxT Gram matrix of residual directions across the 15
batches, plus per-batch residual/main norms for later size/magnitude
confound checks.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
MU = 0.1
EPS = 1e-8
EPOCHS = [1, 5, 10, 15, 20, 25, 30]
N_BATCHES = 15
BATCH_SIZE = 8

# Core D4-vs-D2 conditions (e25) -- have BOTH aux heads active in the model
# architecturally; loss weight per head varies by condition name.
CORE_CONDITIONS = {
    "D4only": (project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs" / "DeepSup_D4only_seed0", 0.9927, 0.0),
    "D2only": (project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs" / "DeepSup_D2only_seed0", 0.0, 1.0014),
    "Both": (project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs" / "DeepSup_seed0", 0.9927, 1.0014),
}

# E39 lambda-sweep conditions -- D4 ONLY (lambda_ds2=0 throughout, verified
# against e39/train_e39_lambda_sweep.py before assuming), used for the
# outcome test (Section 5), not for the core D4-vs-D2 comparison.
E39_CONDITIONS = {
    "A_lambda0": (project_root / "experiments" / "exp_e12_eggo_m" / "e39" / "runs" / "A_lambda0", 0.0),
    "lambda_0.125": (project_root / "experiments" / "exp_e12_eggo_m" / "e39" / "runs" / "lambda_0.125", 0.125),
    "lambda_0.25": (project_root / "experiments" / "exp_e12_eggo_m" / "e39" / "runs" / "lambda_0.25", 0.25),
    "lambda_0.5": (project_root / "experiments" / "exp_e12_eggo_m" / "e39" / "runs" / "lambda_0.5", 0.5),
    "lambda_1.0": (project_root / "experiments" / "exp_e12_eggo_m" / "e39" / "runs" / "lambda_1.0", 1.0),
    "lambda_2.0": (project_root / "experiments" / "exp_e12_eggo_m" / "e39" / "runs" / "lambda_2.0", 2.0),
}


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UNet3D_v3(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def flatten_grad(model):
    parts = []
    for p in model.parameters():
        if p.grad is None:
            parts.append(torch.zeros(p.numel(), device=p.device))
        else:
            parts.append(p.grad.detach().reshape(-1))
    return torch.cat(parts)


def compute_g0_and_gr(model, images, masks, focal_fn, evidential_fn, boundary_criterion, lam3, lam2, term):
    """Returns (g0_flat, gr_flat) as CPU tensors -- one pair of full forward/
    backward passes for g0 (main loss alone), one for the requested term
    (D4 or D2, WITH its lambda applied, matching E38/E39's own convention).
    Model.zero_grad() called before and after to avoid gradient accumulation
    contaminating subsequent measurements."""
    model.zero_grad(set_to_none=True)
    outputs = model(images)
    probs, alpha, beta = outputs["probs"], outputs["alpha"], outputs["beta"]
    boundary_logit = outputs["boundary_logit"]
    focal_loss = focal_fn(probs, masks)
    evidential_loss = evidential_fn(alpha, beta, masks)
    seg_loss = 0.5 * focal_loss + 0.5 * evidential_loss
    boundary_loss = boundary_criterion(boundary_logit, masks)
    loss_main = seg_loss + MU * boundary_loss
    loss_main.backward()
    g0 = flatten_grad(model).cpu()
    model.zero_grad(set_to_none=True)

    if term == "D4":
        lam = lam3
    elif term == "D2":
        lam = lam2
    else:
        raise ValueError(term)

    if lam == 0.0:
        gr = torch.zeros_like(g0)
    else:
        outputs = model(images)
        if term == "D4":
            aux_probs = outputs["aux_probs3"]
            mask_ds = F.avg_pool3d(masks, kernel_size=4, stride=4)
        else:
            aux_probs = outputs["aux_probs2"]
            mask_ds = F.avg_pool3d(masks, kernel_size=2, stride=2)
        loss_r = lam * focal_fn(aux_probs, mask_ds)
        loss_r.backward()
        gr = flatten_grad(model).cpu()
        model.zero_grad(set_to_none=True)

    return g0, gr


def residual_direction(g0, gr):
    """r = gr - (gr.g0 / (|g0|^2+eps)) * g0 ; u = r / (|r|+eps).
    Returns (u, |r|, |g0|) -- u is the CPU tensor (kept only transiently by
    the caller, discarded after dot products are computed)."""
    g0_normsq = float(torch.dot(g0, g0).item())
    proj_coeff = float(torch.dot(gr, g0).item()) / (g0_normsq + EPS)
    r = gr - proj_coeff * g0
    r_norm = float(r.norm().item())
    u = r / (r_norm + EPS)
    return u, r_norm, float(g0.norm().item())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    print(f"Validation set size: {len(val_dataset)}", flush=True)

    batches = []
    for b in range(N_BATCHES):
        idxs = list(range(b * BATCH_SIZE, (b + 1) * BATCH_SIZE))
        imgs, masks = [], []
        for idx in idxs:
            img, mask, _ = val_dataset[idx]
            imgs.append(img)
            masks.append(mask)
        batches.append((torch.stack(imgs).to(device), torch.stack(masks).to(device)))
    print(f"Loaded {len(batches)} diagnostic batches (same convention as E36/E38).", flush=True)

    focal_fn = FocalTverskyLoss()
    evidential_fn = EvidentialBetaLoss(weight=0.5)
    boundary_criterion = torch.nn.BCEWithLogitsLoss()

    all_records = []

    def process_condition_epoch(cond_name, ckpt_path, lam3, lam2, terms, filename_epoch):
        # filename_epoch (1,5,10,...,30) is the CANONICAL epoch label used
        # throughout this analysis and matches E39C's own early/mid/late
        # phase convention exactly. The checkpoint's own internal "epoch"
        # field is 0-indexed (epoch_5.pth's own saved field reads 4, verified
        # directly before trusting either number) -- recorded separately
        # below as a cross-check, never used as the primary epoch label, to
        # avoid a silent off-by-one mismatch against E39C's phase boundaries.
        model, ckpt = load_model(ckpt_path, device)
        for term in terms:
            us, r_norms, g0_norms = [], [], []
            for images, masks in batches:
                g0, gr = compute_g0_and_gr(model, images, masks, focal_fn, evidential_fn,
                                            boundary_criterion, lam3, lam2, term)
                u, r_norm, g0_norm = residual_direction(g0, gr)
                us.append(u)
                r_norms.append(r_norm)
                g0_norms.append(g0_norm)

            # Gram matrix: G_ij = u_i . u_j, computed and stored as a TxT
            # matrix (15x15 here) -- NEVER stores the full TxD matrix of
            # directions simultaneously beyond what's needed for this
            # condition/epoch/term's own 15 vectors (transient, released
            # after this block).
            T = len(us)
            U = torch.stack(us)  # (T, D) -- D~5.6M, T=15, transient
            G = (U @ U.T).numpy() / T  # (T, T) -- matches the prompt's own (1/T) normalization
            del U

            record = {
                "condition": cond_name, "epoch": filename_epoch, "checkpoint_internal_epoch_field": ckpt.get("epoch"),
                "term": term, "lambda_used": (lam3 if term == "D4" else lam2),
                "gram_matrix": G.tolist(), "r_norms": r_norms, "g0_norms": g0_norms,
            }
            all_records.append(record)
            eigs = np.linalg.eigvalsh(G)
            eigs = np.clip(eigs, 0, None)
            tr = eigs.sum()
            tr2 = (eigs ** 2).sum()
            r_eff = (tr ** 2 / tr2) if tr2 > 0 else 0.0
            print(f"  [{cond_name}] epoch={filename_epoch} term={term}: "
                  f"R_eff={r_eff:.3f}  mean|r|={np.mean(r_norms):.4f}  mean|g0|={np.mean(g0_norms):.4f}", flush=True)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n=== Core D4-vs-D2 conditions (e25) ===", flush=True)
    for cond_name, (run_dir, lam3, lam2) in CORE_CONDITIONS.items():
        for epoch in EPOCHS:
            ckpt_path = run_dir / "checkpoints" / f"epoch_{epoch}.pth"
            if not ckpt_path.exists():
                print(f"  MISSING: {ckpt_path} -- skipping", flush=True)
                continue
            terms = ["D4", "D2"]  # both heads architecturally present in all 3 core conditions
            process_condition_epoch(cond_name, ckpt_path, lam3, lam2, terms, epoch)

    print("\n=== E39 lambda-sweep conditions (D4 only) ===", flush=True)
    for cond_name, (run_dir, lam) in E39_CONDITIONS.items():
        for epoch in EPOCHS:
            ckpt_path = run_dir / "checkpoints" / f"epoch_{epoch}.pth"
            if not ckpt_path.exists():
                print(f"  MISSING: {ckpt_path} -- skipping", flush=True)
                continue
            process_condition_epoch(cond_name, ckpt_path, lam, 0.0, ["D4"], epoch)

        with open(OUT_DIR / "E40_directional_gram_data.json", "w") as f:
            json.dump(all_records, f)
        print(f"  [checkpoint] saved {len(all_records)} records so far", flush=True)

    with open(OUT_DIR / "E40_directional_gram_data.json", "w") as f:
        json.dump(all_records, f)
    print(f"\nSaved {len(all_records)} total records to E40_directional_gram_data.json", flush=True)


if __name__ == "__main__":
    main()
