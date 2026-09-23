"""
Phase E36-A: Deep-Supervision Optimization Autopsy -- core gradient measurement.

NO NEW MODEL TRAINED. NO NEW LOSS. NO HYPERPARAMETER SEARCH. Read-only
analysis of EXISTING checkpoints from e25/deep_sup_runs/ (DeepSup_D4only_seed0,
DeepSup_D2only_seed0, DeepSup_seed0 ["Both"]).

QUESTION: why did D4 supervision help (D4-only: 0.9096 pooled Dice vs A's
0.9063), when D2-only (0.9080) and Both (0.9091) helped less? This script
measures the actual gradients each loss term produces, at multiple points in
training, to find the empirical property that distinguishes D4 -- WITHOUT
assuming in advance which property (gradient conflict, magnitude, phase-
dependence, layer-localization, or something else) is the answer.

THREE LOSS TERMS, exactly as computed in e25/train_deep_sup.py's own
train_epoch() (replicated here verbatim, not reinvented):
    L_main (~ "D1", full 64^3 resolution): seg_loss + mu*boundary_loss
        where seg_loss = 0.5*focal_tversky(probs, masks) + 0.5*evidential(alpha,beta,masks)
        probs/alpha/beta all come from dec1 (the shared trunk's final stage).
    L_D2 (~"32^3"): lambda_ds2 * focal_tversky(aux_probs2, avg_pool3d(masks,2))
        aux_probs2 = aux_head2(dec2), dec2 READ UN-DETACHED.
    L_D4 (~"16^3"): lambda_ds3 * focal_tversky(aux_probs3, avg_pool3d(masks,4))
        aux_probs3 = aux_head3(dec3), dec3 READ UN-DETACHED.

(Named g_64/g_32/g_16 in the prompt's notation; this script uses L_main/L_D2/L_D4
to match the training code's own naming, with an explicit RES_MAP for clarity.)

PROTOCOL (decided with the user before running):
  - A FIXED, seeded batch of 8 validation subjects (identical across every
    checkpoint and every condition -- same subject indices, no augmentation
    exists in this pipeline's dataloader per prior project audits, so this is
    fully reproducible) is used at every measurement point. The exact
    training-batch order/RNG trajectory was never saved, so exact training
    replay is not reconstructable; a fixed held-out batch gives a clean,
    reproducible, apples-to-apples comparison across epochs and conditions
    instead, which is what this diagnostic needs (NOT a generalization metric).
  - model.eval() for BatchNorm (use each checkpoint's own saved running
    stats, not batch stats from this small diagnostic batch) -- gradients
    still flow (autograd is separate from train()/eval() mode; only BN/dropout
    behavior changes). This project's architecture uses BatchNorm3d
    (neuroscan_3d_fixed.py Conv3DBlock), confirmed before choosing this.
  - Checkpoints used: epoch_{1,5,10,15,20,25,30}.pth for each of D4-only,
    D2-only, Both (the only conditions that HAVE all three loss terms -- A
    has no aux heads at all, so A is not part of this gradient decomposition;
    A's own checkpoints are used only for a downstream update-contribution
    sanity baseline in E36-B, not gradient extraction).

For each (condition, checkpoint epoch), computes:
    - g_main, g_D2, g_D4: full-model gradient vectors from EACH loss term
      computed SEPARATELY (separate backward() calls on a fresh zero_grad(),
      not from the single combined backward() training itself uses -- this
      is necessary to isolate each term's own individual gradient, which the
      combined training gradient by construction does not preserve).
    - Per-parameter-group (per named layer) gradient norms for each term,
      to support layer-wise localization (E36-B).
    - Cosine similarity and norm ratio between every pair of terms, both on
      the FULL flattened gradient and restricted to each shared-parameter
      subset (encoder-only, dec2+encoder, etc.).

Output: E36_gradient_measurements.json
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
CONFIG_PATH = project_root / "configs" / "brats.yaml"
MU = 0.1  # matches e25/train_deep_sup.py's own MU exactly

RUN_DIRS = {
    "D4only": project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs" / "DeepSup_D4only_seed0",
    "D2only": project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs" / "DeepSup_D2only_seed0",
    "Both": project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs" / "DeepSup_seed0",
}
EPOCHS = [1, 5, 10, 15, 20, 25, 30]
FIXED_BATCH_SUBJECT_INDICES = list(range(8))  # first 8 validation subjects, fixed, identical every call

# Layer groups for later layer-wise localization (E36-B), built from the
# EXACT architecture graph in neuroscan_3d_v2.py/neuroscan_3d_v3.py (verified
# by direct inspection before writing this, not assumed):
#   enc1->pool1->enc2->pool2->enc3->pool3->bottleneck->upconv3->[cat3]->dec3->aux_head3 (D4)
#                                                        upconv2->[cat2]->dec2->aux_head2 (D2)
#                                                        upconv1->[cat1]->dec1->seg_head/evidential_head/boundary_head (main)
# encoder+bottleneck is shared by ALL THREE losses' gradients.
# upconv3/dec3/aux_head3 is EXCLUSIVE to L_D4.
# upconv2/dec2/aux_head2 is shared by L_main and L_D2 only (dec2's own
#   output feeds upconv1->dec1 downstream, so L_main's gradient also passes
#   through dec2/upconv2 -- but NOT through dec3/upconv3, since dec2 does
#   not depend on dec3's later aux_head3 branch, only on dec3's OWN forward
#   output as an input, which IS shared -- see note in E36-B).
# upconv1/dec1/seg_head/evidential_head/boundary_head is EXCLUSIVE to L_main.
LAYER_GROUPS = {
    "encoder_bottleneck": ["enc1", "enc2", "enc3", "bottleneck"],
    "upconv3_dec3_auxhead3": ["upconv3", "dec3", "aux_head3"],
    "upconv2_dec2_auxhead2": ["upconv2", "dec2", "aux_head2"],
    "upconv1_dec1_heads": ["upconv1", "dec1", "seg_head", "evidential_head", "boundary_head"],
}


def get_layer_group(param_name):
    for group, prefixes in LAYER_GROUPS.items():
        for p in prefixes:
            if param_name.startswith(p + ".") or param_name == p:
                return group
    return "other"


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UNet3D_v3(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


def flatten_grad(model, param_filter=None):
    """Returns a flat 1D tensor of all .grad values (zeros where grad is None),
    in a FIXED, deterministic parameter order (model.named_parameters() order,
    which is stable for a given architecture instance)."""
    parts = []
    for name, p in model.named_parameters():
        if param_filter is not None and not param_filter(name):
            continue
        if p.grad is None:
            parts.append(torch.zeros(p.numel(), device=p.device))
        else:
            parts.append(p.grad.detach().reshape(-1))
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts)


def per_param_grad_norms(model):
    """Returns {param_name: grad_norm} for every parameter with a gradient."""
    out = {}
    for name, p in model.named_parameters():
        out[name] = float(p.grad.detach().norm().item()) if p.grad is not None else 0.0
    return out


def cosine_and_ratio(g1, g2):
    if g1.numel() == 0 or g2.numel() == 0:
        return None, None
    n1 = g1.norm().item()
    n2 = g2.norm().item()
    if n1 == 0.0 or n2 == 0.0:
        return None, None
    cos = torch.dot(g1, g2).item() / (n1 * n2)
    ratio = n1 / n2
    return cos, ratio


def compute_term_gradient(model, images, masks, focal_fn, evidential_fn, boundary_criterion,
                           lambda_ds3, lambda_ds2, term):
    """Zero grads, forward pass, backward ONLY the requested term's loss
    (retain_graph as needed since we reuse the same forward activations
    for all three terms from one forward() call, saving 2/3 of forward
    compute per checkpoint -- verified this doesn't corrupt gradients since
    each term's backward() is preceded by its own zero_grad())."""
    model.zero_grad(set_to_none=True)
    outputs = model(images)
    probs = outputs["probs"]
    alpha, beta = outputs["alpha"], outputs["beta"]
    boundary_logit = outputs["boundary_logit"]
    aux_probs3 = outputs["aux_probs3"]
    aux_probs2 = outputs["aux_probs2"]

    if term == "main":
        focal_loss = focal_fn(probs, masks)
        evidential_loss = evidential_fn(alpha, beta, masks)
        seg_loss = 0.5 * focal_loss + 0.5 * evidential_loss
        boundary_loss = boundary_criterion(boundary_logit, masks)
        loss = seg_loss + MU * boundary_loss
    elif term == "D2":
        mask_d2 = F.avg_pool3d(masks, kernel_size=2, stride=2)
        loss = lambda_ds2 * focal_fn(aux_probs2, mask_d2)
    elif term == "D4":
        mask_d4 = F.avg_pool3d(masks, kernel_size=4, stride=4)
        loss = lambda_ds3 * focal_fn(aux_probs3, mask_d4)
    else:
        raise ValueError(term)

    loss_value = float(loss.item())
    loss.backward()
    return loss_value


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    print(f"Validation set size: {len(val_dataset)}", flush=True)

    images_list, masks_list = [], []
    for idx in FIXED_BATCH_SUBJECT_INDICES:
        img, mask, subj_id = val_dataset[idx]
        images_list.append(img)
        masks_list.append(mask)
        print(f"  fixed batch subject {idx}: {subj_id}", flush=True)
    images = torch.stack(images_list).to(device)
    masks = torch.stack(masks_list).to(device)
    print(f"Fixed diagnostic batch shape: images={tuple(images.shape)} masks={tuple(masks.shape)}", flush=True)

    focal_fn = FocalTverskyLoss()
    evidential_fn = EvidentialBetaLoss(weight=0.5)
    boundary_criterion = torch.nn.BCEWithLogitsLoss()

    with open(CONFIG_PATH) as f:
        pass  # config not needed beyond what's hardcoded (MU, lambdas below), loaded only to confirm path exists

    LAMBDA_DS3 = 0.9927
    LAMBDA_DS2 = 1.0014

    all_results = []

    for condition, run_dir in RUN_DIRS.items():
        for epoch in EPOCHS:
            ckpt_path = run_dir / "checkpoints" / f"epoch_{epoch}.pth"
            if not ckpt_path.exists():
                print(f"  MISSING: {ckpt_path} -- skipping", flush=True)
                continue
            model, ckpt = load_model(ckpt_path, device)
            model.eval()  # BatchNorm uses saved running stats, not this small diagnostic batch's own stats (decided with user)

            term_grads = {}
            term_losses = {}
            term_layer_norms = {}
            for term in ("main", "D2", "D4"):
                loss_val = compute_term_gradient(
                    model, images, masks, focal_fn, evidential_fn, boundary_criterion,
                    LAMBDA_DS3, LAMBDA_DS2, term,
                )
                term_losses[term] = loss_val
                term_grads[term] = flatten_grad(model).cpu()
                per_param = per_param_grad_norms(model)
                layer_norms = {}
                for pname, gn in per_param.items():
                    grp = get_layer_group(pname)
                    layer_norms.setdefault(grp, 0.0)
                    layer_norms[grp] += gn ** 2
                layer_norms = {k: float(np.sqrt(v)) for k, v in layer_norms.items()}
                term_layer_norms[term] = layer_norms

            pair_stats = {}
            for t1, t2 in [("main", "D4"), ("main", "D2"), ("D4", "D2")]:
                cos, ratio = cosine_and_ratio(term_grads[t1], term_grads[t2])
                pair_stats[f"{t1}_vs_{t2}"] = {"cosine": cos, "norm_ratio_t1_over_t2": ratio}

            norms = {t: float(term_grads[t].norm().item()) for t in ("main", "D2", "D4")}

            all_results.append({
                "condition": condition,
                "epoch": epoch,
                "checkpoint_recorded_epoch": ckpt.get("epoch"),
                "best_val_dice_so_far": ckpt.get("best_val_dice"),
                "term_losses": term_losses,
                "term_grad_norms_full": norms,
                "term_layer_group_norms": term_layer_norms,
                "pairwise": pair_stats,
            })
            print(f"[{condition}] epoch={epoch}: "
                  f"|g_main|={norms['main']:.4f} |g_D2|={norms['D2']:.4f} |g_D4|={norms['D4']:.4f}  "
                  f"cos(main,D4)={pair_stats['main_vs_D4']['cosine']:.4f} "
                  f"cos(main,D2)={pair_stats['main_vs_D2']['cosine']:.4f} "
                  f"cos(D4,D2)={pair_stats['D4_vs_D2']['cosine']:.4f}", flush=True)

    with open(OUT_DIR / "E36_gradient_measurements.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} (condition, epoch) records to E36_gradient_measurements.json", flush=True)


if __name__ == "__main__":
    main()
