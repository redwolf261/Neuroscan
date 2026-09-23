"""
Phase E43: Representation Change Audit -- raw data collection.

NO TRAINING. NO NEW MODEL. NO NEW LOSS. NO INFERENCE-BEHAVIOR CHANGE. Loads
four already-trained checkpoints (baseline, D4-only, D2-only, Both) and runs
ordinary forward passes (no gradient, no HVP, no double-backward -- much
cheaper than E42) to extract decoder feature maps F1/F2/F3 (dec1/dec2/dec3)
at every one of the 125 validation subjects, paired.

WHY A FORWARD HOOK FOR THE BASELINE MODEL: UNet3D_v2 (the baseline
architecture) only returns 'dec1' in its output dict -- dec2/dec3 are
computed internally (verified directly against neuroscan_3d_fixed.py's own
UNet3D.forward(), the shared parent class both UNet3D_v2 and UNet3D_v3
inherit from) but never exposed. Per this project's own strict "never edit
a frozen file" convention (v1/v2 are explicitly frozen), dec2/dec3 are
captured via a forward hook on model.dec2/model.dec3 instead of modifying
the frozen source -- a read-only, zero-behavior-change way to access an
already-computed intermediate tensor.

REGION MASKS: per-resolution occupancy masks (agreed with the user before
running), built the SAME verified way as E36/E41 -- avg_pool3d(native GT
mask resized to 64^3, kernel=factor, stride=factor) computed FRESH at each
decoder stage's own resolution (16^3 for dec3, 32^3 for dec2, 64^3 for
dec1), NOT a single mask borrowed from one scale and reused at the others.
    interior   = cells with occupancy p_j >= 1-eps (entirely tumor)
    boundary   = cells with 0 < p_j < 1-eps (a lesion boundary crosses this cell)
    background = cells with p_j == 0 (entirely non-tumor)

THREE RAW QUANTITIES COMPUTED (nothing more -- no derived/fancy scalar
until these are inspected, per the explicit instruction for this phase):
    1. Magnitude: M_l = mean(||F_l(x)||_2) over voxels in each region, for
       EACH model (baseline, D4-only, D2-only, Both), each decoder stage l.
    2. Representation change: Delta_F_l = F_l^cond - F_l^baseline (paired,
       same scan, same voxel coordinates), reported as ||Delta_F_l||_2 per
       voxel, aggregated (mean) within each region.
    3. Representation direction: cos(F_l^cond, F_l^baseline) per voxel
       (channel-wise cosine similarity at each spatial location), aggregated
       (mean) within each region.

Reported for D4-only AND D2-only (both vs baseline), matching the prompt's
own "decisive comparison" (Delta_D4 vs Delta_D2), plus Both vs baseline for
completeness.

Output: E43_representation_change_table.json -- one row per (subject, decoder
stage, condition, region) with mean magnitude/delta-norm/cosine.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import scipy.ndimage as ndi
from scipy.ndimage import zoom

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v2 import UNet3D_v2  # noqa: E402
from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
EPS = 1e-9

CHECKPOINTS = {
    "baseline": (project_root / "experiments" / "exp_e12_eggo_m" / "e24" / "gate6_runs" / "A_baseline_seed0" / "checkpoints" / "epoch_30.pth", UNet3D_v2),
    "D4only": (project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs" / "DeepSup_D4only_seed0" / "checkpoints" / "epoch_30.pth", UNet3D_v3),
    "D2only": (project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs" / "DeepSup_D2only_seed0" / "checkpoints" / "epoch_30.pth", UNet3D_v3),
    "Both": (project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs" / "DeepSup_seed0" / "checkpoints" / "epoch_30.pth", UNet3D_v3),
}

DECODER_STAGES = {"dec3": 16, "dec2": 32, "dec1": 64}  # name -> resolution


def resize_nn(volume, target_shape):
    zf = tuple(t / c for t, c in zip(target_shape, volume.shape))
    return zoom(volume, zf, order=0)


def load_model_with_hook(ckpt_path, model_cls, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = model_cls(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    captured = {}

    def make_hook(name):
        def hook(module, inp, out):
            captured[name] = out.detach()
        return hook

    # dec2/dec3 exist as submodules on EVERY model here regardless of class
    # (UNet3D_v2 inherits them from UNet3D; UNet3D_v3 inherits from UNet3D_v2)
    # -- hooked uniformly rather than branching on model_cls, since the
    # underlying computation (and therefore hook validity) is identical.
    model.dec2.register_forward_hook(make_hook("dec2"))
    model.dec3.register_forward_hook(make_hook("dec3"))

    return model, ckpt, captured


def build_region_masks(comp_label_native_shape_mask_64, native_shape):
    """Returns {resolution: {'interior':bool_array, 'boundary':bool_array,
    'background':bool_array}} for the WHOLE-IMAGE (not per-component) binary
    tumor mask at 64^3, matching E36/E41's own verified avg_pool3d convention,
    computed fresh at each of the three decoder resolutions."""
    mask_64 = comp_label_native_shape_mask_64  # (64,64,64) binary float32
    out = {}
    for stage_name, res in DECODER_STAGES.items():
        if res == 64:
            p = mask_64
        else:
            factor = 64 // res
            t = torch.from_numpy(mask_64).unsqueeze(0).unsqueeze(0)
            p = F.avg_pool3d(t, kernel_size=factor, stride=factor).squeeze().numpy()
        interior = p >= 1.0 - 1e-6
        background = p <= 1e-6
        boundary = (~interior) & (~background)
        out[stage_name] = {"interior": interior, "boundary": boundary, "background": background}
    return out


def per_voxel_stats(feat_baseline, feat_cond, region_mask):
    """feat_*: (C, res, res, res) numpy arrays. region_mask: (res,res,res) bool.
    Returns mean magnitude (both), mean ||delta|| , mean cosine, restricted
    to voxels where region_mask is True. Returns None for all if the region
    is empty in this subject (avoids a divide-by-zero / meaningless mean)."""
    if not region_mask.any():
        return None

    fb = feat_baseline[:, region_mask]  # (C, N)
    fc = feat_cond[:, region_mask]      # (C, N)

    mag_b = np.linalg.norm(fb, axis=0)   # (N,)
    mag_c = np.linalg.norm(fc, axis=0)

    delta = fc - fb
    delta_norm = np.linalg.norm(delta, axis=0)

    dot = np.sum(fb * fc, axis=0)
    cos = dot / (mag_b * mag_c + EPS)

    return {
        "n_voxels": int(region_mask.sum()),
        "mean_mag_baseline": float(mag_b.mean()),
        "mean_mag_cond": float(mag_c.mean()),
        "mean_delta_norm": float(delta_norm.mean()),
        "mean_cosine": float(cos.mean()),
        "median_cosine": float(np.median(cos)),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    print(f"Validation set size: {len(val_dataset)}", flush=True)

    models = {}
    captures = {}
    for cond_name, (ckpt_path, cls) in CHECKPOINTS.items():
        model, ckpt, captured = load_model_with_hook(ckpt_path, cls, device)
        models[cond_name] = model
        captures[cond_name] = captured
        print(f"  Loaded {cond_name}: {ckpt_path.name}  best_val_dice={ckpt.get('best_val_dice')}", flush=True)

    records = []
    for subject_idx in range(len(val_dataset)):
        image, mask, subject_id = val_dataset[subject_idx]
        image_b = image.unsqueeze(0).to(device)

        subject_dir = val_dataset.subject_dirs[subject_idx]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        native_labeled, _ = ndi.label(seg_binary_native > 0.5)
        mask_64 = resize_nn(seg_binary_native, (64, 64, 64))
        mask_64 = (mask_64 > 0.5).astype(np.float32)  # nearest-neighbor binary mask at 64^3, matching size_64 convention

        region_masks = build_region_masks(mask_64, seg_binary_native.shape)

        # dec1 IS returned directly by both v2/v3 in their own output dict --
        # captured from the SAME forward call that also fires the dec2/dec3
        # hooks (registered once, at model-load time), so a single
        # model(image_b) call per condition yields all three stages
        # consistently from that one pass.
        feats = {}
        with torch.no_grad():
            for cond_name, model in models.items():
                out = model(image_b)
                dec1 = out["dec1"].squeeze(0).cpu().numpy()
                dec2 = captures[cond_name]["dec2"].squeeze(0).cpu().numpy()
                dec3 = captures[cond_name]["dec3"].squeeze(0).cpu().numpy()
                feats[cond_name] = {"dec1": dec1, "dec2": dec2, "dec3": dec3}

        for cond_name in ("D4only", "D2only", "Both"):
            for stage_name in ("dec1", "dec2", "dec3"):
                for region_name in ("interior", "boundary", "background"):
                    region_mask = region_masks[stage_name][region_name]
                    stats = per_voxel_stats(feats["baseline"][stage_name], feats[cond_name][stage_name], region_mask)
                    if stats is None:
                        continue
                    records.append({
                        "subject_id": subject_id, "condition": cond_name,
                        "decoder_stage": stage_name, "region": region_name,
                        **stats,
                    })

        if (subject_idx + 1) % 25 == 0:
            print(f"  processed {subject_idx+1}/{len(val_dataset)} subjects", flush=True)

    with open(OUT_DIR / "E43_representation_change_table.json", "w") as f:
        json.dump(records, f)
    print(f"\nSaved {len(records)} records.", flush=True)


if __name__ == "__main__":
    main()
