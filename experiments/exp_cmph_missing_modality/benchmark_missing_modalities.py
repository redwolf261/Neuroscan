"""
Benchmark script for evaluating 3D U-Net segmentation models across all 15
possible combinations of missing/incomplete MRI modalities on BraTS 2023 GLI.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset, MODALITIES, REGIONS
from neuroscan_3d_v5 import UNet3D_v5
from neuroscan_3d_v17_cmph import UNet3D_v17_CMPH


# 15 clinical modality combinations
MODALITY_COMBOS = {
    "Full (4/4)": [1, 1, 1, 1],
    "No-T1c": [0, 1, 1, 1],
    "No-T1n": [1, 0, 1, 1],
    "No-T2f": [1, 1, 0, 1],
    "No-T2w": [1, 1, 1, 0],
    "T2-only (No T1c/T1n)": [0, 0, 1, 1],
    "No T1c/T2f": [0, 1, 0, 1],
    "No T1c/T2w": [0, 1, 1, 0],
    "No T1n/T2f": [1, 0, 0, 1],
    "No T1n/T2w": [1, 0, 1, 0],
    "T1-only (No T2f/T2w)": [1, 1, 0, 0],
    "T1c-only": [1, 0, 0, 0],
    "T1n-only": [0, 1, 0, 0],
    "T2f-only": [0, 0, 1, 0],
    "T2w-only": [0, 0, 0, 1]
}


def compute_dice_score(preds, targets, eps=1e-5):
    """
    Computes per-region Dice score: preds, targets are binary (3, D, H, W).
    Regions: 0=ET, 1=TC, 2=WT
    """
    dice_scores = {}
    for i, reg in enumerate(REGIONS):
        p = preds[i]
        t = targets[i]
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        if union == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = (2.0 * intersection) / (union + eps)
        dice_scores[reg] = float(dice)
    dice_scores['mean'] = float(np.mean([dice_scores[r] for r in REGIONS]))
    return dice_scores


def evaluate_model_on_combos(model, dataset, device, num_subjects=20, patch_size=(128, 128, 128)):
    """
    Evaluates model across all 15 modality combinations.
    """
    model.eval()
    results = {combo: {'ET': [], 'TC': [], 'WT': [], 'mean': []} for combo in MODALITY_COMBOS}

    with torch.no_grad():
        for subj_idx in range(min(num_subjects, len(dataset))):
            vol, target, sid = dataset[subj_idx]
            vol = vol.unsqueeze(0).to(device) # (1, 4, D, H, W)
            target = target.numpy()           # (3, D, H, W)

            # Extract center crop padded to exact patch_size (128, 128, 128)
            D, H, W = vol.shape[2:]
            pd, ph, pw = patch_size
            cd, ch, cw = D // 2, H // 2, W // 2
            sd, ed = max(0, cd - pd//2), min(D, cd + pd//2)
            sh, eh = max(0, ch - ph//2), min(H, ch + ph//2)
            sw, ew = max(0, cw - pw//2), min(W, cw + pw//2)

            sub_vol = vol[:, :, sd:ed, sh:eh, sw:ew]
            sub_tgt = target[:, sd:ed, sh:eh, sw:ew]

            # Pad to exact (pd, ph, pw)
            pad_d = pd - sub_vol.shape[2]
            pad_h = ph - sub_vol.shape[3]
            pad_w = pw - sub_vol.shape[4]
            patch_vol = F.pad(sub_vol, (0, pad_w, 0, pad_h, 0, pad_d))
            patch_target = np.pad(sub_tgt, ((0,0), (0, pad_d), (0, pad_h), (0, pad_w)))

            for combo_name, mask_list in MODALITY_COMBOS.items():
                mask_tensor = torch.tensor([mask_list], dtype=torch.float32, device=device) # (1, 4)
                
                # Apply mask to input volume
                masked_vol = patch_vol * mask_tensor.view(1, 4, 1, 1, 1)

                if hasattr(model, 'cmph_gate'):
                    out = model(masked_vol, mask_tensor)
                    probs = torch.sigmoid(out['logits'])[0].cpu().numpy()
                else:
                    out = model(masked_vol)
                    if isinstance(out, dict):
                        if 'probs' in out:
                            probs = out['probs'][0].cpu().numpy()
                        else:
                            probs = torch.sigmoid(out['logits'])[0].cpu().numpy()
                    else:
                        probs = torch.sigmoid(out)[0].cpu().numpy()

                preds = (probs > 0.5).astype(np.float32)

                dices = compute_dice_score(preds, patch_target)
                for reg in ['ET', 'TC', 'WT', 'mean']:
                    results[combo_name][reg].append(dices[reg])

    summary = {}
    for combo_name, vals in results.items():
        summary[combo_name] = {
            'ET': float(np.mean(vals['ET'])),
            'TC': float(np.mean(vals['TC'])),
            'WT': float(np.mean(vals['WT'])),
            'mean': float(np.mean(vals['mean']))
        }
    return summary


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    val_dataset = BraTSMultimodalDataset(root_dir="Dataset/Training", split='val')
    print(f"Loaded validation set: {len(val_dataset)} subjects.")

    # 1. Evaluate baseline UNet3D_v5
    print("\n--- Evaluating Baseline UNet3D_v5 on 15 Modality Permutations ---")
    model_v5 = UNet3D_v5(in_channels=4, out_channels=3).to(device)
    # Check if checkpoint exists
    ckpt_path = "checkpoints/E131_v5control_seed0/best.pth"
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}...")
        model_v5.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    summary_v5 = evaluate_model_on_combos(model_v5, val_dataset, device, num_subjects=15)
    
    # 2. Evaluate UNet3D_v17_CMPH
    print("\n--- Evaluating UNet3D_v17_CMPH on 15 Modality Permutations ---")
    model_v17 = UNet3D_v17_CMPH().to(device)
    cmph_ckpt = "checkpoints/E230_cmph_seed0/best_cmph.pth"
    if os.path.exists(cmph_ckpt):
        print(f"Loading trained CMPH checkpoint from {cmph_ckpt}...")
        model_v17.load_state_dict(torch.load(cmph_ckpt, map_location=device))
    summary_v17 = evaluate_model_on_combos(model_v17, val_dataset, device, num_subjects=15)

    # Print Comparison Table
    print("\n" + "="*80)
    print(f"{'Modality Scenario':<25} | {'Baseline (v5) Mean Dice':<22} | {'CMPH-Net (v17) Mean Dice':<22} | {'Delta':<8}")
    print("="*80)
    for combo in MODALITY_COMBOS:
        b_mean = summary_v5[combo]['mean']
        c_mean = summary_v17[combo]['mean']
        diff = c_mean - b_mean
        print(f"{combo:<25} | {b_mean:<22.4f} | {c_mean:<22.4f} | {diff:+.4f}")
    print("="*80)

    # Save results to JSON
    os.makedirs("experiments/exp_cmph_missing_modality", exist_ok=True)
    with open("experiments/exp_cmph_missing_modality/benchmark_results.json", "w") as f:
        json.dump({"baseline_v5": summary_v5, "cmph_v17": summary_v17}, f, indent=2)
    print("Benchmark results saved to experiments/exp_cmph_missing_modality/benchmark_results.json")
