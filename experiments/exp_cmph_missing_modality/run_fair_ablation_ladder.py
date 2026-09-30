"""
Rigorous Fair Ablation Ladder for Incomplete/Missing Modality Brain Tumor Segmentation.

Ablation Arms (Trained under identical seed, data, and modality-dropout distribution):
- Arm 1: Standard UNet3D + Modality Dropout (The true, fair baseline without broken BatchNorm)
- Arm 2: Multi-Stem UNet3D + Modality Dropout (Stem Disentanglement only, no prototype bank)
- Arm 3: Full CMPH-Net (Multi-Stem + Missingness-Conditioned Prototype Memory Bank + Cross-Attention)
"""

import os
import sys
import time
import math
import random
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset, MODALITIES, REGIONS
from neuroscan_3d_v5 import UNet3D_v5
from neuroscan_3d_v17_cmph import UNet3D_v17_CMPH, DoubleConv3D


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


class MultiStemUNet_NoProto(nn.Module):
    """
    Arm 2: Multi-Stem Disentangled U-Net WITHOUT prototype memory bank.
    Isolates whether separate input stems provide benefit over standard U-Net.
    """
    def __init__(self, in_channels=4, out_channels=3, base_channels=32):
        super().__init__()
        self.stem_t1c = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.stem_t1n = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.stem_t2f = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.stem_t2w = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.enc1_fusion = DoubleConv3D(64, base_channels)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv3D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv3D(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv3D(base_channels * 4, base_channels * 8)

        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3D(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(base_channels * 2, base_channels)
        self.seg_head = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, modality_mask=None):
        B = x.shape[0]
        if modality_mask is None:
            modality_mask = torch.ones((B, 4), dtype=torch.float32, device=x.device)

        m_t1c = modality_mask[:, 0].view(B, 1, 1, 1, 1)
        m_t1n = modality_mask[:, 1].view(B, 1, 1, 1, 1)
        m_t2f = modality_mask[:, 2].view(B, 1, 1, 1, 1)
        m_t2w = modality_mask[:, 3].view(B, 1, 1, 1, 1)

        f_t1c = self.stem_t1c(x[:, 0:1]) * m_t1c
        f_t1n = self.stem_t1n(x[:, 1:2]) * m_t1n
        f_t2f = self.stem_t2f(x[:, 2:3]) * m_t2f
        f_t2w = self.stem_t2w(x[:, 3:4]) * m_t2w

        e1 = self.enc1_fusion(torch.cat([f_t1c, f_t1n, f_t2f, f_t2w], dim=1))
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        bn = self.bottleneck(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(bn), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits = self.seg_head(d1)
        return {'logits': logits}


def random_modality_dropout(p_drop=0.5):
    if random.random() > p_drop:
        return torch.ones((1, 4), dtype=torch.float32)
    mask = [1 if random.random() > 0.4 else 0 for _ in range(4)]
    if sum(mask) == 0:
        mask[random.randint(0, 3)] = 1
    return torch.tensor([mask], dtype=torch.float32)


def train_single_arm(model, train_loader, epochs=5, lr=0.0003, max_batches=50, device='cuda', name="Arm"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    print(f"\n--- Training {name} for {epochs} epochs ---")
    model.train()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        total_loss = 0.0
        num_b = 0

        for i, (images, targets, sid) in enumerate(train_loader):
            if i >= max_batches:
                break
            images = images.to(device)
            targets = targets.to(device)
            B = images.shape[0]

            optimizer.zero_grad()
            masks = [random_modality_dropout() for _ in range(B)]
            modality_mask = torch.cat(masks, dim=0).to(device)

            # Apply mask directly to input for standard UNet, or pass mask parameter
            masked_images = images * modality_mask.view(B, 4, 1, 1, 1)

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                if hasattr(model, 'cmph_gate'):
                    out = model(images, modality_mask)
                    logits = out['logits']
                elif isinstance(model, MultiStemUNet_NoProto):
                    out = model(images, modality_mask)
                    logits = out['logits']
                else:
                    out = model(masked_images)
                    logits = out['probs'] if isinstance(out, dict) and 'probs' in out else (out['logits'] if isinstance(out, dict) else out)

                probs = torch.sigmoid(logits) if not (isinstance(model, UNet3D_v5) and 'probs' in out) else logits
                smooth = 1e-5
                intersection = (probs * targets).sum(dim=(2, 3, 4))
                union = probs.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4))
                dice_loss = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
                loss = dice_loss.mean()

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_b += 1

        elapsed = time.time() - t0
        print(f"[{name}] Epoch {epoch:02d}/{epochs:02d} | Loss: {total_loss/max(num_b,1):.4f} | Time: {elapsed:.1f}s")
    return model


def evaluate_arm(model, val_dataset, device, num_subjects=15, patch_size=(128, 128, 128)):
    model.eval()
    results = {combo: [] for combo in MODALITY_COMBOS}

    with torch.no_grad():
        for subj_idx in range(min(num_subjects, len(val_dataset))):
            vol, target, sid = val_dataset[subj_idx]
            vol = vol.unsqueeze(0).to(device)
            target = target.numpy()

            D, H, W = vol.shape[2:]
            pd, ph, pw = patch_size
            cd, ch, cw = D // 2, H // 2, W // 2
            sd, ed = max(0, cd - pd//2), min(D, cd + pd//2)
            sh, eh = max(0, ch - ph//2), min(H, ch + ph//2)
            sw, ew = max(0, cw - pw//2), min(W, cw + pw//2)

            sub_vol = vol[:, :, sd:ed, sh:eh, sw:ew]
            sub_tgt = target[:, sd:ed, sh:eh, sw:ew]

            pad_d = pd - sub_vol.shape[2]
            pad_h = ph - sub_vol.shape[3]
            pad_w = pw - sub_vol.shape[4]
            patch_vol = F.pad(sub_vol, (0, pad_w, 0, pad_h, 0, pad_d))
            patch_tgt = np.pad(sub_tgt, ((0,0), (0, pad_d), (0, pad_h), (0, pad_w)))

            for combo_name, mask_list in MODALITY_COMBOS.items():
                mask_tensor = torch.tensor([mask_list], dtype=torch.float32, device=device)
                masked_vol = patch_vol * mask_tensor.view(1, 4, 1, 1, 1)

                if hasattr(model, 'cmph_gate'):
                    out = model(masked_vol, mask_tensor)
                    probs = torch.sigmoid(out['logits'])[0].cpu().numpy()
                elif isinstance(model, MultiStemUNet_NoProto):
                    out = model(masked_vol, mask_tensor)
                    probs = torch.sigmoid(out['logits'])[0].cpu().numpy()
                else:
                    out = model(masked_vol)
                    if isinstance(out, dict) and 'probs' in out:
                        probs = out['probs'][0].cpu().numpy()
                    elif isinstance(out, dict) and 'logits' in out:
                        probs = torch.sigmoid(out['logits'])[0].cpu().numpy()
                    else:
                        probs = torch.sigmoid(out)[0].cpu().numpy()

                preds = (probs > 0.5).astype(np.float32)

                # Mean dice across ET, TC, WT
                dices = []
                for i in range(3):
                    p = preds[i]
                    t = patch_tgt[i]
                    inter = (p * t).sum()
                    union = p.sum() + t.sum()
                    d = 1.0 if union == 0 and inter == 0 else (2.0 * inter) / (union + 1e-5)
                    dices.append(d)
                results[combo_name].append(float(np.mean(dices)))

    return {k: float(np.mean(v)) for k, v in results.items()}


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Fair Ablation Ladder on {device}...")

    train_dataset = BraTSMultimodalDataset(root_dir="Dataset/Training", split='train', seed=42)
    val_dataset = BraTSMultimodalDataset(root_dir="Dataset/Training", split='val', seed=42)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)

    # 1. Arm 1: Standard UNet3D + Modality Dropout (True Fair Baseline)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    arm1 = UNet3D_v5(in_channels=4, out_channels=3).to(device)
    arm1 = train_single_arm(arm1, train_loader, epochs=5, device=device, name="Arm 1: Standard UNet + Dropout")
    res_arm1 = evaluate_arm(arm1, val_dataset, device)

    # 2. Arm 2: Multi-Stem UNet3D + Modality Dropout (Stem Disentanglement Only)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    arm2 = MultiStemUNet_NoProto(in_channels=4, out_channels=3).to(device)
    arm2 = train_single_arm(arm2, train_loader, epochs=5, device=device, name="Arm 2: Multi-Stem (No Prototypes)")
    res_arm2 = evaluate_arm(arm2, val_dataset, device)

    # 3. Arm 3: Full CMPH-Net (Multi-Stem + Prototype Memory Bank + Cross-Attention)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    arm3 = UNet3D_v17_CMPH(in_channels=4, out_channels=3).to(device)
    arm3 = train_single_arm(arm3, train_loader, epochs=5, device=device, name="Arm 3: Full CMPH-Net (With Prototypes)")
    res_arm3 = evaluate_arm(arm3, val_dataset, device)

    # Print Full Comparative Ablation Matrix
    print("\n" + "="*105)
    print(f"{'Modality Scenario':<22} | {'Arm 1 (Std Dropout)':<20} | {'Arm 2 (Multi-Stem)':<18} | {'Arm 3 (Full CMPH)':<18} | {'Delta (Arm3 - Arm2)':<18}")
    print("="*105)
    
    delta_proto_list = []
    delta_stem_list = []
    
    for combo in MODALITY_COMBOS:
        d1 = res_arm1[combo]
        d2 = res_arm2[combo]
        d3 = res_arm3[combo]
        delta_stem = d2 - d1
        delta_proto = d3 - d2
        delta_proto_list.append(delta_proto)
        delta_stem_list.append(delta_stem)
        print(f"{combo:<22} | {d1:<20.4f} | {d2:<18.4f} | {d3:<18.4f} | {delta_proto:+18.4f}")

    print("="*105)
    print(f"{'AVERAGE':<22} | {np.mean(list(res_arm1.values())):<20.4f} | {np.mean(list(res_arm2.values())):<18.4f} | {np.mean(list(res_arm3.values())):<18.4f} | {np.mean(delta_proto_list):+18.4f}")
    print("="*105)

    out_file = "experiments/exp_cmph_missing_modality/fair_ablation_results.json"
    with open(out_file, "w") as f:
        json.dump({"arm1_std_dropout": res_arm1, "arm2_multi_stem": res_arm2, "arm3_full_cmph": res_arm3}, f, indent=2)
    print(f"\nFair ablation ladder results saved to {out_file}")
