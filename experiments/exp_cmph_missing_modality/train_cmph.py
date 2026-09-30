"""
Training script for UNet3D_v17_CMPH on BraTS 2023 GLI under dynamic modality dropout.
"""

import os
import sys
import time
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset
from neuroscan_3d_v17_cmph import UNet3D_v17_CMPH, compute_cmph_loss


def random_modality_dropout(p_drop=0.5):
    """
    Randomly drops 0 to 3 modalities to simulate all clinical incomplete acquisition scenarios.
    Returns binary mask (1, 4) with at least one modality present.
    """
    if random.random() > p_drop:
        return torch.ones((1, 4), dtype=torch.float32)
    
    # Sample dropout mask ensuring at least 1 modality is kept
    mask = [1 if random.random() > 0.4 else 0 for _ in range(4)]
    if sum(mask) == 0:
        mask[random.randint(0, 3)] = 1
    return torch.tensor([mask], dtype=torch.float32)


def train_cmph_epoch(model, loader, optimizer, scaler, device, epoch, max_batches=50):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for i, (images, targets, sid) in enumerate(loader):
        if i >= max_batches:
            break

        images = images.to(device)   # (B, 4, D, H, W)
        targets = targets.to(device) # (B, 3, D, H, W)
        B = images.shape[0]

        optimizer.zero_grad()

        # Generate modality mask per item in batch
        masks = [random_modality_dropout() for _ in range(B)]
        modality_mask = torch.cat(masks, dim=0).to(device) # (B, 4)

        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            # Full modality pass as teacher (when mask has missing entries)
            teacher_out = None
            if modality_mask.mean() < 1.0:
                with torch.no_grad():
                    teacher_mask = torch.ones((B, 4), dtype=torch.float32, device=device)
                    teacher_out = model(images, teacher_mask)

            # Incomplete modality pass as student
            outputs = model(images, modality_mask)
            loss = compute_cmph_loss(outputs, targets, teacher_out)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--save_dir", type=str, default="checkpoints/E230_cmph_seed0")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training UNet3D_v17_CMPH on {device}...")

    os.makedirs(args.save_dir, exist_ok=True)

    train_dataset = BraTSMultimodalDataset(root_dir="Dataset/Training", split='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = UNet3D_v17_CMPH().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    print(f"Starting {args.epochs} epochs of CMPH Training...")
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_cmph_epoch(model, train_loader, optimizer, scaler, device, epoch)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Time: {elapsed:.1f}s")

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_cmph.pth"))

    print(f"Training completed. Model saved to {args.save_dir}/best_cmph.pth")
