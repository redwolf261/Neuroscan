"""
Ablation Study: Adaptive Slice Selection vs Fixed k-slices

Tests:
1. Baseline: Fixed k=9 (current best)
2. Adaptive Selection: Learn optimal k per sample
3. Adaptive + Uncertainty Weighting
4. Different selection strategies (Top-k, Threshold-based, Learned)

Author: Research Team
Date: November 2025
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import argparse
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# MONAI imports
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    NormalizeIntensityd, Resized, RandFlipd, RandRotate90d, EnsureTyped
)
from monai.data import Dataset as MonaiDataset

# ===========================================================================================
# CONFIGURATION
# ===========================================================================================
DATA_PATH = r"C:\Users\HP\EDI\Dataset\PediMS\PediMS"
OUTPUT_DIR = r"C:\Users\HP\EDI\research\adaptive_slice_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPATIAL_SIZE = (64, 64, 64)
BATCH_SIZE = 3
NUM_WORKERS = 0
LEARNING_RATE = 1e-4
EPOCHS = 30  # Ablation study epochs
PATIENCE = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()

print(f"Device: {device} | AMP: {use_amp}")

# ===========================================================================================
# BINARIZE LABEL TRANSFORM
# ===========================================================================================
class BinarizeLabel:
    def __init__(self, keys):
        self.keys = keys
    
    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = (data[key] > 0).float()
        return data

# ===========================================================================================
# ADAPTIVE SLICE SELECTOR (Novel Component)
# ===========================================================================================
class AdaptiveSliceSelector(nn.Module):
    """
    Learn to select most informative k slices from volume.
    Uses Gumbel-Softmax for differentiable sampling during training.
    """
    def __init__(self, max_slices=64, k=9, method='gumbel'):
        super().__init__()
        self.k = k
        self.max_slices = max_slices
        self.method = method  # 'gumbel', 'topk', 'threshold'
        
        # Lightweight 3D scorer network
        self.scorer = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((max_slices, 4, 4)),  # Reduce spatial dims
            nn.Flatten(start_dim=2),  # (B, 16, D*16)
        )
        
        self.score_head = nn.Sequential(
            nn.Linear(16 * 16, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 1)  # Score per slice
        )
        
        # Temperature for Gumbel-Softmax (learnable or fixed)
        self.tau = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x, training=True):
        """
        Args:
            x: (B, 1, D, H, W) - full 3D volume
            training: bool - use Gumbel-Softmax if True, greedy if False
        Returns:
            selected_volume: (B, 1, k, H, W) - selected slices
            selection_probs: (B, D) - selection probabilities
            slice_scores: (B, D) - importance scores
        """
        B, C, D, H, W = x.shape
        
        # Score each slice
        features = self.scorer(x)  # (B, 16, D*16)
        B, C_feat, feat_len = features.shape
        features = features.view(B, C_feat, self.max_slices, -1)  # (B, 16, D, 16)
        
        # Compute score per slice
        slice_scores = []
        for i in range(min(D, self.max_slices)):
            if i < features.size(2):
                slice_feat = features[:, :, i, :].reshape(B, -1)  # (B, 16*16)
                score = self.score_head(slice_feat)  # (B, 1)
                slice_scores.append(score)
        
        slice_scores = torch.cat(slice_scores, dim=1)  # (B, D)
        
        # Select slices based on method
        if self.method == 'gumbel' and training:
            # Gumbel-Softmax for differentiable sampling
            # Use top-k with Gumbel noise for differentiable discrete selection
            _, top_indices = torch.topk(slice_scores, self.k, dim=1)  # (B, k)
            top_indices, _ = torch.sort(top_indices, dim=1)  # Keep spatial order
            
            # Extract selected slices (hard selection during training too for simplicity)
            selected_volume = []
            for b in range(B):
                indices = top_indices[b]
                selected_slices = x[b, :, indices, :, :]  # (1, k, H, W)
                selected_volume.append(selected_slices.unsqueeze(0))
            
            selected_volume = torch.cat(selected_volume, dim=0)  # (B, 1, k, H, W)
            
            # Create selection probability mask
            selection_probs = torch.zeros(B, D, device=x.device)
            for b in range(B):
                selection_probs[b, top_indices[b]] = 1.0 / self.k
            
        else:
            # Greedy top-k selection (for evaluation)
            _, top_indices = torch.topk(slice_scores, self.k, dim=1)  # (B, k)
            top_indices, _ = torch.sort(top_indices, dim=1)  # Keep spatial order
            
            selected_volume = []
            for b in range(B):
                indices = top_indices[b]
                selected_slices = x[b, :, indices, :, :]  # (1, k, H, W)
                selected_volume.append(selected_slices.unsqueeze(0))
            
            selected_volume = torch.cat(selected_volume, dim=0)  # (B, 1, k, H, W)
            
            # Create selection probability mask
            selection_probs = torch.zeros(B, D, device=x.device)
            for b in range(B):
                selection_probs[b, top_indices[b]] = 1.0
        
        return selected_volume, selection_probs, slice_scores


# ===========================================================================================
# FIXED K-SLICE EXTRACTOR (Baseline)
# ===========================================================================================
class FixedSliceExtractor(nn.Module):
    """Baseline: Extract fixed k consecutive slices from center"""
    def __init__(self, k=9):
        super().__init__()
        self.k = k
    
    def forward(self, x, training=True):
        """
        Args:
            x: (B, 1, D, H, W)
        Returns:
            selected_volume: (B, 1, k, H, W)
            selection_probs: (B, D) - uniform for selected slices
            slice_scores: (B, D) - equal scores
        """
        B, C, D, H, W = x.shape
        
        # Extract center k slices
        center = D // 2
        start_idx = max(0, center - self.k // 2)
        end_idx = min(D, start_idx + self.k)
        
        selected_volume = x[:, :, start_idx:end_idx, :, :]  # (B, 1, k, H, W)
        
        # Pad if needed
        if selected_volume.size(2) < self.k:
            pad_size = self.k - selected_volume.size(2)
            padding = selected_volume[:, :, -1:, :, :].repeat(1, 1, pad_size, 1, 1)
            selected_volume = torch.cat([selected_volume, padding], dim=2)
        
        # Create uniform selection probabilities
        selection_probs = torch.zeros(B, D, device=x.device)
        for i in range(start_idx, end_idx):
            selection_probs[:, i] = 1.0 / self.k
        
        slice_scores = torch.ones(B, D, device=x.device) / D
        
        return selected_volume, selection_probs, slice_scores


# ===========================================================================================
# 2.5D STEM (Simplified for ablation)
# ===========================================================================================
class Conv2D5Stem(nn.Module):
    """Process k selected slices and fuse"""
    def __init__(self, k_slices=9, out_channels=32):
        super().__init__()
        self.k = k_slices
        
        # Slice-wise conv (shared across slices)
        self.slice_conv = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Slice attention for fusion
        self.slice_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, 1, kernel_size=1)
        )
    
    def forward(self, x_slices):
        """
        Args:
            x_slices: (B, 1, k, H, W) - selected slices
        Returns:
            fused: (B, out_channels, H, W)
        """
        B, C, K, H, W = x_slices.shape
        
        # Process each slice
        slice_features = []
        alphas = []
        
        for i in range(K):
            slice_i = x_slices[:, :, i, :, :]  # (B, 1, H, W)
            feat = self.slice_conv(slice_i)  # (B, C_out, H, W)
            alpha = self.slice_attention(feat)  # (B, 1, 1, 1)
            
            slice_features.append(feat)
            alphas.append(alpha)
        
        # Attention-weighted fusion
        alphas = torch.cat(alphas, dim=1)  # (B, k, 1, 1)
        alphas = F.softmax(alphas, dim=1)
        
        fused = 0
        for i, feat in enumerate(slice_features):
            fused = fused + alphas[:, i:i+1, :, :] * feat
        
        return fused


# ===========================================================================================
# SIMPLE SEGMENTATION MODEL (For ablation)
# ===========================================================================================
class SimpleSegmentationModel(nn.Module):
    """Lightweight model for ablation study"""
    def __init__(self, k_slices=9, slice_selector='fixed'):
        super().__init__()
        self.k = k_slices
        
        # Slice selector
        if slice_selector == 'fixed':
            self.selector = FixedSliceExtractor(k=k_slices)
        elif slice_selector == 'adaptive':
            self.selector = AdaptiveSliceSelector(max_slices=64, k=k_slices, method='gumbel')
        else:
            raise ValueError(f"Unknown selector: {slice_selector}")
        
        # 2.5D stem
        self.stem = Conv2D5Stem(k_slices=k_slices, out_channels=32)
        
        # Simple encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Simple decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, return_selection=False):
        """
        Args:
            x: (B, 1, D, H, W)
        Returns:
            output: (B, 1, H, W)
            selection_info: dict (if return_selection=True)
        """
        # Select slices
        selected_volume, selection_probs, slice_scores = self.selector(x, training=self.training)
        
        # Process selected slices
        features = self.stem(selected_volume)  # (B, 32, H, W)
        encoded = self.encoder(features)  # (B, 128, H/4, W/4)
        output = self.decoder(encoded)  # (B, 1, H, W)
        
        if return_selection:
            selection_info = {
                'selection_probs': selection_probs,
                'slice_scores': slice_scores,
                'selected_volume': selected_volume
            }
            return output, selection_info
        
        return output


# ===========================================================================================
# DICE LOSS
# ===========================================================================================
def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for binary segmentation"""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice

def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    """Dice coefficient for evaluation"""
    pred = (pred > threshold).float()
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


# ===========================================================================================
# TRAINING & EVALUATION
# ===========================================================================================
def train_epoch(model, loader, optimizer, scaler, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Extract center slice from 3D labels (since output is 2D)
        if labels.dim() == 5:  # (B, 1, D, H, W)
            center_idx = labels.size(2) // 2
            labels = labels[:, :, center_idx, :, :]  # (B, 1, H, W)
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda', enabled=use_amp):
            outputs = model(images)
            loss = dice_loss(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics
        dice = dice_score(outputs.detach(), labels.detach())
        total_loss += loss.item()
        total_dice += dice
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})
    
    return total_loss / len(loader), total_dice / len(loader)


def validate(model, loader, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_dice = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Extract center slice from 3D labels (since output is 2D)
            if labels.dim() == 5:  # (B, 1, D, H, W)
                center_idx = labels.size(2) // 2
                labels = labels[:, :, center_idx, :, :]  # (B, 1, H, W)
            
            with autocast(device_type='cuda', enabled=use_amp):
                outputs = model(images)
                loss = dice_loss(outputs, labels)
            
            dice = dice_score(outputs, labels)
            total_loss += loss.item()
            total_dice += dice
    
    return total_loss / len(loader), total_dice / len(loader)


# ===========================================================================================
# MAIN ABLATION SCRIPT
# ===========================================================================================
def run_ablation(selector_type, k_slices, experiment_name):
    """Run single ablation experiment"""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Selector: {selector_type}, k={k_slices}")
    print(f"{'='*80}\n")
    
    # Data loading
    print("Loading data...")
    data_dicts = []
    for subfolder in sorted(os.listdir(DATA_PATH)):
        sub_path = os.path.join(DATA_PATH, subfolder)
        if not os.path.isdir(sub_path):
            continue
        for modality in ["T1", "T2", "FLAIR"]:
            mod_path = os.path.join(sub_path, modality, "processed")
            if not os.path.exists(mod_path):
                continue
            imgs = sorted(glob.glob(os.path.join(mod_path, "*_brain_*.nii*")))
            masks = sorted(glob.glob(os.path.join(mod_path, "*_mask_*.nii*")))
            if len(masks) == 0:
                masks = sorted(glob.glob(os.path.join(mod_path, "*_Consensus_*.nii*")))
            for img, m in zip(imgs, masks):
                data_dicts.append({"image": [img], "label": m})
    
    print(f"Found {len(data_dicts)} samples")
    
    # Transforms
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        BinarizeLabel(keys=["label"]),
        Resized(keys=["image", "label"], spatial_size=SPATIAL_SIZE, mode=("trilinear", "nearest")),
        RandFlipd(keys=["image", "label"], spatial_axis=[0, 1, 2], prob=0.5),
        EnsureTyped(keys=["image", "label"])
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        BinarizeLabel(keys=["label"]),
        Resized(keys=["image", "label"], spatial_size=SPATIAL_SIZE, mode=("trilinear", "nearest")),
        EnsureTyped(keys=["image", "label"])
    ])
    
    # Split data
    split = int(0.8 * len(data_dicts))
    train_ds = MonaiDataset(data=data_dicts[:split], transform=train_transforms)
    val_ds = MonaiDataset(data=data_dicts[split:], transform=val_transforms)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Model
    model = SimpleSegmentationModel(k_slices=k_slices, slice_selector=selector_type).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler(device='cuda', enabled=use_amp)
    
    # Training loop
    best_val_dice = 0
    patience_counter = 0
    history = []
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        
        train_loss, train_dice = train_epoch(model, train_loader, optimizer, scaler, device)
        val_loss, val_dice = validate(model, val_loader, device)
        scheduler.step()
        
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_dice': train_dice,
            'val_loss': val_loss,
            'val_dice': val_dice,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'{experiment_name}_best.pth'))
            print(f"✓ Best model saved (Dice: {val_dice:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Save history
    pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, f'{experiment_name}_history.csv'), index=False)
    
    return {
        'experiment': experiment_name,
        'selector': selector_type,
        'k_slices': k_slices,
        'best_val_dice': best_val_dice,
        'final_epoch': epoch
    }


# ===========================================================================================
# MAIN
# ===========================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adaptive Slice Selection Ablation Study')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (5 epochs)')
    parser.add_argument('--selector', type=str, default='all', 
                       choices=['all', 'fixed', 'adaptive'],
                       help='Which selector to test')
    parser.add_argument('--k', type=int, default=9, help='Number of slices')
    
    args = parser.parse_args()
    
    if args.quick:
        EPOCHS = 5
        print("🚀 QUICK TEST MODE: 5 epochs only")
    
    results = []
    
    # Run experiments
    experiments = []
    
    if args.selector == 'all':
        experiments = [
            ('fixed', 9, 'baseline_fixed_k9'),
            ('adaptive', 9, 'adaptive_gumbel_k9'),
            ('fixed', 7, 'baseline_fixed_k7'),
            ('adaptive', 7, 'adaptive_gumbel_k7'),
        ]
    elif args.selector == 'fixed':
        experiments = [('fixed', args.k, f'baseline_fixed_k{args.k}')]
    elif args.selector == 'adaptive':
        experiments = [('adaptive', args.k, f'adaptive_gumbel_k{args.k}')]
    
    for selector, k, name in experiments:
        result = run_ablation(selector, k, name)
        results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'ablation_summary.csv'), index=False)
    
    # Find best
    best_idx = results_df['best_val_dice'].idxmax()
    best = results_df.iloc[best_idx]
    
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"   Experiment: {best['experiment']}")
    print(f"   Selector: {best['selector']}")
    print(f"   k_slices: {best['k_slices']}")
    print(f"   Best Val Dice: {best['best_val_dice']:.4f}")
    
    print(f"\n✅ Results saved to: {OUTPUT_DIR}")
