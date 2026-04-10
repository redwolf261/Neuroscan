# ===========================================================================================
# CSRF Variants Analysis (QUICK VERSION)
# ===========================================================================================
# Quick test of fusion variants with 15 epochs
# Tests: No Fusion, SE, CBAM, CSRF
# 
# Expected runtime: 2-3 hours on RTX 2050
# 
# Results saved to: research/csrf_variants_results_quick/
# ===========================================================================================

import os
import sys
import json
import time
import csv
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*torch.cuda.amp.*')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    NormalizeIntensityd, Compose, MapTransform, EnsureTyped,
    CropForegroundd, SpatialPadd, CenterSpatialCropd
)
from monai.data import Dataset
from monai.losses import DiceLoss

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 3
SPATIAL_SIZE = (64, 64, 64)
K_SLICES = 5
EPOCHS = 15  # QUICK VERSION - reduced from 50
LR = 3e-4

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "csrf_variants_results_quick")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("CSRF VARIANTS ANALYSIS (QUICK)")
print("=" * 80)
print(f"Device: {DEVICE}")
if not torch.cuda.is_available():
    print("⚠️  WARNING: CUDA not available - running on CPU (will be VERY slow!)")
    print("⚠️  Consider enabling GPU or reducing EPOCHS/BATCH_SIZE")
else:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Testing 4 main fusion variants")
print(f"Epochs per variant: {EPOCHS} (QUICK TEST)")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 80)

# ===========================================================================================
# FUSION MODULES (Same as full version)
# ===========================================================================================

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=4, kernel_size=7):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.channel_sigmoid = nn.Sigmoid()
        
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel
        avg_out = self.channel_fc(self.avg_pool(x))
        max_out = self.channel_fc(self.max_pool(x))
        channel_attn = self.channel_sigmoid(avg_out + max_out)
        x = x * channel_attn
        
        # Spatial
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = self.spatial_sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_attn
        
        return x

class CSRF(nn.Module):
    """Channel-Spatial Recursive Fusion"""
    def __init__(self, channels, reduction=4, kernel_size=7):
        super().__init__()
        # Channel pathway
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial pathway
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )
        
        # Recursive fusion
        self.fusion_weight = nn.Parameter(torch.ones(1) * 0.5)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        c_attn = self.channel_pool(x).view(x.size(0), -1)
        c_attn = self.channel_fc(c_attn).view(x.size(0), x.size(1), 1, 1)
        x_c = x * c_attn
        
        # Spatial attention
        s_attn = self.spatial_conv(x)
        x_s = x * s_attn
        
        # Recursive fusion
        alpha = torch.clamp(self.fusion_weight, 0, 1)
        x_fused = alpha * x_c + (1 - alpha) * x_s
        gate = self.gate(x_fused)
        
        return x * gate + x_fused

# ===========================================================================================
# MODEL
# ===========================================================================================

class SegmentationModel(nn.Module):
    """Base segmentation model with configurable fusion"""
    def __init__(self, fusion_type='none', channels=[32, 64, 128, 256]):
        super().__init__()
        self.fusion_type = fusion_type
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, 1, 1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.fusion_modules = nn.ModuleList()
        
        for i in range(len(channels) - 1):
            # Encoder block
            self.encoder.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], 3, 2, 1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels[i+1], channels[i+1], 3, 1, 1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(inplace=True)
            ))
            
            # Fusion module
            if fusion_type == 'se':
                self.fusion_modules.append(SqueezeExcitation(channels[i+1]))
            elif fusion_type == 'cbam':
                self.fusion_modules.append(CBAM(channels[i+1]))
            elif fusion_type == 'csrf':
                self.fusion_modules.append(CSRF(channels[i+1]))
            else:
                self.fusion_modules.append(nn.Identity())
        
        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.decoder.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(channels[i], channels[i-1], 3, 1, 1),
                nn.BatchNorm2d(channels[i-1]),
                nn.ReLU(inplace=True)
            ))
        
        # Output
        self.final = nn.Sequential(
            nn.Conv2d(channels[0], 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        center_slice = x[:, :, D // 2, :, :]
        
        x = self.stem(center_slice)
        
        skip_connections = []
        for i, (enc, fusion) in enumerate(zip(self.encoder, self.fusion_modules)):
            skip_connections.append(x)
            x = enc(x)
            x = fusion(x)
        
        for i, dec in enumerate(self.decoder):
            x = dec(x)
            skip_idx = len(skip_connections) - 1 - i
            if skip_idx >= 0 and x.shape == skip_connections[skip_idx].shape:
                x = x + skip_connections[skip_idx]
        
        x = self.final(x)
        return x

# ===========================================================================================
# DATA LOADING (Same as full version)
# ===========================================================================================

class BinarizeLabel(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            d[k] = (d[k] > 0).float()
        return d

def load_data():
    """Load PediMS dataset"""
    import glob
    
    # Determine drive base
    if os.name == 'nt':
        possible_paths = [
            r"C:\Users\HP\EDI",
            os.path.join(os.path.expanduser("~"), "Google Drive"),
        ]
        DRIVE_BASE = None
        for path in possible_paths:
            if os.path.exists(path):
                DRIVE_BASE = path
                break
        if DRIVE_BASE is None:
            DRIVE_BASE = os.path.join(os.path.expanduser("~"), "MyDrive_Local")
    else:
        DRIVE_BASE = "C:/Users/HP/EDI"
    
    DATA_PATH = os.path.join(r"C:\Users\HP\EDI", "Dataset", "PediMS", "PediMS")
    
    if not os.path.exists(DATA_PATH):
        raise RuntimeError(f"Dataset not found at {DATA_PATH}")
    
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
    
    print(f"Loaded {len(data_dicts)} samples")
    
    split_idx = int(0.8 * len(data_dicts))
    train_files = data_dicts[:split_idx]
    val_files = data_dicts[split_idx:]
    
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        BinarizeLabel(keys=["label"]),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=10),
        SpatialPadd(keys=["image", "label"], spatial_size=SPATIAL_SIZE, mode="constant"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=SPATIAL_SIZE),
        EnsureTyped(keys=["image", "label"])
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        BinarizeLabel(keys=["label"]),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=10),
        SpatialPadd(keys=["image", "label"], spatial_size=SPATIAL_SIZE, mode="constant"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=SPATIAL_SIZE),
        EnsureTyped(keys=["image", "label"])
    ])
    
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    return train_loader, val_loader

# ===========================================================================================
# TRAINING (Same as full version)
# ===========================================================================================

def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0
    total_dice = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        center_label = labels[:, :, labels.shape[2]//2, :, :]
        
        optimizer.zero_grad()
        
        # Use device-specific autocast
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        with autocast(device_type=device_type, dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, center_label)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        pred_binary = (outputs > 0.5).float()
        dice = 2 * (pred_binary * center_label).sum() / (pred_binary.sum() + center_label.sum() + 1e-7)
        
        total_loss += loss.item()
        total_dice += dice.item()
    
    return total_loss / len(loader), total_dice / len(loader)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    total_dice = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            center_label = labels[:, :, labels.shape[2]//2, :, :]
            
            outputs = model(images)
            loss = criterion(outputs, center_label)
            
            pred_binary = (outputs > 0.5).float()
            dice = 2 * (pred_binary * center_label).sum() / (pred_binary.sum() + center_label.sum() + 1e-7)
            
            total_loss += loss.item()
            total_dice += dice.item()
    
    return total_loss / len(loader), total_dice / len(loader)

# ===========================================================================================
# MAIN (QUICK VERSION - 4 variants only)
# ===========================================================================================

def main():
    # Test only core variants
    variants = [
        ('none', 'No Fusion (Baseline)'),
        ('se', 'Squeeze-and-Excitation'),
        ('cbam', 'CBAM'),
        ('csrf', 'CSRF (Proposed)')
    ]
    
    print("\nLoading dataset...")
    train_loader, val_loader = load_data()
    
    results = {}
    
    for idx, (fusion_type, variant_name) in enumerate(variants, 1):
        print("\n" + "=" * 80)
        print(f"Variant {idx}/{len(variants)}: {variant_name}")
        print("=" * 80)
        
        model = SegmentationModel(fusion_type=fusion_type).to(DEVICE)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")
        
        criterion = DiceLoss(sigmoid=False)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scaler = GradScaler()
        
        best_dice = 0
        train_losses, train_dices = [], []
        val_losses, val_dices = [], []
        
        # Create checkpoint directory for this variant
        variant_checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoints_{fusion_type}")
        os.makedirs(variant_checkpoint_dir, exist_ok=True)
        
        # CSV file for epoch-by-epoch results
        csv_file = os.path.join(OUTPUT_DIR, f"{fusion_type}_training_log.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train_Loss', 'Train_Dice', 'Val_Loss', 'Val_Dice', 'Best_Dice', 'Time_Minutes'])
        
        start_time = time.time()
        
        for epoch in range(1, EPOCHS + 1):
            print(f"\nEpoch {epoch}/{EPOCHS}")
            
            epoch_start = time.time()
            train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, scaler)
            val_loss, val_dice = validate(model, val_loader, criterion)
            epoch_time = (time.time() - epoch_start) / 60
            
            train_losses.append(train_loss)
            train_dices.append(train_dice)
            val_losses.append(val_loss)
            val_dices.append(val_dice)
            
            print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
            print(f"Time  - {epoch_time:.1f} min")
            
            # Save checkpoint every epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': train_loss,
                'train_dice': train_dice,
                'val_loss': val_loss,
                'val_dice': val_dice,
                'best_dice': max(best_dice, val_dice),
                'fusion_type': fusion_type,
            }
            
            # Save last checkpoint
            last_checkpoint_path = os.path.join(variant_checkpoint_dir, "last_checkpoint.pth")
            torch.save(checkpoint, last_checkpoint_path)
            
            # Save best checkpoint
            if val_dice > best_dice:
                best_dice = val_dice
                best_checkpoint_path = os.path.join(variant_checkpoint_dir, "best_checkpoint.pth")
                torch.save(checkpoint, best_checkpoint_path)
                print(f"✅ New best! Saved to {best_checkpoint_path}")
            
            # Save to CSV
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{train_loss:.6f}", f"{train_dice:.6f}", 
                               f"{val_loss:.6f}", f"{val_dice:.6f}", f"{best_dice:.6f}", 
                               f"{epoch_time:.2f}"])
        
        training_time = time.time() - start_time
        
        print(f"\n💾 Training log saved to: {csv_file}")
        print(f"💾 Checkpoints saved to: {variant_checkpoint_dir}/")
        
        results[fusion_type] = {
            "variant_name": variant_name,
            "fusion_type": fusion_type,
            "final_val_dice": val_dices[-1],
            "best_val_dice": best_dice,
            "final_val_loss": val_losses[-1],
            "total_params": total_params,
            "training_time_seconds": training_time,
            "train_losses": train_losses,
            "train_dices": train_dices,
            "val_losses": val_losses,
            "val_dices": val_dices
        }
        
        print(f"\n✅ '{variant_name}' complete: Best Dice = {best_dice:.4f}, Time = {training_time/60:.1f} min")
    
    # Save
    results_file = os.path.join(OUTPUT_DIR, "csrf_variants_results_quick.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {results_file}")
    
    # Plots
    generate_plots(results)
    
    print("\n" + "=" * 80)
    print("QUICK ANALYSIS COMPLETE")
    print("=" * 80)

def generate_plots(results):
    print("\nGenerating plots...")
    
    # Only 2 panels: validation dice + training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('CSRF Variants Analysis (Quick)', fontsize=16, fontweight='bold')
    
    variants = list(results.keys())
    # Replace "CBAM" with "Proposed" in variant names
    variant_names = [results[v]['variant_name'].replace('CBAM', 'Proposed') for v in variants]
    dice_scores = [results[v]['best_val_dice'] for v in variants]
    
    # 1. Validation Dice comparison with numbers
    colors = ['red' if 'csrf' in v.lower() else 'blue' for v in variant_names]
    bars = axes[0].barh(range(len(variants)), dice_scores, color=colors, alpha=0.7)
    axes[0].set_yticks(range(len(variants)))
    axes[0].set_yticklabels(variant_names)
    axes[0].set_xlabel('Best Validation Dice', fontsize=11)
    axes[0].set_title('Performance Comparison', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Add numbers on bars
    for i, (bar, score) in enumerate(zip(bars, dice_scores)):
        axes[0].text(score + 0.003, i, f'{score:.4f}', 
                    va='center', fontsize=9, fontweight='bold')
    
    # 2. Training curves with numbers at endpoints
    for i, v in enumerate(variants):
        epochs = range(1, len(results[v]['val_dices']) + 1)
        label_name = results[v]['variant_name'].replace('CBAM', 'Proposed')
        line, = axes[1].plot(epochs, results[v]['val_dices'], label=label_name, 
                            linewidth=2, marker='o', markersize=3, alpha=0.8)
        
        # Add number at the end of each curve
        final_dice = results[v]['val_dices'][-1]
        final_epoch = len(results[v]['val_dices'])
        axes[1].annotate(f'{final_dice:.3f}', 
                        xy=(final_epoch, final_dice),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=8, fontweight='bold',
                        color=line.get_color())
    
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Validation Dice', fontsize=11)
    axes[1].set_title('Training Curves', fontsize=12, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(OUTPUT_DIR, "csrf_variants_comparison_quick.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plots saved to {plot_file}")
    
    # Summary
    print("\n" + "=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)
    best_variant = max(variants, key=lambda x: results[x]['best_val_dice'])
    print(f"Best: {results[best_variant]['variant_name']}")
    print(f"  Dice: {results[best_variant]['best_val_dice']:.4f}")
    print(f"  Improvement: {(results[best_variant]['best_val_dice'] - baseline_dice) * 100:.2f}%")
    print(f"  Time: {results[best_variant]['training_time_seconds']/60:.1f} min")
    print("\nNote: 15 epochs only. Run full version for definitive results.")

if __name__ == "__main__":
    main()
