# ===========================================================================================
# CSRF Variants Analysis
# ===========================================================================================
# Tests different channel-spatial fusion strategies
# Variants: No Fusion, Average, SE, CBAM, CSRF (proposed), and combinations
# 
# Expected runtime: 8-10 hours on RTX 2050
# 
# Results saved to: research/csrf_variants_results/
# ===========================================================================================

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    NormalizeIntensityd, Compose, MapTransform, EnsureTyped,
    CropForegroundd, SpatialPadd, CenterSpatialCropd
)
from monai.data import Dataset
from monai.losses import DiceLoss
import pandas as pd

# Import research utilities
from research_utils import (
    atomic_save, compute_metrics, validate_with_metrics,
    create_experiment_directories, print_metrics, format_time
)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 3
SPATIAL_SIZE = (64, 64, 64)
K_SLICES = 5
EPOCHS = 50
LR = 3e-4

# Create experiment directories (separate from final_model.py outputs)
LOCAL_OUTPUT_DIR, GDRIVE_OUTPUT_DIR = create_experiment_directories(
    "csrf_variants",
    "Research_CSRFVariants"
)

print("=" * 80)
print("CSRF VARIANTS ANALYSIS")
print("=" * 80)
print(f"Device: {DEVICE}")
print(f"Testing 8 fusion variants")
print(f"Epochs per variant: {EPOCHS}")
print(f"Local output: {LOCAL_OUTPUT_DIR}")
print(f"Google Drive: {GDRIVE_OUTPUT_DIR}")
print("=" * 80)

# ===========================================================================================
# FUSION MODULES
# ===========================================================================================

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block (channel attention only)"""
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
        # Channel attention
        avg_out = self.channel_fc(self.avg_pool(x))
        max_out = self.channel_fc(self.max_pool(x))
        channel_attn = self.channel_sigmoid(avg_out + max_out)
        x = x * channel_attn
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = self.spatial_sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_attn
        
        return x

class CSRF(nn.Module):
    """Channel-Spatial Recursive Fusion (Proposed)"""
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
# MODEL VARIANTS
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
        """
        Args:
            x: (B, 1, D, H, W)
        Returns:
            (B, 1, H, W)
        """
        B, C, D, H, W = x.shape
        
        # Extract center slice
        center_slice = x[:, :, D // 2, :, :]  # (B, 1, H, W)
        
        # Stem
        x = self.stem(center_slice)
        
        # Encoder with fusion
        skip_connections = []
        for i, (enc, fusion) in enumerate(zip(self.encoder, self.fusion_modules)):
            skip_connections.append(x)
            x = enc(x)
            x = fusion(x)
        
        # Decoder
        for i, dec in enumerate(self.decoder):
            x = dec(x)
            skip_idx = len(skip_connections) - 1 - i
            if skip_idx >= 0 and x.shape == skip_connections[skip_idx].shape:
                x = x + skip_connections[skip_idx]
        
        # Output
        x = self.final(x)
        
        return x

# ===========================================================================================
# DATA LOADING
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
    
    print(f"Loaded {len(data_dicts)} samples from {DATA_PATH}")
    
    # Split
    split_idx = int(0.8 * len(data_dicts))
    train_files = data_dicts[:split_idx]
    val_files = data_dicts[split_idx:]
    
    # Transforms
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=SPATIAL_SIZE, mode="constant"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=SPATIAL_SIZE),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        BinarizeLabel(keys=["label"]),
        EnsureTyped(keys=["image", "label"])
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=SPATIAL_SIZE, mode="constant"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=SPATIAL_SIZE),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        BinarizeLabel(keys=["label"]),
        EnsureTyped(keys=["image", "label"])
    ])
    
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    return train_loader, val_loader

# ===========================================================================================
# TRAINING FUNCTIONS
# ===========================================================================================

def train_epoch(model, loader, criterion, optimizer, scaler):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        
        center_label = labels[:, :, labels.shape[2]//2, :, :]
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
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
    """Validate model with comprehensive metrics"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            center_label = labels[:, :, labels.shape[2]//2, :, :]
            
            outputs = model(images)
            loss = criterion(outputs, center_label)
            
            pred_binary = (outputs > 0.5).float()
            
            total_loss += loss.item()
            
            # Collect for comprehensive metrics
            all_preds.append(pred_binary.cpu().numpy().flatten())
            all_labels.append(center_label.cpu().numpy().flatten())
    
    # Concatenate all predictions
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Compute comprehensive metrics
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = total_loss / len(loader)
    
    return metrics

# ===========================================================================================
# MAIN EXECUTION
# ===========================================================================================

def main():
    # Define variants to test
    variants = [
        ('none', 'No Fusion (Baseline)'),
        ('average', 'Average Pooling'),
        ('se', 'Squeeze-and-Excitation (SE)'),
        ('cbam', 'CBAM'),
        ('csrf', 'CSRF (Proposed)'),
        ('se_residual', 'SE + Residual'),
        ('cbam_deep', 'CBAM Deep'),
        ('csrf_light', 'CSRF Lightweight')
    ]
    
    # Load data once
    print("\nLoading dataset...")
    train_loader, val_loader = load_data()
    
    # Results storage
    results = {}
    
    # Test each variant
    for idx, (fusion_type, variant_name) in enumerate(variants, 1):
        print("\n" + "=" * 80)
        print(f"Variant {idx}/{len(variants)}: {variant_name}")
        print(f"Fusion type: {fusion_type}")
        print("=" * 80)
        
        # Create model
        model = SegmentationModel(fusion_type=fusion_type).to(DEVICE)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Training setup
        criterion = DiceLoss(sigmoid=False)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scaler = GradScaler('cuda')
        
        # Create variant-specific directory
        variant_dir = os.path.join(GDRIVE_OUTPUT_DIR, variant_name.replace(" ", "_"))
        os.makedirs(variant_dir, exist_ok=True)
        
        # CSV logging lists
        train_log_data = []
        val_log_data = []
        
        # Training
        best_val_dice = 0
        start_time = time.time()
        
        for epoch in range(1, EPOCHS + 1):
            print(f"\n{'='*80}")
            print(f"📊 Epoch {epoch}/{EPOCHS} - Variant: {variant_name}")
            print(f"{'='*80}")
            
            # Train
            train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, scaler)
            
            # Validate with comprehensive metrics
            val_metrics = validate(model, val_loader, criterion)
            
            # Log to lists
            train_log_data.append({
                'epoch': epoch,
                'loss': train_loss,
                'dice': train_dice,
                'lr': optimizer.param_groups[0]['lr']
            })
            
            val_log_data.append({
                'epoch': epoch,
                'loss': val_metrics['loss'],
                'dice': val_metrics['dice'],
                'precision': val_metrics['precision'],
                'recall': val_metrics['recall'],
                'f1': val_metrics['f1'],
                'specificity': val_metrics['specificity']
            })
            
            # Save CSVs every epoch
            pd.DataFrame(train_log_data).to_csv(
                os.path.join(variant_dir, 'train_logs.csv'), index=False
            )
            pd.DataFrame(val_log_data).to_csv(
                os.path.join(variant_dir, 'val_logs.csv'), index=False
            )
            
            # Print metrics with formatting
            print(f"🔹 Train  → Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
            print(f"🔸 Val    → Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}")
            print(f"          → Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}, Specificity: {val_metrics['specificity']:.4f}")
            
            # Save best model to Google Drive
            if val_metrics['dice'] > best_val_dice:
                best_val_dice = val_metrics['dice']
                best_model_path = os.path.join(variant_dir, 'best_model.pth')
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': val_metrics['dice'],
                    'val_metrics': val_metrics,
                    'config': {
                        'fusion_type': fusion_type,
                        'variant_name': variant_name
                    }
                }
                if atomic_save(checkpoint_data, best_model_path):
                    print(f"✅ NEW BEST! Dice: {val_metrics['dice']:.4f} → Saved to Google Drive")
                else:
                    print(f"⚠️  Failed to save best model")
            else:
                print(f"📊 Current: {val_metrics['dice']:.4f} | Best: {best_val_dice:.4f} (no improvement)")
        
        training_time = time.time() - start_time
        
        # Store results for plotting
        results[fusion_type] = {
            "variant_name": variant_name,
            "fusion_type": fusion_type,
            "final_val_dice": val_log_data[-1]['dice'],
            "best_val_dice": best_val_dice,
            "final_val_loss": val_log_data[-1]['loss'],
            "total_params": total_params,
            "trainable_params": trainable_params,
            "training_time_seconds": training_time,
            # Store final epoch metrics
            "final_precision": val_log_data[-1]['precision'],
            "final_recall": val_log_data[-1]['recall'],
            "final_f1": val_log_data[-1]['f1'],
            "final_specificity": val_log_data[-1]['specificity']
        }
        
        print(f"\n✅ Variant '{variant_name}' complete:")
        print(f"   Best Val Dice: {best_val_dice:.4f}")
        print(f"   Training time: {format_time(training_time)}")
        print(f"   CSVs saved to: {variant_dir}")
        print(f"   Best model saved to Google Drive: {os.path.join(variant_dir, 'best_model.pth')}")
    
    # Save results
    results_file = os.path.join(LOCAL_OUTPUT_DIR, "csrf_variants_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ All variants tested! Results saved to {results_file}")
    
    # Generate plots
    generate_plots(results)
    
    print("\n" + "=" * 80)
    print("CSRF VARIANTS ANALYSIS COMPLETE")
    print("=" * 80)

def generate_plots(results):
    """Generate comparison plots"""
    print("\nGenerating comparison plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CSRF Variants Analysis', fontsize=16, fontweight='bold')
    
    variants = list(results.keys())
    variant_names = [results[v]['variant_name'] for v in variants]
    dice_scores = [results[v]['best_val_dice'] for v in variants]
    params = [results[v]['total_params'] for v in variants]
    times = [results[v]['training_time_seconds']/60 for v in variants]
    
    # 1. Dice comparison
    colors = ['red' if 'csrf' in v.lower() else 'blue' for v in variant_names]
    axes[0, 0].barh(range(len(variants)), dice_scores, color=colors, alpha=0.7)
    axes[0, 0].set_yticks(range(len(variants)))
    axes[0, 0].set_yticklabels(variant_names, fontsize=9)
    axes[0, 0].set_xlabel('Best Validation Dice')
    axes[0, 0].set_title('Performance Comparison')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # 2. Training curves (best 3) - Read from CSV files
    sorted_variants = sorted(variants, key=lambda x: results[x]['best_val_dice'], reverse=True)[:3]
    for v in sorted_variants:
        variant_name = results[v]['variant_name'].replace(" ", "_")
        csv_path = os.path.join(GDRIVE_OUTPUT_DIR, variant_name, 'val_logs.csv')
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            axes[0, 1].plot(df['epoch'], df['dice'], label=results[v]['variant_name'], linewidth=2)
        except Exception as e:
            print(f"⚠️ Could not load training curve for {results[v]['variant_name']}: {e}")
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Validation Dice')
    axes[0, 1].set_title('Training Curves (Top 3)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Parameters vs Performance
    axes[0, 2].scatter(params, dice_scores, s=150, alpha=0.6)
    for i, name in enumerate(variant_names):
        axes[0, 2].annotate(name, (params[i], dice_scores[i]), fontsize=8)
    axes[0, 2].set_xlabel('Total Parameters')
    axes[0, 2].set_ylabel('Best Validation Dice')
    axes[0, 2].set_title('Parameters vs Performance')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Training time
    axes[1, 0].bar(range(len(variants)), times, alpha=0.7)
    axes[1, 0].set_xticks(range(len(variants)))
    axes[1, 0].set_xticklabels(variant_names, rotation=45, ha='right', fontsize=8)
    axes[1, 0].set_ylabel('Training Time (minutes)')
    axes[1, 0].set_title('Training Time Comparison')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 5. Improvement over baseline
    baseline_dice = results['none']['best_val_dice']
    improvements = [(results[v]['best_val_dice'] - baseline_dice) * 100 for v in variants[1:]]
    axes[1, 1].barh(range(len(improvements)), improvements, alpha=0.7)
    axes[1, 1].set_yticks(range(len(improvements)))
    axes[1, 1].set_yticklabels([results[v]['variant_name'] for v in variants[1:]], fontsize=9)
    axes[1, 1].set_xlabel('Improvement over Baseline (%)')
    axes[1, 1].set_title('Relative Improvement')
    axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=1)
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    # 6. Efficiency (Dice/Time)
    efficiency = [dice_scores[i] / times[i] for i in range(len(variants))]
    axes[1, 2].bar(range(len(variants)), efficiency, alpha=0.7)
    axes[1, 2].set_xticks(range(len(variants)))
    axes[1, 2].set_xticklabels(variant_names, rotation=45, ha='right', fontsize=8)
    axes[1, 2].set_ylabel('Efficiency (Dice/min)')
    axes[1, 2].set_title('Training Efficiency')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_file = os.path.join(LOCAL_OUTPUT_DIR, "csrf_variants_comparison.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plots saved to {plot_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    best_variant = max(variants, key=lambda x: results[x]['best_val_dice'])
    print(f"Best variant: {results[best_variant]['variant_name']}")
    print(f"  Best Val Dice: {results[best_variant]['best_val_dice']:.4f}")
    print(f"  Improvement over baseline: {(results[best_variant]['best_val_dice'] - baseline_dice) * 100:.2f}%")
    print(f"  Parameters: {results[best_variant]['total_params']:,}")
    print(f"  Training time: {results[best_variant]['training_time_seconds']/60:.1f} minutes")

if __name__ == "__main__":
    main()
