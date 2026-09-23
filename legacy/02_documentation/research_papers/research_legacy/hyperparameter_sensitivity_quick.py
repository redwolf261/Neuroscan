# ===========================================================================================
# Hyperparameter Sensitivity Analysis (QUICK VERSION)
# ===========================================================================================
# Quick test with reduced epochs (10) and fewer configurations
# Tests only k_slices [3, 5, 7] and window_size [4, 8] = 6 configs
# 
# Expected runtime: 1-2 hours on RTX 2050
# 
# Results saved to: research/hyperparam_sensitivity_results_quick/
# ===========================================================================================

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    NormalizeIntensityd, Compose, MapTransform, EnsureTyped
)
from monai.data import Dataset
from monai.losses import DiceLoss

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 3
SPATIAL_SIZE = (64, 64, 64)
EPOCHS = 10  # QUICK VERSION - reduced from 50
LR = 3e-4

# Hyperparameters to test (REDUCED)
K_SLICES_VALUES = [3, 5, 7]  # Removed 9
WINDOW_SIZE_VALUES = [4, 8]  # Removed 16

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "hyperparam_sensitivity_results_quick")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("HYPERPARAMETER SENSITIVITY ANALYSIS (QUICK)")
print("=" * 80)
print(f"Device: {DEVICE}")
print(f"Testing k_slices: {K_SLICES_VALUES}")
print(f"Testing window_size: {WINDOW_SIZE_VALUES}")
print(f"Total configurations: {len(K_SLICES_VALUES) * len(WINDOW_SIZE_VALUES)}")
print(f"Epochs per config: {EPOCHS} (QUICK TEST)")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 80)

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
    
    # Split into train/val (80/20)
    split_idx = int(0.8 * len(data_dicts))
    train_files = data_dicts[:split_idx]
    val_files = data_dicts[split_idx:]
    
    # Transforms
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        BinarizeLabel(keys=["label"]),
        EnsureTyped(keys=["image", "label"])
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
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
# SIMPLIFIED MODEL
# ===========================================================================================

class SimplifiedModel(nn.Module):
    """Simplified 2.5D model for hyperparameter testing"""
    def __init__(self, k_slices=5, window_size=4, channels=[32, 64, 128, 256]):
        super().__init__()
        self.k = k_slices
        self.window_size = window_size
        
        # 2.5D stem
        self.stem_conv = nn.Conv2d(1, channels[0], 3, 1, 1)
        self.stem_bn = nn.BatchNorm2d(channels[0])
        self.stem_relu = nn.ReLU(inplace=True)
        
        # Encoder stages
        self.encoder = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.encoder.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], 3, 2, 1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels[i+1], channels[i+1], 3, 1, 1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(inplace=True)
            ))
        
        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.decoder.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(channels[i], channels[i-1], 3, 1, 1),
                nn.BatchNorm2d(channels[i-1]),
                nn.ReLU(inplace=True)
            ))
        
        # Final output
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
        
        # Extract center k slices
        center = D // 2
        start_idx = max(0, center - self.k // 2)
        end_idx = min(D, start_idx + self.k)
        
        # Process center slice (simplified for speed)
        center_slice = x[:, :, center, :, :]  # (B, 1, H, W)
        
        # Stem
        x = self.stem_conv(center_slice)
        x = self.stem_bn(x)
        x = self.stem_relu(x)
        
        # Encoder
        skip_connections = []
        for enc in self.encoder:
            skip_connections.append(x)
            x = enc(x)
        
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
        
        # Get center slice label
        center_label = labels[:, :, labels.shape[2]//2, :, :]
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, center_label)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Dice
        pred_binary = (outputs > 0.5).float()
        dice = 2 * (pred_binary * center_label).sum() / (pred_binary.sum() + center_label.sum() + 1e-7)
        
        total_loss += loss.item()
        total_dice += dice.item()
    
    return total_loss / len(loader), total_dice / len(loader)

def validate(model, loader, criterion):
    """Validate model"""
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
# MAIN EXECUTION
# ===========================================================================================

def main():
    # Load data once
    print("\nLoading dataset...")
    train_loader, val_loader = load_data()
    
    # Results storage
    results = {}
    
    # Test all configurations
    config_num = 0
    total_configs = len(K_SLICES_VALUES) * len(WINDOW_SIZE_VALUES)
    
    for k_slices in K_SLICES_VALUES:
        for window_size in WINDOW_SIZE_VALUES:
            config_num += 1
            config_name = f"k{k_slices}_w{window_size}"
            
            print("\n" + "=" * 80)
            print(f"Configuration {config_num}/{total_configs}: {config_name}")
            print(f"k_slices={k_slices}, window_size={window_size}")
            print("=" * 80)
            
            # Create model
            model = SimplifiedModel(k_slices=k_slices, window_size=window_size).to(DEVICE)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            
            # Training setup
            criterion = DiceLoss(sigmoid=False)
            optimizer = optim.Adam(model.parameters(), lr=LR)
            scaler = GradScaler()
            
            # Training
            best_dice = 0
            train_losses = []
            train_dices = []
            val_losses = []
            val_dices = []
            
            start_time = time.time()
            
            for epoch in range(1, EPOCHS + 1):
                print(f"\nEpoch {epoch}/{EPOCHS}")
                
                train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, scaler)
                val_loss, val_dice = validate(model, val_loader, criterion)
                
                train_losses.append(train_loss)
                train_dices.append(train_dice)
                val_losses.append(val_loss)
                val_dices.append(val_dice)
                
                print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
                print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
                
                if val_dice > best_dice:
                    best_dice = val_dice
            
            training_time = time.time() - start_time
            
            # Store results
            results[config_name] = {
                "k_slices": k_slices,
                "window_size": window_size,
                "final_val_dice": val_dices[-1],
                "best_val_dice": best_dice,
                "final_val_loss": val_losses[-1],
                "total_params": total_params,
                "trainable_params": trainable_params,
                "training_time_seconds": training_time,
                "train_losses": train_losses,
                "train_dices": train_dices,
                "val_losses": val_losses,
                "val_dices": val_dices
            }
            
            print(f"\n✅ Config {config_name} complete:")
            print(f"   Best Val Dice: {best_dice:.4f}")
            print(f"   Training time: {training_time/60:.1f} minutes")
    
    # Save results
    results_file = os.path.join(OUTPUT_DIR, "sensitivity_results_quick.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ All configurations tested! Results saved to {results_file}")
    
    # Generate plots
    generate_plots(results)
    
    print("\n" + "=" * 80)
    print("QUICK ANALYSIS COMPLETE")
    print("=" * 80)

def generate_plots(results):
    """Generate analysis plots"""
    print("\nGenerating analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Hyperparameter Sensitivity Analysis (Quick)', fontsize=16, fontweight='bold')
    
    # Extract data
    configs = list(results.keys())
    k_values = [results[c]['k_slices'] for c in configs]
    w_values = [results[c]['window_size'] for c in configs]
    dice_values = [results[c]['best_val_dice'] for c in configs]
    times = [results[c]['training_time_seconds']/60 for c in configs]
    
    # 1. Dice heatmap
    dice_matrix = np.array(dice_values).reshape(len(K_SLICES_VALUES), len(WINDOW_SIZE_VALUES))
    im = axes[0, 0].imshow(dice_matrix, cmap='RdYlGn', aspect='auto')
    axes[0, 0].set_xticks(range(len(WINDOW_SIZE_VALUES)))
    axes[0, 0].set_yticks(range(len(K_SLICES_VALUES)))
    axes[0, 0].set_xticklabels(WINDOW_SIZE_VALUES)
    axes[0, 0].set_yticklabels(K_SLICES_VALUES)
    axes[0, 0].set_xlabel('Window Size')
    axes[0, 0].set_ylabel('K Slices')
    axes[0, 0].set_title('Dice Score Heatmap')
    plt.colorbar(im, ax=axes[0, 0])
    
    # Add values to heatmap
    for i in range(len(K_SLICES_VALUES)):
        for j in range(len(WINDOW_SIZE_VALUES)):
            axes[0, 0].text(j, i, f'{dice_matrix[i, j]:.3f}',
                           ha='center', va='center', color='black', fontsize=11)
    
    # 2. Dice by k_slices
    for w in WINDOW_SIZE_VALUES:
        k_vals = []
        d_vals = []
        for k in K_SLICES_VALUES:
            config = f"k{k}_w{w}"
            k_vals.append(k)
            d_vals.append(results[config]['best_val_dice'])
        axes[0, 1].plot(k_vals, d_vals, marker='o', label=f'window={w}', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('K Slices')
    axes[0, 1].set_ylabel('Best Val Dice')
    axes[0, 1].set_title('Dice vs K Slices')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Training time comparison
    axes[1, 0].bar(range(len(configs)), times, alpha=0.7)
    axes[1, 0].set_xticks(range(len(configs)))
    axes[1, 0].set_xticklabels(configs, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Training Time (minutes)')
    axes[1, 0].set_title('Training Time per Configuration')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Efficiency plot
    axes[1, 1].scatter(times, dice_values, s=150, alpha=0.6)
    for i, config in enumerate(configs):
        axes[1, 1].annotate(config, (times[i], dice_values[i]), fontsize=9)
    axes[1, 1].set_xlabel('Training Time (minutes)')
    axes[1, 1].set_ylabel('Best Val Dice')
    axes[1, 1].set_title('Efficiency: Performance vs Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(OUTPUT_DIR, "sensitivity_analysis_plots_quick.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plots saved to {plot_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("QUICK SUMMARY STATISTICS")
    print("=" * 80)
    best_config = max(results.keys(), key=lambda x: results[x]['best_val_dice'])
    print(f"Best configuration: {best_config}")
    print(f"  k_slices: {results[best_config]['k_slices']}")
    print(f"  window_size: {results[best_config]['window_size']}")
    print(f"  Best Val Dice: {results[best_config]['best_val_dice']:.4f}")
    print(f"  Training time: {results[best_config]['training_time_seconds']/60:.1f} minutes")
    print("\nNote: This is a quick test (10 epochs). Run full version for definitive results.")

if __name__ == "__main__":
    main()
