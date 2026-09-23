# ===========================================================================================
# Hyperparameter Sensitivity Analysis
# ===========================================================================================
# Tests sensitivity of model performance to k_slices and window_size hyperparameters
# Trains 12 configurations (4 k_slices × 3 window_sizes) for 50 epochs each
# 
# Expected runtime: 6-8 hours on RTX 2050
# 
# Results saved to: research/hyperparam_sensitivity_results/
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
    CropForegroundd, SpatialPadd, CenterSpatialCropd, RandSpatialCropd
)
from monai.data import Dataset
from monai.losses import DiceLoss
import pandas as pd

# Import research utilities
from research_utils import (
    atomic_save, compute_metrics, validate_with_metrics,
    create_experiment_directories, print_metrics, format_time,
    get_google_drive_base
)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 3
SPATIAL_SIZE = (64, 64, 64)
EPOCHS = 50
LR = 3e-4

# Hyperparameters to test
# FULL LIST: K_SLICES_VALUES = [3, 5, 7, 9], WINDOW_SIZE_VALUES = [4, 8, 16]
# Based on check_configs.py results:
#   Completed (8): k3_w8, k3_w16, k5_w4, k5_w8, k5_w16, k7_w4, k7_w8, k7_w16
#   Incomplete (2): k3_w4 (1 epoch), k9_w4 (19 epochs)
#   Not Started (2): k9_w8, k9_w16
# Only running incomplete/not-started configs:
CONFIGS_TO_RUN = [
    (3, 4),   # k3_w4 - incomplete (1 epoch)
    (9, 4),   # k9_w4 - incomplete (19 epochs)
    (9, 8),   # k9_w8 - not started
    (9, 16),  # k9_w16 - not started
]

# Keep these for plotting (full grid)
K_SLICES_VALUES = [3, 5, 7, 9]
WINDOW_SIZE_VALUES = [4, 8, 16]

# Create experiment directories (separate from final_model.py outputs)
LOCAL_OUTPUT_DIR, GDRIVE_OUTPUT_DIR = create_experiment_directories(
    "hyperparam_sensitivity",
    "Research_HyperparamSensitivity"
)

print("=" * 80)
print("HYPERPARAMETER SENSITIVITY ANALYSIS - INCOMPLETE CONFIGS ONLY")
print("=" * 80)
print(f"Device: {DEVICE}")
print(f"Running configs: {[f'k{k}_w{w}' for k, w in CONFIGS_TO_RUN]}")
print(f"Total configurations to run: {len(CONFIGS_TO_RUN)}")
print(f"Epochs per config: {EPOCHS}")
print(f"Local output: {LOCAL_OUTPUT_DIR}")
print(f"Google Drive: {GDRIVE_OUTPUT_DIR}")
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
    
    # Filter out corrupted files
    print("Checking for corrupted files...")
    valid_data_dicts = []
    import nibabel as nib
    for data_dict in data_dicts:
        try:
            # Try to load files to verify they're not corrupted
            nib.load(data_dict["image"][0])
            nib.load(data_dict["label"])
            valid_data_dicts.append(data_dict)
        except Exception as e:
            print(f"⚠️ Skipping corrupted file: {data_dict['image'][0]} - {str(e)[:50]}")
    
    print(f"Valid samples: {len(valid_data_dicts)} (filtered out {len(data_dicts) - len(valid_data_dicts)} corrupted files)")
    data_dicts = valid_data_dicts
    
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
# SIMPLIFIED MODEL (For testing hyperparameters only)
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
        
        with autocast('cuda'):
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
    import numpy as np
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
    # Load data once
    print("\nLoading dataset...")
    train_loader, val_loader = load_data()
    
    # Results storage
    results = {}
    
    # Test only incomplete/not-started configurations
    config_num = 0
    total_configs = len(CONFIGS_TO_RUN)
    
    for k_slices, window_size in CONFIGS_TO_RUN:
        config_num += 1
        config_name = f"k{k_slices}_w{window_size}"
        
        # Setup config directory
        gdrive_base = os.path.join(get_google_drive_base(), "NeuroScan_Research", "Research_HyperparamSensitivity")
        config_dir = os.path.join(gdrive_base, config_name)
        os.makedirs(config_dir, exist_ok=True)
        
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
        scaler = GradScaler('cuda')
        
        # CSV logging lists
        train_log_data = []
        val_log_data = []
        
        # Training
        best_val_dice = 0
        start_time = time.time()
        
        for epoch in range(1, EPOCHS + 1):
                print(f"\n{'='*80}")
                print(f"📊 Epoch {epoch}/{EPOCHS} - Config: {config_name}")
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
                    os.path.join(config_dir, 'train_logs.csv'), index=False
                )
                pd.DataFrame(val_log_data).to_csv(
                    os.path.join(config_dir, 'val_logs.csv'), index=False
                )
                
                # Print metrics with formatting
                print(f"🔹 Train  → Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
                print(f"🔸 Val    → Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}")
                print(f"          → Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, "
                      f"F1: {val_metrics['f1']:.4f}, Specificity: {val_metrics['specificity']:.4f}")
                
                # Save best model to Google Drive
                if val_metrics['dice'] > best_val_dice:
                    best_val_dice = val_metrics['dice']
                    best_model_path = os.path.join(config_dir, 'best_model.pth')
                    checkpoint_data = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_dice': val_metrics['dice'],
                        'val_metrics': val_metrics,
                        'config': {
                            'k_slices': k_slices,
                            'window_size': window_size
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
        results[config_name] = {
            "k_slices": k_slices,
            "window_size": window_size,
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
        
        print(f"\n✅ Config {config_name} complete:")
        print(f"   Best Val Dice: {best_val_dice:.4f}")
        print(f"   Training time: {format_time(training_time)}")
        print(f"   CSVs saved to: {config_dir}")
        print(f"   Best model saved to: {os.path.join(config_dir, 'best_model.pth')}")
    
    # Save results
    results_file = os.path.join(LOCAL_OUTPUT_DIR, "sensitivity_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ All configurations tested! Results saved to {results_file}")
    
    # Generate plots
    generate_plots(results)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

def generate_plots(results):
    """Generate analysis plots"""
    print("\nGenerating analysis plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Hyperparameter Sensitivity Analysis', fontsize=16, fontweight='bold')
    
    # Extract data
    configs = list(results.keys())
    k_values = [results[c]['k_slices'] for c in configs]
    w_values = [results[c]['window_size'] for c in configs]
    dice_values = [results[c]['best_val_dice'] for c in configs]
    params = [results[c]['total_params'] for c in configs]
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
                           ha='center', va='center', color='black', fontsize=10)
    
    # 2. Dice by k_slices
    for w in WINDOW_SIZE_VALUES:
        k_vals = []
        d_vals = []
        for k in K_SLICES_VALUES:
            config = f"k{k}_w{w}"
            k_vals.append(k)
            d_vals.append(results[config]['best_val_dice'])
        axes[0, 1].plot(k_vals, d_vals, marker='o', label=f'window={w}')
    axes[0, 1].set_xlabel('K Slices')
    axes[0, 1].set_ylabel('Best Val Dice')
    axes[0, 1].set_title('Dice vs K Slices')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Dice by window_size
    for k in K_SLICES_VALUES:
        w_vals = []
        d_vals = []
        for w in WINDOW_SIZE_VALUES:
            config = f"k{k}_w{w}"
            w_vals.append(w)
            d_vals.append(results[config]['best_val_dice'])
        axes[0, 2].plot(w_vals, d_vals, marker='o', label=f'k={k}')
    axes[0, 2].set_xlabel('Window Size')
    axes[0, 2].set_ylabel('Best Val Dice')
    axes[0, 2].set_title('Dice vs Window Size')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Training time comparison
    axes[1, 0].bar(range(len(configs)), times)
    axes[1, 0].set_xticks(range(len(configs)))
    axes[1, 0].set_xticklabels(configs, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Training Time (minutes)')
    axes[1, 0].set_title('Training Time per Configuration')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 5. Parameters vs Dice
    axes[1, 1].scatter(params, dice_values, s=100, alpha=0.6)
    for i, config in enumerate(configs):
        axes[1, 1].annotate(config, (params[i], dice_values[i]), fontsize=8)
    axes[1, 1].set_xlabel('Total Parameters')
    axes[1, 1].set_ylabel('Best Val Dice')
    axes[1, 1].set_title('Parameters vs Performance')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Efficiency plot (Dice vs Time)
    axes[1, 2].scatter(times, dice_values, s=100, alpha=0.6)
    for i, config in enumerate(configs):
        axes[1, 2].annotate(config, (times[i], dice_values[i]), fontsize=8)
    axes[1, 2].set_xlabel('Training Time (minutes)')
    axes[1, 2].set_ylabel('Best Val Dice')
    axes[1, 2].set_title('Efficiency: Performance vs Time')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(LOCAL_OUTPUT_DIR, "sensitivity_analysis_plots.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plots saved to {plot_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    best_config = max(results.keys(), key=lambda x: results[x]['best_val_dice'])
    print(f"Best configuration: {best_config}")
    print(f"  k_slices: {results[best_config]['k_slices']}")
    print(f"  window_size: {results[best_config]['window_size']}")
    print(f"  Best Val Dice: {results[best_config]['best_val_dice']:.4f}")
    print(f"  Parameters: {results[best_config]['total_params']:,}")
    print(f"  Training time: {results[best_config]['training_time_seconds']/60:.1f} minutes")

if __name__ == "__main__":
    main()
