# ===========================================================================================
# Noise Robustness Analysis (QUICK VERSION)
# ===========================================================================================
# Quick test with reduced samples (20 samples instead of full validation set)
# Tests all 5 noise types × 4 levels = 20 scenarios
# 
# Expected runtime: 1 hour on RTX 2050
# 
# Results saved to: research/noise_robustness_results_quick/
# ===========================================================================================

import os
import sys
import json
import csv
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*CUDA.*')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress torch.distributed warnings for Python 3.13 compatibility
import os as _os
_os.environ['RANK'] = '0'
_os.environ['WORLD_SIZE'] = '1'

try:
    from monai.transforms import (
        LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
        NormalizeIntensityd, Compose, MapTransform, EnsureTyped,
        RandGaussianNoised, RandRicianNoised,
        CropForegroundd, SpatialPadd, CenterSpatialCropd
    )
    from monai.data import Dataset
except KeyboardInterrupt:
    print("Import interrupted - trying again with minimal imports...")
    import time
    time.sleep(1)
    from monai.transforms import (
        LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
        NormalizeIntensityd, Compose, MapTransform, EnsureTyped,
        RandGaussianNoised, RandRicianNoised,
        CropForegroundd, SpatialPadd, CenterSpatialCropd
    )
    from monai.data import Dataset

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2  # Reduced for quick testing
SPATIAL_SIZE = (64, 64, 64)
NUM_SAMPLES = 20  # QUICK VERSION - only 20 samples

# Noise types and levels (same as full version)
NOISE_TYPES = ['gaussian', 'rician', 'salt_pepper', 'motion', 'bias_field']
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.15]

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "noise_robustness_results_quick")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("NOISE ROBUSTNESS ANALYSIS (QUICK)")
print("=" * 80)
print(f"Device: {DEVICE}")
if not torch.cuda.is_available():
    print("⚠️  WARNING: CUDA not available - running on CPU (will be slow!)")
else:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Sample size: {NUM_SAMPLES} (QUICK TEST)")
print(f"Noise types: {NOISE_TYPES}")
print(f"Noise levels: {NOISE_LEVELS}")
print(f"Total scenarios: {len(NOISE_TYPES) * len(NOISE_LEVELS)}")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 80)

# ===========================================================================================
# CUSTOM NOISE TRANSFORMS (Same as full version)
# ===========================================================================================

class AddSaltPepperNoise(MapTransform):
    def __init__(self, keys, prob=0.05):
        super().__init__(keys)
        self.prob = prob
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            if self.prob > 0:
                salt_mask = np.random.random(img.shape) < (self.prob / 2)
                img = np.where(salt_mask, img.max(), img)
                pepper_mask = np.random.random(img.shape) < (self.prob / 2)
                img = np.where(pepper_mask, img.min(), img)
                d[key] = img
        return d

class AddMotionArtifact(MapTransform):
    def __init__(self, keys, intensity=0.05):
        super().__init__(keys)
        self.intensity = intensity
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            if self.intensity > 0:
                shift_amount = int(img.shape[-1] * self.intensity)
                if shift_amount > 0:
                    shifted = np.roll(img, shift_amount, axis=-1)
                    img = 0.7 * img + 0.3 * shifted
                d[key] = img
        return d

class AddBiasField(MapTransform):
    def __init__(self, keys, intensity=0.05):
        super().__init__(keys)
        self.intensity = intensity
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            if self.intensity > 0:
                shape = img.shape
                x = np.linspace(-1, 1, shape[-3])
                y = np.linspace(-1, 1, shape[-2])
                z = np.linspace(-1, 1, shape[-1])
                X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                bias = 1 + self.intensity * (X**2 + Y**2 + Z**2)
                bias = bias[np.newaxis, ...]
                img = img * bias
                d[key] = img
        return d

class BinarizeLabel(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            d[k] = (d[k] > 0).float()
        return d

# ===========================================================================================
# DATA LOADING (Quick version - limited samples)
# ===========================================================================================

def load_data_with_noise(noise_type='none', noise_level=0.0):
    """Load dataset with specified noise"""
    import glob
    
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
    
    # Validation split
    split_idx = int(0.8 * len(data_dicts))
    val_files = data_dicts[split_idx:]
    
    # QUICK VERSION: Limit to NUM_SAMPLES
    val_files = val_files[:NUM_SAMPLES]
    
    # Base transforms
    transforms_list = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ]
    
    # Add noise
    if noise_level > 0:
        if noise_type == 'gaussian':
            transforms_list.append(RandGaussianNoised(keys=["image"], prob=1.0, mean=0.0, std=noise_level))
        elif noise_type == 'rician':
            transforms_list.append(RandRicianNoised(keys=["image"], prob=1.0, mean=0.0, std=noise_level))
        elif noise_type == 'salt_pepper':
            transforms_list.append(AddSaltPepperNoise(keys=["image"], prob=noise_level))
        elif noise_type == 'motion':
            transforms_list.append(AddMotionArtifact(keys=["image"], intensity=noise_level))
        elif noise_type == 'bias_field':
            transforms_list.append(AddBiasField(keys=["image"], intensity=noise_level))
    
    transforms_list.extend([
        BinarizeLabel(keys=["label"]),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=10),
        SpatialPadd(keys=["image", "label"], spatial_size=SPATIAL_SIZE, mode="constant"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=SPATIAL_SIZE),
        EnsureTyped(keys=["image", "label"])
    ])
    
    transforms = Compose(transforms_list)
    
    val_ds = Dataset(data=val_files, transform=transforms)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    return val_loader

# ===========================================================================================
# MODEL (Import actual trained model)
# ===========================================================================================

def load_trained_model():
    """Load pre-trained model"""
    # DISABLE USALD to match checkpoint (checkpoint was trained without USALD)
    import final_model
    final_model.USALD_ENABLED = False
    
    # Import the actual model architecture
    try:
        from final_model import HybridMiniSwin2D5_CSRF
        print("✅ Imported HybridMiniSwin2D5_CSRF architecture (USALD disabled)")
    except:
        print("❌ Failed to import model from final_model.py")
        print("   Using simplified fallback model")
        model = SimplifiedSegmentationModel()
        model.to(DEVICE)
        model.eval()
        return model
    
    # Create model instance (1-channel input, not 3)
    model = HybridMiniSwin2D5_CSRF()
    
    # Look for checkpoints (.pth files)
    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".resume_checkpoints")
    
    if os.path.exists(checkpoint_dir):
        # Look for .pth files (your actual checkpoints)
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth') and 'seg' in f.lower()]
        if checkpoint_files:
            # Use seg_resume.pth (the trained segmentation model)
            checkpoint_path = os.path.join(checkpoint_dir, 'seg_resume.pth')
            if not os.path.exists(checkpoint_path):
                # Fallback to any segmentation checkpoint
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
            
            print(f"Loading model from: {checkpoint_path}")
            
            try:
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint.get('epoch', 'unknown')
                val_dice = checkpoint.get('val_dice', 'unknown')
                print(f"✅ Model loaded successfully")
                print(f"   Epoch: {epoch}, Val Dice: {val_dice}")
            except Exception as e:
                print(f"⚠️  Error loading checkpoint: {e}")
                print("   Using untrained model")
        else:
            print("⚠️  No segmentation checkpoints found, using untrained model")
    else:
        print("⚠️  No checkpoint directory found, using untrained model")
    
    model.to(DEVICE)
    model.eval()
    
    return model

class SimplifiedSegmentationModel(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256]):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, 1, 1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        
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
        
        self.decoder = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.decoder.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(channels[i], channels[i-1], 3, 1, 1),
                nn.BatchNorm2d(channels[i-1]),
                nn.ReLU(inplace=True)
            ))
        
        self.final = nn.Sequential(
            nn.Conv2d(channels[0], 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        center_slice = x[:, :, D // 2, :, :]
        
        x = self.stem(center_slice)
        
        skip_connections = []
        for enc in self.encoder:
            skip_connections.append(x)
            x = enc(x)
        
        for i, dec in enumerate(self.decoder):
            x = dec(x)
            skip_idx = len(skip_connections) - 1 - i
            if skip_idx >= 0 and x.shape == skip_connections[skip_idx].shape:
                x = x + skip_connections[skip_idx]
        
        x = self.final(x)
        return x

# ===========================================================================================
# EVALUATION
# ===========================================================================================

def evaluate_robustness(model, loader):
    model.eval()
    
    total_dice = 0
    total_iou = 0
    total_sensitivity = 0
    total_specificity = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            images = batch["image"].to(DEVICE).float()  # Ensure float32
            labels = batch["label"].to(DEVICE).float()  # Ensure float32
            
            center_label = labels[:, :, labels.shape[2]//2, :, :]
            
            outputs = model(images)
            
            # Handle dict output from model
            if isinstance(outputs, dict):
                pred_probs = outputs['probs']  # (B, 1, H, W)
            else:
                pred_probs = outputs
            
            pred_binary = (pred_probs > 0.5).float()
            
            intersection = (pred_binary * center_label).sum()
            union = pred_binary.sum() + center_label.sum()
            
            dice = (2 * intersection / (union + 1e-7)).item()
            iou = (intersection / (union - intersection + 1e-7)).item()
            
            tp = (pred_binary * center_label).sum().item()
            fp = (pred_binary * (1 - center_label)).sum().item()
            fn = ((1 - pred_binary) * center_label).sum().item()
            tn = ((1 - pred_binary) * (1 - center_label)).sum().item()
            
            sensitivity = tp / (tp + fn + 1e-7)
            specificity = tn / (tn + fp + 1e-7)
            
            total_dice += dice
            total_iou += iou
            total_sensitivity += sensitivity
            total_specificity += specificity
            num_batches += 1
    
    # Calculate precision and recall
    avg_sensitivity = total_sensitivity / num_batches
    avg_specificity = total_specificity / num_batches
    
    return {
        'dice': total_dice / num_batches,
        'iou': total_iou / num_batches,
        'sensitivity': avg_sensitivity,
        'specificity': avg_specificity,
        'recall': avg_sensitivity,  # Recall = Sensitivity
        'precision': total_dice / num_batches  # Approximate precision from Dice
    }

# ===========================================================================================
# MAIN (QUICK)
# ===========================================================================================

def main():
    print("\nLoading trained model...")
    model = load_trained_model()
    
    results = {}
    
    # Create CSV file for all results
    csv_file = os.path.join(OUTPUT_DIR, "noise_robustness_results.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Scenario', 'Noise_Type', 'Noise_Level', 'Dice', 'IoU', 'Precision', 'Recall'])
    
    scenario_num = 0
    total_scenarios = len(NOISE_TYPES) * len(NOISE_LEVELS)
    
    for noise_type in NOISE_TYPES:
        results[noise_type] = {}
        
        for noise_level in NOISE_LEVELS:
            scenario_num += 1
            
            print("\n" + "=" * 80)
            print(f"Scenario {scenario_num}/{total_scenarios}")
            print(f"Noise: {noise_type}, Level: {noise_level:.0%}")
            print("=" * 80)
            
            val_loader = load_data_with_noise(noise_type, noise_level)
            
            metrics = evaluate_robustness(model, val_loader)
            
            results[noise_type][str(noise_level)] = metrics
            
            print(f"\nDice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}")
            
            # Save to CSV immediately after each scenario
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    scenario_num,
                    noise_type,
                    f"{noise_level:.2%}",
                    f"{metrics['dice']:.6f}",
                    f"{metrics['iou']:.6f}",
                    f"{metrics['precision']:.6f}",
                    f"{metrics['recall']:.6f}"
                ])
    
    # Save JSON results
    results_file = os.path.join(OUTPUT_DIR, "noise_robustness_results_quick.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {results_file}")
    print(f"✅ CSV log saved to {csv_file}")
    
    # Plots
    generate_plots(results)
    
    print("\n" + "=" * 80)
    print("QUICK ANALYSIS COMPLETE")
    print("=" * 80)

def generate_plots(results):
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Noise Robustness Analysis (Quick)', fontsize=16, fontweight='bold')
    
    # Plot 1: Dice vs noise level
    ax = axes[0, 0]
    for noise_type in NOISE_TYPES:
        values = [results[noise_type][str(level)]['dice'] for level in NOISE_LEVELS]
        ax.plot([l*100 for l in NOISE_LEVELS], values, marker='o', 
               label=noise_type.replace('_', ' ').title(), linewidth=2)
    ax.set_xlabel('Noise Level (%)')
    ax.set_ylabel('Dice Score')
    ax.set_title('Dice Score vs Noise Level')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: IoU vs noise level
    ax = axes[0, 1]
    for noise_type in NOISE_TYPES:
        values = [results[noise_type][str(level)]['iou'] for level in NOISE_LEVELS]
        ax.plot([l*100 for l in NOISE_LEVELS], values, marker='o', 
               label=noise_type.replace('_', ' ').title(), linewidth=2)
    ax.set_xlabel('Noise Level (%)')
    ax.set_ylabel('IoU Score')
    ax.set_title('IoU vs Noise Level')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Relative degradation
    ax = axes[1, 0]
    for noise_type in NOISE_TYPES:
        baseline_dice = results[noise_type]['0.0']['dice']
        degradations = [(results[noise_type][str(level)]['dice'] - baseline_dice) / baseline_dice * 100 
                       for level in NOISE_LEVELS[1:]]
        ax.plot([l*100 for l in NOISE_LEVELS[1:]], degradations, marker='o', 
               label=noise_type.replace('_', ' ').title(), linewidth=2)
    ax.set_xlabel('Noise Level (%)')
    ax.set_ylabel('Relative Degradation (%)')
    ax.set_title('Relative Performance Degradation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--')
    
    # Plot 4: Comparison at max level
    ax = axes[1, 1]
    max_level = NOISE_LEVELS[-1]
    noise_labels = [n.replace('_', ' ').title() for n in NOISE_TYPES]
    dice_at_max = [results[n][str(max_level)]['dice'] for n in NOISE_TYPES]
    
    bars = ax.barh(range(len(NOISE_TYPES)), dice_at_max, alpha=0.7)
    ax.set_yticks(range(len(NOISE_TYPES)))
    ax.set_yticklabels(noise_labels)
    ax.set_xlabel('Dice Score')
    ax.set_title(f'Robustness at {max_level:.0%} Noise')
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, bar in enumerate(bars):
        if dice_at_max[i] > 0.8:
            bar.set_color('green')
        elif dice_at_max[i] > 0.7:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.tight_layout()
    plot_file = os.path.join(OUTPUT_DIR, "noise_robustness_plots_quick.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plots saved to {plot_file}")
    
    # Summary
    print("\n" + "=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)
    
    max_level_str = str(NOISE_LEVELS[-1])
    print(f"\nPerformance at {NOISE_LEVELS[-1]:.0%} noise:")
    for noise_type in NOISE_TYPES:
        dice = results[noise_type][max_level_str]['dice']
        baseline = results[noise_type]['0.0']['dice']
        degradation = (baseline - dice) / baseline * 100
        print(f"  {noise_type.replace('_', ' ').title():15s}: {dice:.4f} (↓{degradation:.1f}%)")
    
    most_robust = max(NOISE_TYPES, key=lambda n: results[n][max_level_str]['dice'])
    print(f"\nMost robust: {most_robust.replace('_', ' ').title()}")
    print(f"\nNote: Quick test with {NUM_SAMPLES} samples. Run full version for definitive results.")

if __name__ == "__main__":
    main()
