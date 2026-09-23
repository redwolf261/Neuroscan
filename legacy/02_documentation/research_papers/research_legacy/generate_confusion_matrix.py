# ===========================================================================================
# Confusion Matrix Generation for Segmentation Model
# ===========================================================================================
# Generates confusion matrices showing pixel-level classification performance
# Includes: Overall confusion matrix, per-class metrics, and visualizations
# 
# Expected runtime: 10-15 minutes on RTX 2050
# 
# Results saved to: paper_figures/confusion_matrix/
# ===========================================================================================

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import actual model architecture
from final_model import HybridMiniSwin2D5_CSRF

from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    NormalizeIntensityd, Compose, MapTransform, EnsureTyped,
    SpatialPadd, CenterSpatialCropd, Resized
)
from monai.data import Dataset

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
SPATIAL_SIZE = (64, 64, 64)
NUM_SAMPLES = 50  # Number of validation samples to use

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "paper_figures", "confusion_matrix")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("CONFUSION MATRIX GENERATION")
print("=" * 80)
print(f"Device: {DEVICE}")
print(f"Samples: {NUM_SAMPLES}")
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

def load_validation_data():
    """Load validation data"""
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
    
    print(f"Total samples: {len(data_dicts)}")
    
    # Validation split
    split_idx = int(0.8 * len(data_dicts))
    val_files = data_dicts[split_idx:split_idx + NUM_SAMPLES]
    
    print(f"Using {len(val_files)} validation samples")
    
    # Transforms
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        # Resize to consistent spatial size
        Resized(keys=["image", "label"], spatial_size=SPATIAL_SIZE, mode=("trilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        BinarizeLabel(keys=["label"]),
        EnsureTyped(keys=["image", "label"])
    ])
    
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    return val_loader

# ===========================================================================================
# MODEL LOADING
# ===========================================================================================

class SimplifiedSegmentationModel(nn.Module):
    """Simplified model for evaluation"""
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

def load_trained_model():
    """Load pre-trained model using actual architecture"""
    # Use the actual model architecture
    model = HybridMiniSwin2D5_CSRF()
    
    # Priority 1: Google Drive best_model.pth (highest quality)
    # Priority 2: Local .resume_checkpoints/seg_resume.pth (resume checkpoint)
    gdrive_best_path = r"C:\Users\HP\EDI\NeuroScan_FinalModel_2.5D_MAE\segmentation\best_model.pth"
    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".resume_checkpoints")
    local_resume_path = os.path.join(checkpoint_dir, "seg_resume.pth")
    
    # Determine which checkpoint to use
    if os.path.exists(gdrive_best_path):
        checkpoint_path = gdrive_best_path
        print(f"📂 Loading BEST MODEL from Google Drive!")
        print(f"   Path: {gdrive_best_path}")
    elif os.path.exists(local_resume_path):
        checkpoint_path = local_resume_path
        print(f"📂 Loading local resume checkpoint")
        print(f"   Path: {local_resume_path}")
        print(f"   Note: Best model on Google Drive preferred for optimal results")
    elif os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(('.pt', '.pth'))]
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            print(f"📂 Loading fallback checkpoint: {latest_checkpoint}")
        else:
            checkpoint_path = None
    else:
        checkpoint_path = None
    
    if checkpoint_path:
        print(f"Loading model from: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("✅ Model loaded successfully!")
            print(f"   Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"   Best Dice: {checkpoint.get('best_dice', 'N/A')}")
        except Exception as e:
            print(f"⚠️  Warning: Could not load checkpoint: {e}")
            print("Using fresh model (results may be less meaningful)")
    else:
        print("⚠️  No checkpoints found, using fresh model")
    
    model.to(DEVICE)
    model.eval()
    
    return model

# ===========================================================================================
# CONFUSION MATRIX COMPUTATION
# ===========================================================================================

def compute_confusion_matrix(model, loader):
    """Compute confusion matrix from predictions"""
    print("\nGenerating predictions...")
    
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing batches"):
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            # Get center slice
            center_label = labels[:, :, labels.shape[2]//2, :, :]
            
            # Predict
            outputs = model(images)
            pred_binary = (outputs > 0.5).float()
            
            # Flatten and collect
            all_preds.append(pred_binary.cpu().numpy().flatten())
            all_labels.append(center_label.cpu().numpy().flatten())
    
    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    print(f"\nTotal pixels analyzed: {len(all_preds):,}")
    print(f"Positive pixels (lesions): {all_labels.sum():,} ({all_labels.sum()/len(all_labels)*100:.2f}%)")
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels.astype(int), all_preds.astype(int))
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        'dice_coefficient': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        'iou': tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    }
    
    return cm, metrics, all_preds, all_labels

# ===========================================================================================
# VISUALIZATION
# ===========================================================================================

def plot_confusion_matrices(cm, metrics):
    """Generate comprehensive confusion matrix visualizations"""
    print("\nGenerating visualizations...")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ===========================================================================================
    # 1. Raw Confusion Matrix (with counts)
    # ===========================================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Normalize for better visualization
    cm_display = cm.astype('float')
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                cbar_kws={'label': 'Count'}, square=True)
    ax1.set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_xticklabels(['Background (0)', 'Lesion (1)'])
    ax1.set_yticklabels(['Background (0)', 'Lesion (1)'])
    
    # ===========================================================================================
    # 2. Normalized Confusion Matrix (percentages)
    # ===========================================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax2,
                cbar_kws={'label': 'Percentage (%)'}, square=True, vmin=0, vmax=100)
    ax2.set_title('Confusion Matrix (Normalized by Row)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_xticklabels(['Background (0)', 'Lesion (1)'])
    ax2.set_yticklabels(['Background (0)', 'Lesion (1)'])
    
    # ===========================================================================================
    # 3. Class-wise Performance
    # ===========================================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    class_metrics = {
        'Background': {
            'Precision': metrics['specificity'],
            'Recall': metrics['true_negatives'] / (metrics['true_negatives'] + metrics['false_positives']),
            'F1-Score': 2 * metrics['specificity'] * metrics['true_negatives'] / 
                       (metrics['specificity'] + metrics['true_negatives'] + 1e-7)
        },
        'Lesion': {
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score']
        }
    }
    
    x = np.arange(2)
    width = 0.25
    
    precisions = [class_metrics['Background']['Precision'], class_metrics['Lesion']['Precision']]
    recalls = [class_metrics['Background']['Recall'], class_metrics['Lesion']['Recall']]
    f1s = [class_metrics['Background']['F1-Score'], class_metrics['Lesion']['F1-Score']]
    
    ax3.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    ax3.bar(x, recalls, width, label='Recall', alpha=0.8)
    ax3.bar(x + width, f1s, width, label='F1-Score', alpha=0.8)
    
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Class-wise Performance Metrics', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Background', 'Lesion'])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 1.1])
    
    # ===========================================================================================
    # 4. Overall Metrics Summary
    # ===========================================================================================
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    
    # Create text summary
    summary_text = f"""
    CONFUSION MATRIX SUMMARY
    {'=' * 80}
    
    Classification Counts:
      • True Negatives (TN):  {metrics['true_negatives']:,} pixels (correctly identified background)
      • False Positives (FP): {metrics['false_positives']:,} pixels (background wrongly classified as lesion)
      • False Negatives (FN): {metrics['false_negatives']:,} pixels (lesion wrongly classified as background)
      • True Positives (TP):  {metrics['true_positives']:,} pixels (correctly identified lesion)
    
    Performance Metrics:
      • Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%) - Overall correctness
      • Precision:       {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%) - Of predicted lesions, how many are correct
      • Recall (Sens.):  {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%) - Of actual lesions, how many detected
      • Specificity:     {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%) - Of actual background, how many correct
      • F1-Score:        {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%) - Harmonic mean of precision & recall
      • Dice Coeff:      {metrics['dice_coefficient']:.4f} ({metrics['dice_coefficient']*100:.2f}%) - Segmentation overlap
      • IoU:             {metrics['iou']:.4f} ({metrics['iou']*100:.2f}%) - Intersection over Union
    
    Error Analysis:
      • False Positive Rate: {metrics['false_positives']/(metrics['false_positives']+metrics['true_negatives'])*100:.2f}%
      • False Negative Rate: {metrics['false_negatives']/(metrics['false_negatives']+metrics['true_positives'])*100:.2f}%
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # ===========================================================================================
    # 5. Metric Comparison Bar Chart
    # ===========================================================================================
    ax5 = fig.add_subplot(gs[2, 0])
    
    metric_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'Dice', 'IoU']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['specificity'],
        metrics['f1_score'],
        metrics['dice_coefficient'],
        metrics['iou']
    ]
    
    colors = ['green' if v >= 0.85 else 'orange' if v >= 0.75 else 'red' for v in metric_values]
    
    bars = ax5.barh(metric_names, metric_values, color=colors, alpha=0.7)
    ax5.set_xlabel('Score', fontsize=12)
    ax5.set_title('Performance Metrics Overview', fontsize=14, fontweight='bold')
    ax5.set_xlim([0, 1])
    ax5.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, metric_values)):
        ax5.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10)
    
    # ===========================================================================================
    # 6. Confusion Matrix Components (TP, FP, TN, FN)
    # ===========================================================================================
    ax6 = fig.add_subplot(gs[2, 1])
    
    components = ['True\nNegatives', 'False\nPositives', 'False\nNegatives', 'True\nPositives']
    values = [metrics['true_negatives'], metrics['false_positives'], 
              metrics['false_negatives'], metrics['true_positives']]
    colors_comp = ['green', 'orange', 'red', 'blue']
    
    bars = ax6.bar(components, values, color=colors_comp, alpha=0.7)
    ax6.set_ylabel('Pixel Count', fontsize=12)
    ax6.set_title('Confusion Matrix Components', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    total = sum(values)
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,}\n({val/total*100:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    
    # ===========================================================================================
    # 7. ROC-style Performance Plot
    # ===========================================================================================
    ax7 = fig.add_subplot(gs[2, 2])
    
    # Plot TPR vs FPR point
    tpr = metrics['recall']  # True Positive Rate (Sensitivity)
    fpr = 1 - metrics['specificity']  # False Positive Rate
    
    # Plot diagonal (random classifier)
    ax7.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')
    
    # Plot perfect classifier
    ax7.plot([0, 0, 1], [0, 1, 1], 'g--', alpha=0.3, label='Perfect Classifier')
    
    # Plot our model
    ax7.scatter([fpr], [tpr], s=200, c='red', marker='o', 
               label=f'Our Model\n(TPR={tpr:.3f}, FPR={fpr:.3f})', zorder=5)
    
    ax7.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax7.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax7.set_title('ROC Space Visualization', fontsize=14, fontweight='bold')
    ax7.legend(loc='lower right')
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim([0, 1])
    ax7.set_ylim([0, 1])
    
    # Add annotations
    ax7.annotate(f'Distance from perfect: {np.sqrt(fpr**2 + (1-tpr)**2):.3f}',
                xy=(fpr, tpr), xytext=(0.5, 0.3),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Add overall title
    fig.suptitle('Comprehensive Confusion Matrix Analysis - MS Lesion Segmentation',
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    output_file = os.path.join(OUTPUT_DIR, 'confusion_matrix_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comprehensive visualization to: {output_file}")
    
    plt.close()

# ===========================================================================================
# MAIN EXECUTION
# ===========================================================================================

def main():
    print("\n" + "=" * 80)
    print("STARTING CONFUSION MATRIX GENERATION")
    print("=" * 80)
    
    # Load model
    print("\nStep 1: Loading trained model...")
    model = load_trained_model()
    
    # Load data
    print("\nStep 2: Loading validation data...")
    val_loader = load_validation_data()
    
    # Compute confusion matrix
    print("\nStep 3: Computing confusion matrix...")
    cm, metrics, all_preds, all_labels = compute_confusion_matrix(model, val_loader)
    
    # Generate visualizations
    print("\nStep 4: Generating visualizations...")
    plot_confusion_matrices(cm, metrics)
    
    # Save metrics to JSON
    metrics_file = os.path.join(OUTPUT_DIR, 'confusion_matrix_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved metrics to: {metrics_file}")
    
    # Generate text report
    report_file = os.path.join(OUTPUT_DIR, 'classification_report.txt')
    with open(report_file, 'w') as f:
        f.write("CONFUSION MATRIX ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. CONFUSION MATRIX (Raw Counts)\n")
        f.write("-" * 80 + "\n")
        f.write(f"                    Predicted Negative  Predicted Positive\n")
        f.write(f"True Negative       {metrics['true_negatives']:>18,}  {metrics['false_positives']:>18,}\n")
        f.write(f"True Positive       {metrics['false_negatives']:>18,}  {metrics['true_positives']:>18,}\n\n")
        
        f.write("2. PERFORMANCE METRICS\n")
        f.write("-" * 80 + "\n")
        for key, value in metrics.items():
            if key not in ['true_negatives', 'false_positives', 'false_negatives', 'true_positives']:
                f.write(f"{key:25s}: {value:.6f} ({value*100:.2f}%)\n")
        
        f.write("\n3. INTERPRETATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total pixels analyzed: {metrics['true_negatives'] + metrics['false_positives'] + metrics['false_negatives'] + metrics['true_positives']:,}\n")
        f.write(f"Class balance: {(metrics['true_positives'] + metrics['false_negatives']) / (metrics['true_negatives'] + metrics['false_positives'] + metrics['false_negatives'] + metrics['true_positives']) * 100:.2f}% lesion pixels\n")
        f.write(f"\nModel correctly classifies {metrics['accuracy']*100:.2f}% of all pixels.\n")
        f.write(f"Of predicted lesions, {metrics['precision']*100:.2f}% are actually lesions (Precision).\n")
        f.write(f"Of actual lesions, {metrics['recall']*100:.2f}% are detected (Recall/Sensitivity).\n")
        f.write(f"Of actual background, {metrics['specificity']*100:.2f}% are correctly classified (Specificity).\n")
    
    print(f"✅ Saved text report to: {report_file}")
    
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  1. {os.path.join(OUTPUT_DIR, 'confusion_matrix_analysis.png')}")
    print(f"  2. {os.path.join(OUTPUT_DIR, 'confusion_matrix_metrics.json')}")
    print(f"  3. {os.path.join(OUTPUT_DIR, 'classification_report.txt')}")
    print("\n✅ All confusion matrix visualizations generated successfully!")

if __name__ == "__main__":
    main()
