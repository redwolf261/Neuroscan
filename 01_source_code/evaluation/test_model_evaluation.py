"""
Test Model Evaluation on Test Dataset
Evaluates the model from app.py on the PediMS test dataset
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import glob
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Import model architecture
from final_model import HybridMiniSwin2D5_CBAM, SPATIAL_SIZE, K_SLICES, STAGE_CHANNELS

# MONAI imports - specific submodules
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.utility.dictionary import EnsureChannelFirstd
from monai.transforms.spatial.dictionary import Orientationd, Spacingd, Resized
from monai.transforms.intensity.dictionary import NormalizeIntensityd
from monai.transforms.croppad.dictionary import CenterSpatialCropd
from monai.transforms.transform import MapTransform
from monai.data.dataset import Dataset
from monai.data.dataloader import DataLoader
from monai.metrics.meandice import DiceMetric
from monai.inferers.utils import sliding_window_inference

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = r"G:\My Drive\OptimalModel_FrozenSelector_20251218_135038\deployment\model.pth"
BATCH_SIZE = 1  # For testing
NUM_WORKERS = 2

# Dataset path
DATA_BASE = r"C:\Users\HP\EDI"
DATA_PATH = os.path.join(DATA_BASE, "Dataset", "PediMS", "PediMS")

print("=" * 80)
print("MODEL EVALUATION ON TEST DATASET")
print("=" * 80)
print(f"Device: {DEVICE}")
print(f"Model Path: {MODEL_PATH}")
print(f"Dataset Path: {DATA_PATH}")
print(f"Spatial Size: {SPATIAL_SIZE}")
print(f"K-Slices: {K_SLICES}")
print(f"Stage Channels: {STAGE_CHANNELS}")
print("=" * 80)

# ============================================================================
# PREPROCESSING TRANSFORMS
# ============================================================================

class BinarizeLabel(MapTransform):
    """Convert label to binary (0 or 1)"""
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        d["label"] = (d["label"] > 0).float()
        return d

class Extract2D5Slices(MapTransform):
    """Extract k neighboring slices for 2.5D processing"""
    def __init__(self, keys: List[str], k_slices: int = K_SLICES) -> None:
        super().__init__(keys)
        self.k_slices = k_slices
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.keys:
            volume = d[key]  # Shape: (C, D, H, W)
            
            # Get center slice
            depth = volume.shape[1]
            center_slice = depth // 2
            
            # Calculate slice range
            half_k = self.k_slices // 2
            start_slice = max(0, center_slice - half_k)
            end_slice = min(depth, center_slice + half_k + 1)
            
            # Extract slices
            slices = volume[:, start_slice:end_slice, :, :]
            
            # Pad if needed
            if slices.shape[1] < self.k_slices:
                pad_before = (self.k_slices - slices.shape[1]) // 2
                pad_after = self.k_slices - slices.shape[1] - pad_before
                slices = torch.nn.functional.pad(
                    slices, 
                    (0, 0, 0, 0, pad_before, pad_after), 
                    mode='replicate'
                )
            
            d[key] = slices
        return d

def load_test_data():
    """Load test dataset (validation split)"""
    print("\nLoading test dataset...")
    
    if not os.path.exists(DATA_PATH):
        raise RuntimeError(f"Dataset not found at {DATA_PATH}")
    
    # Collect all data
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
            
            for img, mask in zip(imgs, masks):
                data_dicts.append({"image": [img], "label": mask})
    
    print(f"Total samples loaded: {len(data_dicts)}")
    
    # Use validation split (last 20% as test set)
    split_idx = int(0.8 * len(data_dicts))
    test_files = data_dicts[split_idx:]
    
    print(f"Test samples: {len(test_files)}")
    
    # Transforms
    test_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        Resized(keys=["image", "label"], spatial_size=SPATIAL_SIZE, mode=("trilinear", "nearest")),
        BinarizeLabel(keys=["label"]),
        Extract2D5Slices(keys=["image", "label"], k_slices=K_SLICES),
    ])
    
    # Create dataset and loader
    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return test_loader, len(test_files)

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():
    """Load trained model"""
    print("\nLoading model...")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    # Initialize model
    model = HybridMiniSwin2D5_CBAM(
        k_slices=K_SLICES,
        channels=STAGE_CHANNELS,
        use_adaptive_selection=True
    )
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(DEVICE)
    model.eval()
    
    # Print model info
    if 'training_info' in checkpoint:
        training_info = checkpoint['training_info']
        val_metrics = training_info.get('val_metrics', {})
        print(f"✓ Model loaded successfully")
        print(f"  Epoch: {training_info.get('final_epoch', 'N/A')}")
        print(f"  Val Dice: {val_metrics.get('dice', 'N/A'):.4f}")
        print(f"  Val Precision: {val_metrics.get('precision', 'N/A'):.4f}")
        print(f"  Val Recall: {val_metrics.get('recall', 'N/A'):.4f}")
        print(f"  Val F1: {val_metrics.get('f1', 'N/A'):.4f}")
    else:
        print(f"✓ Model loaded successfully")
    
    return model

# ============================================================================
# EVALUATION
# ============================================================================

def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, Any]:
    """Calculate detailed metrics"""
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0.5).float()
    
    # True Positives, False Positives, False Negatives, True Negatives
    tp = ((pred_binary == 1) & (target_binary == 1)).sum().item()
    fp = ((pred_binary == 1) & (target_binary == 0)).sum().item()
    fn = ((pred_binary == 0) & (target_binary == 1)).sum().item()
    tn = ((pred_binary == 0) & (target_binary == 0)).sum().item()
    
    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Dice coefficient
    intersection = (pred_binary * target_binary).sum().item()
    dice = (2.0 * intersection) / (pred_binary.sum().item() + target_binary.sum().item()) if (pred_binary.sum().item() + target_binary.sum().item()) > 0 else 0.0
    
    # IoU (Jaccard)
    union = pred_binary.sum().item() + target_binary.sum().item() - intersection
    iou = intersection / union if union > 0 else 0.0
    
    return {
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

def evaluate_model(model: nn.Module, test_loader: DataLoader) -> Dict[str, Any]:
    """Evaluate model on test set"""
    print("\nEvaluating model on test set...")
    
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Testing"):
            inputs = batch_data["image"].to(DEVICE)
            labels = batch_data["label"].to(DEVICE)
            
            # Forward pass
            outputs = model(inputs)
            
            # Apply sigmoid
            outputs = torch.sigmoid(outputs)
            
            # Calculate metrics for this batch
            for i in range(outputs.shape[0]):
                metrics = calculate_metrics(outputs[i], labels[i])
                all_metrics.append(metrics)
    
    # Aggregate metrics
    avg_metrics = {
        'dice': np.mean([m['dice'] for m in all_metrics]),
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics]),
        'f1': np.mean([m['f1'] for m in all_metrics]),
        'iou': np.mean([m['iou'] for m in all_metrics]),
    }
    
    # Standard deviations
    std_metrics = {
        'dice_std': np.std([m['dice'] for m in all_metrics]),
        'precision_std': np.std([m['precision'] for m in all_metrics]),
        'recall_std': np.std([m['recall'] for m in all_metrics]),
        'f1_std': np.std([m['f1'] for m in all_metrics]),
        'iou_std': np.std([m['iou'] for m in all_metrics]),
    }
    
    # Total confusion matrix
    total_tp = sum([m['tp'] for m in all_metrics])
    total_fp = sum([m['fp'] for m in all_metrics])
    total_fn = sum([m['fn'] for m in all_metrics])
    total_tn = sum([m['tn'] for m in all_metrics])
    
    return avg_metrics, std_metrics, all_metrics, (total_tp, total_fp, total_fn, total_tn)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    try:
        # Load data
        test_loader, num_samples = load_test_data()
        
        # Load model
        model = load_model()
        
        # Evaluate
        avg_metrics, std_metrics, all_metrics, confusion = evaluate_model(model, test_loader)
        
        # Print results
        print("\n" + "=" * 80)
        print("TEST RESULTS")
        print("=" * 80)
        print(f"Number of test samples: {num_samples}")
        print("\nAverage Metrics:")
        print(f"  Dice Score:    {avg_metrics['dice']:.4f} ± {std_metrics['dice_std']:.4f}")
        print(f"  Precision:     {avg_metrics['precision']:.4f} ± {std_metrics['precision_std']:.4f}")
        print(f"  Recall:        {avg_metrics['recall']:.4f} ± {std_metrics['recall_std']:.4f}")
        print(f"  F1 Score:      {avg_metrics['f1']:.4f} ± {std_metrics['f1_std']:.4f}")
        print(f"  IoU (Jaccard): {avg_metrics['iou']:.4f} ± {std_metrics['iou_std']:.4f}")
        
        print("\nConfusion Matrix (Total):")
        tp, fp, fn, tn = confusion
        print(f"  True Positives:  {tp:,}")
        print(f"  False Positives: {fp:,}")
        print(f"  False Negatives: {fn:,}")
        print(f"  True Negatives:  {tn:,}")
        
        print("\nPer-Sample Statistics:")
        print(f"  Best Dice:     {max([m['dice'] for m in all_metrics]):.4f}")
        print(f"  Worst Dice:    {min([m['dice'] for m in all_metrics]):.4f}")
        print(f"  Median Dice:   {np.median([m['dice'] for m in all_metrics]):.4f}")
        
        # Save results to file
        output_file = os.path.join(SCRIPT_DIR, "test_results.txt")
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL EVALUATION ON TEST DATASET\n")
            f.write("=" * 80 + "\n")
            f.write(f"Model Path: {MODEL_PATH}\n")
            f.write(f"Dataset Path: {DATA_PATH}\n")
            f.write(f"Number of test samples: {num_samples}\n")
            f.write("\nAverage Metrics:\n")
            f.write(f"  Dice Score:    {avg_metrics['dice']:.4f} ± {std_metrics['dice_std']:.4f}\n")
            f.write(f"  Precision:     {avg_metrics['precision']:.4f} ± {std_metrics['precision_std']:.4f}\n")
            f.write(f"  Recall:        {avg_metrics['recall']:.4f} ± {std_metrics['recall_std']:.4f}\n")
            f.write(f"  F1 Score:      {avg_metrics['f1']:.4f} ± {std_metrics['f1_std']:.4f}\n")
            f.write(f"  IoU (Jaccard): {avg_metrics['iou']:.4f} ± {std_metrics['iou_std']:.4f}\n")
            f.write("\nConfusion Matrix (Total):\n")
            f.write(f"  True Positives:  {tp:,}\n")
            f.write(f"  False Positives: {fp:,}\n")
            f.write(f"  False Negatives: {fn:,}\n")
            f.write(f"  True Negatives:  {tn:,}\n")
            f.write("\nPer-Sample Statistics:\n")
            f.write(f"  Best Dice:     {max([m['dice'] for m in all_metrics]):.4f}\n")
            f.write(f"  Worst Dice:    {min([m['dice'] for m in all_metrics]):.4f}\n")
            f.write(f"  Median Dice:   {np.median([m['dice'] for m in all_metrics]):.4f}\n")
        
        print(f"\n✓ Results saved to: {output_file}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
