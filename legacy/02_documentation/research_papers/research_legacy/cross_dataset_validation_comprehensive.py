"""
Comprehensive Cross-Dataset Validation
=======================================
Evaluates the Final Adaptive Model (82.31% Dice on PediMS) on:
1. LGG Dataset (Brain Tumors - Low-Grade Glioma)
2. MSLesionSeg Dataset (Adult MS Lesions)

Purpose: Demonstrate cross-pathology and cross-demographic generalization
Training: PediMS (Pediatric MS, 45 patients)
Validation: LGG (Brain Tumors, 110 patients) + MSLesionSeg (Adult MS, ~50 patients)

Model: HybridMiniSwin2D5_CBAM with Adaptive Slice Selection
Checkpoint: OptimalModel_Evidential/deployment/model.pth
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import json
from datetime import datetime
from tqdm import tqdm
import csv
import warnings
import nibabel as nib
from pathlib import Path
from scipy.ndimage import zoom

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from final_model import HybridMiniSwin2D5_CBAM

# MONAI imports for MSLesionSeg
try:
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
        NormalizeIntensityd, Resized, EnsureTyped
    )
    from monai.data import Dataset as MonaiDataset
    MONAI_AVAILABLE = True
except ImportError:
    print("⚠️ MONAI not available - MSLesionSeg validation will be skipped")
    MONAI_AVAILABLE = False


# ============================================================================
# DATASET 1: LGG Brain Tumor Dataset
# ============================================================================

class LGGDataset(Dataset):
    """
    LGG Brain Tumor MRI Dataset Loader
    
    Dataset Info:
    - 110 patients with low-grade glioma
    - 256x256 resolution TIFF images
    - 3 channels: pre-contrast, FLAIR, post-contrast
    - Binary segmentation masks (0=background, 255=tumor)
    """
    
    def __init__(self, root_dir, target_size=(64, 64), k_slices=9):
        """
        Args:
            root_dir: Path to kaggle_3m folder
            target_size: Resize images to this size (model expects 64x64)
            k_slices: Number of slices for adaptive selection
        """
        self.root_dir = root_dir
        self.target_size = target_size
        self.k_slices = k_slices
        
        # Get all patient folders
        self.patient_folders = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('TCGA_')
        ])
        
        # Build list of all image-mask pairs with tumor
        self.samples = []
        print(f"Loading LGG dataset from: {root_dir}")
        print(f"Found {len(self.patient_folders)} patients")
        
        for patient_id in tqdm(self.patient_folders, desc="Scanning LGG patients"):
            patient_dir = os.path.join(root_dir, patient_id)
            
            # Get all image files
            image_files = sorted([
                f for f in os.listdir(patient_dir)
                if f.endswith('.tif') and '_mask' not in f
            ])
            
            # Group slices by patient
            patient_slices = []
            for img_file in image_files:
                mask_file = img_file.replace('.tif', '_mask.tif')
                mask_path = os.path.join(patient_dir, mask_file)
                
                # Check if mask has tumor (only include slices with tumors)
                if os.path.exists(mask_path):
                    mask = np.array(Image.open(mask_path))
                    if np.sum(mask > 0) > 50:  # At least 50 tumor pixels
                        patient_slices.append({
                            'image': os.path.join(patient_dir, img_file),
                            'mask': mask_path,
                            'patient_id': patient_id,
                            'slice_idx': len(patient_slices)
                        })
            
            # Add patient slices if they have tumors
            if len(patient_slices) > 0:
                self.samples.extend(patient_slices)
        
        print(f"\nLGG Dataset Summary:")
        print(f"  Total patients: {len(self.patient_folders)}")
        print(f"  Slices with tumors: {len(self.samples)}")
        print(f"  Target size: {target_size}")
        print(f"  K-slices for adaptive selection: {k_slices}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image (TIFF format, 3 channels)
        image = np.array(Image.open(sample['image']))
        
        # Resize image
        if len(image.shape) == 2:
            image = Image.fromarray(image).resize(self.target_size, Image.BILINEAR)
            image = np.array(image)
        else:
            # Multi-channel image
            resized_channels = []
            for c in range(image.shape[2]):
                channel = Image.fromarray(image[:, :, c]).resize(self.target_size, Image.BILINEAR)
                resized_channels.append(np.array(channel))
            image = np.stack(resized_channels, axis=2)
        
        # Use FLAIR channel (channel 1) as per LGG dataset description
        if len(image.shape) == 3:
            flair = image[:, :, 1] if image.shape[2] > 1 else image[:, :, 0]
        else:
            flair = image
        
        # Normalize to [0, 1]
        flair = flair.astype(np.float32)
        if flair.max() > 0:
            flair = flair / flair.max()
        
        # Create 3D volume by replicating (simulate multiple slices)
        # Since LGG is 2D, we create a pseudo-3D volume
        volume = np.stack([flair] * 64, axis=0)  # (64, 64, 64)
        
        # Add channel dimension -> (1, 64, 64, 64)
        volume = volume[np.newaxis, ...]
        
        # Load mask
        mask = np.array(Image.open(sample['mask']))
        mask = Image.fromarray(mask).resize(self.target_size, Image.NEAREST)
        mask = np.array(mask)
        
        # Normalize mask to [0, 1]
        mask = (mask > 127).astype(np.float32)
        
        # Convert to tensors
        volume = torch.from_numpy(volume).float()
        mask = torch.from_numpy(mask).float()
        
        return {
            'image': volume,
            'mask': mask,
            'patient_id': sample['patient_id'],
            'slice_idx': sample['slice_idx']
        }


# ============================================================================
# DATASET 2: MSLesionSeg Dataset (if MONAI available)
# ============================================================================

class MSLesionSegDataset(Dataset):
    """
    MSLesionSeg Dataset Loader
    
    Dataset Structure:
    - MSLesSeg Dataset/MSLesSeg Dataset/train/P1/T1/P1_T1_FLAIR.nii.gz
    - MSLesSeg Dataset/MSLesSeg Dataset/train/P1/T1/P1_T1_MASK.nii.gz
    - 53 patients (train), multiple timepoints (T1, T2, T3)
    """
    
    def __init__(self, root_dir, target_size=(64, 64, 64)):
        """
        Args:
            root_dir: Path to train or test folder
            target_size: Resize volumes to this size
        """
        self.root_dir = root_dir
        self.target_size = target_size
        self.samples = []
        
        print(f"Loading MSLesionSeg dataset from: {root_dir}")
        
        # Scan all patient folders
        if not os.path.exists(root_dir):
            print(f"⚠️ Directory not found: {root_dir}")
            return
        
        patient_folders = sorted([d for d in os.listdir(root_dir) if d.startswith('P')])
        
        for patient in tqdm(patient_folders, desc="Scanning MSLesionSeg patients"):
            patient_dir = os.path.join(root_dir, patient)
            
            # Check each timepoint
            for timepoint in ['T1', 'T2', 'T3']:
                timepoint_dir = os.path.join(patient_dir, timepoint)
                
                if not os.path.exists(timepoint_dir):
                    continue
                
                # Look for FLAIR image
                flair_file = f"{patient}_{timepoint}_FLAIR.nii.gz"
                flair_path = os.path.join(timepoint_dir, flair_file)
                
                # Look for mask (check both uppercase and lowercase)
                possible_mask_names = [
                    f"{patient}_{timepoint}_MASK.nii.gz",  # Uppercase (actual format)
                    f"{patient}_{timepoint}_mask.nii.gz",
                    f"{patient}_{timepoint}_lesion.nii.gz",
                    f"{patient}_{timepoint}_seg.nii.gz"
                ]
                
                mask_path = None
                for mask_name in possible_mask_names:
                    test_path = os.path.join(timepoint_dir, mask_name)
                    if os.path.exists(test_path):
                        mask_path = test_path
                        break
                
                # Check all files to find mask
                if mask_path is None:
                    all_files = os.listdir(timepoint_dir)
                    mask_files = [f for f in all_files if 'mask' in f.lower() or 'lesion' in f.lower() or 'seg' in f.lower()]
                    if mask_files:
                        mask_path = os.path.join(timepoint_dir, mask_files[0])
                
                if os.path.exists(flair_path) and mask_path and os.path.exists(mask_path):
                    # Check if mask has lesions
                    try:
                        mask_nii = nib.load(mask_path)
                        mask_data = mask_nii.get_fdata()
                        if np.sum(mask_data > 0) > 10:  # At least 10 voxels
                            self.samples.append({
                                'flair': flair_path,
                                'mask': mask_path,
                                'patient': patient,
                                'timepoint': timepoint
                            })
                    except:
                        continue
        
        print(f"\nMSLesionSeg Dataset Summary:")
        print(f"  Total patients scanned: {len(patient_folders)}")
        print(f"  Valid samples with lesions: {len(self.samples)}")
        print(f"  Target size: {target_size}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load FLAIR
        flair_nii = nib.load(sample['flair'])
        flair = flair_nii.get_fdata().astype(np.float32)
        
        # Load mask
        mask_nii = nib.load(sample['mask'])
        mask = mask_nii.get_fdata().astype(np.float32)
        
        # Normalize FLAIR
        if flair.max() > 0:
            flair = (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)
        
        # Binarize mask
        mask = (mask > 0).astype(np.float32)
        
        # Resize to target size
        zoom_factors = [
            self.target_size[0] / flair.shape[0],
            self.target_size[1] / flair.shape[1],
            self.target_size[2] / flair.shape[2]
        ]
        
        flair = zoom(flair, zoom_factors, order=1)
        mask = zoom(mask, zoom_factors, order=0)
        
        # Add channel dimension -> (1, D, H, W)
        flair = flair[np.newaxis, ...]
        
        # Convert to tensors
        flair = torch.from_numpy(flair).float()
        mask = torch.from_numpy(mask).float()
        
        return {
            'image': flair,
            'mask': mask,
            'case': f"{sample['patient']}_{sample['timepoint']}"
        }


# ============================================================================
# METRICS
# ============================================================================

def dice_score(pred, target, smooth=1e-5):
    """Calculate Dice coefficient"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()


def precision_score(pred, target, smooth=1e-5):
    """Calculate Precision"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    
    prec = (tp + smooth) / (tp + fp + smooth)
    return prec.item()


def recall_score(pred, target, smooth=1e-5):
    """Calculate Recall"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    
    rec = (tp + smooth) / (tp + fn + smooth)
    return rec.item()


def f1_score(pred, target):
    """Calculate F1 score"""
    prec = precision_score(pred, target)
    rec = recall_score(pred, target)
    
    f1 = 2 * (prec * rec) / (prec + rec + 1e-5)
    return f1


def specificity_score(pred, target, smooth=1e-5):
    """Calculate Specificity"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    
    spec = (tn + smooth) / (tn + fp + smooth)
    return spec.item()


def iou_score(pred, target, smooth=1e-5):
    """Calculate IoU (Jaccard Index)"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


# ============================================================================
# VALIDATOR CLASS
# ============================================================================

class CrossDatasetValidator:
    """Manages cross-dataset validation on multiple datasets"""
    
    def __init__(self, model_path, output_dir, device='cuda'):
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = device
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize logging
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(output_dir, f'cross_validation_log_{self.timestamp}.txt')
        
        self.log("="*80)
        self.log("COMPREHENSIVE CROSS-DATASET VALIDATION")
        self.log("="*80)
        self.log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Model: HybridMiniSwin2D5_CBAM with Adaptive Slice Selection")
        self.log(f"Training Dataset: PediMS (Pediatric MS, 45 patients)")
        self.log(f"Training Performance: 82.31% Dice (Best epoch 12/32)")
        self.log(f"Validation Datasets:")
        self.log(f"  1. LGG (Brain Tumors - Low-Grade Glioma, 110 patients)")
        self.log(f"  2. MSLesionSeg (Adult MS Lesions, ~50 patients)")
        self.log("="*80)
        
        # Load model
        self.model = self.load_model()
    
    def log(self, message):
        """Log to file and console"""
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    
    def load_model(self):
        """Load trained model"""
        self.log("\n" + "="*80)
        self.log("LOADING MODEL")
        self.log("="*80)
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model configuration
        model_config = checkpoint.get('model_config', {})
        k_slices = model_config.get('k_slices', 9)
        channels = model_config.get('channels', [32, 64, 128, 256, 512])
        window_size = model_config.get('window_size', 4)
        
        self.log(f"Model Configuration:")
        self.log(f"  k_slices: {k_slices}")
        self.log(f"  channels: {channels}")
        self.log(f"  window_size: {window_size}")
        
        # Initialize model
        model = HybridMiniSwin2D5_CBAM(
            k_slices=k_slices,
            channels=channels,
            use_adaptive_selection=True
        ).to(self.device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Get training info
        training_info = checkpoint.get('training_info', {})
        val_metrics = training_info.get('val_metrics', {})
        
        self.log(f"\nTraining Performance (PediMS validation):")
        self.log(f"  Dice: {val_metrics.get('dice', 0)*100:.2f}%")
        self.log(f"  Precision: {val_metrics.get('precision', 0)*100:.2f}%")
        self.log(f"  Recall: {val_metrics.get('recall', 0)*100:.2f}%")
        self.log(f"  F1: {val_metrics.get('f1', 0)*100:.2f}%")
        self.log(f"  Best Epoch: {training_info.get('final_epoch', 'N/A')}")
        
        self.log("\n✅ Model loaded successfully")
        
        return model
    
    def validate_dataset(self, dataset, dataset_name, batch_size=1):
        """Validate on a specific dataset"""
        self.log("\n" + "="*80)
        self.log(f"VALIDATING ON: {dataset_name}")
        self.log("="*80)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        results = []
        all_metrics = {
            'dice': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'specificity': [],
            'iou': []
        }
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
                try:
                    # Get data
                    if isinstance(batch, dict):
                        images = batch['image'].to(self.device)
                        masks = batch['mask'].to(self.device)
                        case_id = batch.get('patient_id', batch.get('case', ['unknown']))[0]
                    else:
                        images, masks = batch
                        images = images.to(self.device)
                        masks = masks.to(self.device)
                        case_id = 'unknown'
                    
                    # Forward pass
                    output = self.model(images)
                    
                    # Handle dict output (evidential model)
                    if isinstance(output, dict):
                        preds = torch.sigmoid(output['probs'])
                    else:
                        preds = torch.sigmoid(output)
                    
                    # Calculate metrics
                    dice = dice_score(preds, masks)
                    prec = precision_score(preds, masks)
                    rec = recall_score(preds, masks)
                    f1 = f1_score(preds, masks)
                    spec = specificity_score(preds, masks)
                    iou = iou_score(preds, masks)
                    
                    # Store results
                    results.append({
                        'case': case_id,
                        'dice': dice,
                        'precision': prec,
                        'recall': rec,
                        'f1': f1,
                        'specificity': spec,
                        'iou': iou
                    })
                    
                    for key in all_metrics.keys():
                        all_metrics[key].append(results[-1][key])
                
                except Exception as e:
                    self.log(f"⚠️ Error processing case: {e}")
                    continue
        
        # Calculate average metrics
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        std_metrics = {k: np.std(v) for k, v in all_metrics.items()}
        
        # Log results
        self.log(f"\n{dataset_name} Results:")
        self.log(f"  Samples evaluated: {len(results)}")
        self.log(f"\nAverage Metrics:")
        self.log(f"  Dice:        {avg_metrics['dice']*100:.2f}% ± {std_metrics['dice']*100:.2f}%")
        self.log(f"  Precision:   {avg_metrics['precision']*100:.2f}% ± {std_metrics['precision']*100:.2f}%")
        self.log(f"  Recall:      {avg_metrics['recall']*100:.2f}% ± {std_metrics['recall']*100:.2f}%")
        self.log(f"  F1:          {avg_metrics['f1']*100:.2f}% ± {std_metrics['f1']*100:.2f}%")
        self.log(f"  Specificity: {avg_metrics['specificity']*100:.2f}% ± {std_metrics['specificity']*100:.2f}%")
        self.log(f"  IoU:         {avg_metrics['iou']*100:.2f}% ± {std_metrics['iou']*100:.2f}%")
        
        # Save per-sample results
        csv_path = os.path.join(self.output_dir, f'{dataset_name.lower().replace(" ", "_")}_results_{self.timestamp}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['case', 'dice', 'precision', 'recall', 'f1', 'specificity', 'iou'])
            writer.writeheader()
            writer.writerows(results)
        
        self.log(f"\n✅ Per-sample results saved to: {csv_path}")
        
        return avg_metrics, std_metrics, results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Paths
    MODEL_PATH = r"C:\Users\HP\EDI\OptimalModel_Evidential\deployment\model.pth"
    LGG_ROOT = r"C:\Users\HP\EDI\LGG\kaggle_3m"
    MSLESSEG_ROOT = r"C:\Users\HP\EDI\Dataset\MSLESSEG\MSLesSeg Dataset\MSLesSeg Dataset\train"
    OUTPUT_DIR = r"C:\Users\HP\EDI\csv_data\cross_dataset_validation"
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n🚀 Starting Cross-Dataset Validation")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}\n")
    
    # Initialize validator
    validator = CrossDatasetValidator(MODEL_PATH, OUTPUT_DIR, DEVICE)
    
    all_results = {}
    
    # ========== VALIDATE ON LGG ==========
    validator.log("\n" + "="*80)
    validator.log("PREPARING LGG DATASET")
    validator.log("="*80)
    
    try:
        lgg_dataset = LGGDataset(
            root_dir=LGG_ROOT,
            target_size=(64, 64),
            k_slices=9
        )
        
        lgg_avg, lgg_std, lgg_results = validator.validate_dataset(
            lgg_dataset,
            "LGG Brain Tumors",
            batch_size=1
        )
        
        all_results['LGG'] = {
            'avg': lgg_avg,
            'std': lgg_std,
            'samples': len(lgg_results)
        }
    
    except Exception as e:
        validator.log(f"\n❌ LGG validation failed: {e}")
        all_results['LGG'] = None
    
    # ========== VALIDATE ON MSLesionSeg ==========
    validator.log("\n" + "="*80)
    validator.log("PREPARING MSLesionSeg DATASET")
    validator.log("="*80)
    
    try:
        mslesseg_dataset = MSLesionSegDataset(
            root_dir=MSLESSEG_ROOT,
            target_size=(64, 64, 64)
        )
        
        if len(mslesseg_dataset) > 0:
            mslesseg_avg, mslesseg_std, mslesseg_results = validator.validate_dataset(
                mslesseg_dataset,
                "MSLesionSeg Adult MS",
                batch_size=1
            )
            
            all_results['MSLesionSeg'] = {
                'avg': mslesseg_avg,
                'std': mslesseg_std,
                'samples': len(mslesseg_results)
            }
        else:
            validator.log("⚠️ No valid MSLesionSeg samples found")
            all_results['MSLesionSeg'] = None
    
    except Exception as e:
        import traceback
        validator.log(f"\n❌ MSLesionSeg validation failed: {e}")
        validator.log(traceback.format_exc())
        all_results['MSLesionSeg'] = None
    
    # ========== SUMMARY ==========
    validator.log("\n" + "="*80)
    validator.log("FINAL SUMMARY")
    validator.log("="*80)
    
    validator.log(f"\nTraining Dataset: PediMS (Pediatric MS)")
    validator.log(f"  Performance: 82.31% Dice")
    
    for dataset_name, result in all_results.items():
        if result is not None:
            validator.log(f"\n{dataset_name} (Zero-shot):")
            validator.log(f"  Samples: {result['samples']}")
            validator.log(f"  Dice: {result['avg']['dice']*100:.2f}% ± {result['std']['dice']*100:.2f}%")
            
            # Calculate generalization gap
            gap = (0.8231 - result['avg']['dice']) * 100
            validator.log(f"  Generalization Gap: {gap:.2f}%")
        else:
            validator.log(f"\n{dataset_name}: Not evaluated")
    
    # Save summary JSON
    summary_path = os.path.join(OUTPUT_DIR, f'summary_{validator.timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': validator.timestamp,
            'model_path': MODEL_PATH,
            'training_performance': 0.8231,
            'results': {
                k: {
                    'samples': v['samples'],
                    'dice_mean': float(v['avg']['dice']),
                    'dice_std': float(v['std']['dice']),
                    'precision_mean': float(v['avg']['precision']),
                    'recall_mean': float(v['avg']['recall']),
                    'f1_mean': float(v['avg']['f1'])
                } if v is not None else None
                for k, v in all_results.items()
            }
        }, f, indent=2)
    
    validator.log(f"\n✅ Summary saved to: {summary_path}")
    validator.log("\n" + "="*80)
    validator.log("CROSS-DATASET VALIDATION COMPLETE")
    validator.log("="*80)


if __name__ == "__main__":
    main()
