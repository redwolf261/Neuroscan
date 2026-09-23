"""
Few-Shot Fine-Tuning on MS60 Dataset
====================================
Fine-tune the model on a small subset of MS60 to adapt to the domain.
This is a standard technique for cross-dataset adaptation.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))
from final_model import HybridMiniSwin2D5_CSRF

# Paths
DATA_DIR = Path(r"C:\Users\HP\EDI\MS cross validation")
MODEL_PATH = Path(r"C:\Users\HP\EDI\.resume_checkpoints\seg_resume.pth")
OUTPUT_PATH = Path(r"C:\Users\HP\EDI\.resume_checkpoints\seg_resume_ms60_fewshot.pth")

# Hyperparameters
NUM_TRAIN_PATIENTS = 5  # Use 5 patients for few-shot training
NUM_VAL_PATIENTS = 2    # Use 2 for validation
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5  # Very small LR for fine-tuning
BATCH_SIZE = 2

print("="*80)
print("FEW-SHOT FINE-TUNING ON MS60")
print("="*80)
print(f"Train patients: {NUM_TRAIN_PATIENTS}")
print(f"Val patients: {NUM_VAL_PATIENTS}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print("="*80)
print()


class MS60FewShotDataset(torch.utils.data.Dataset):
    """Few-shot dataset for MS60"""
    
    def __init__(self, data_dir, patient_indices, mode='train'):
        self.data_dir = Path(data_dir)
        self.samples = []
        self.mode = mode
        
        patient_dirs = sorted([d for d in self.data_dir.iterdir() 
                              if d.is_dir() and d.name.startswith('Patient-')])
        
        # Select specific patients
        selected_patients = [patient_dirs[i] for i in patient_indices if i < len(patient_dirs)]
        
        print(f"Loading {mode} data from patients: {[p.name for p in selected_patients]}")
        
        for patient_dir in selected_patients:
            patient_id = patient_dir.name
            patient_num = patient_id.split('-')[1]
            
            flair_path = patient_dir / f"{patient_num}-Flair.nii"
            mask_path = patient_dir / f"{patient_num}-LesionSeg-Flair.nii"
            
            if not flair_path.exists() or not mask_path.exists():
                continue
            
            try:
                flair_data = nib.load(flair_path).get_fdata()
                mask_data = nib.load(mask_path).get_fdata()
                
                # Get ALL slices with lesions
                for slice_idx in range(mask_data.shape[2]):
                    lesion_count = mask_data[:, :, slice_idx].sum()
                    if lesion_count > 0:
                        self.samples.append({
                            'patient_id': patient_id,
                            'flair_path': flair_path,
                            'mask_path': mask_path,
                            'slice_idx': slice_idx,
                        })
            except Exception as e:
                print(f"Error loading {patient_id}: {e}")
                continue
        
        print(f"  Loaded {len(self.samples)} slices")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        flair_data = nib.load(sample['flair_path']).get_fdata()
        mask_data = nib.load(sample['mask_path']).get_fdata()
        
        slice_idx = sample['slice_idx']
        
        # Extract 2.5D volume
        k = 5
        start_slice = max(0, slice_idx - k//2)
        end_slice = min(flair_data.shape[2], slice_idx + k//2 + 1)
        
        volume = flair_data[:, :, start_slice:end_slice]
        mask = mask_data[:, :, slice_idx]
        
        if volume.shape[2] < k:
            pad_before = (k - volume.shape[2]) // 2
            pad_after = k - volume.shape[2] - pad_before
            volume = np.pad(volume, ((0,0), (0,0), (pad_before, pad_after)), mode='edge')
        
        volume = (volume - volume.mean()) / (volume.std() + 1e-8)
        
        volume_tensor = torch.from_numpy(volume).float().unsqueeze(0)
        volume_tensor = volume_tensor.permute(0, 3, 1, 2)
        
        # Resize
        target_size = (240, 240)
        volume_tensor = F.interpolate(volume_tensor, size=target_size, mode='bilinear', align_corners=False)
        
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        mask_tensor = F.interpolate(mask_tensor, size=target_size, mode='nearest')
        mask_tensor = mask_tensor.squeeze()
        
        return volume_tensor, mask_tensor


def dice_loss(pred, target):
    """Dice loss for training"""
    smooth = 1e-6
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return 1 - dice


def compute_metrics(pred, target):
    """Compute validation metrics"""
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0.5).float()
    
    tp = (pred_binary * target_binary).sum().item()
    fp = (pred_binary * (1 - target_binary)).sum().item()
    fn = ((1 - pred_binary) * target_binary).sum().item()
    
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    return {'dice': dice, 'precision': precision, 'recall': recall}


def train():
    """Few-shot training"""
    
    device = torch.device('cuda')
    
    # Load pretrained model
    print("Loading pretrained model...")
    model = HybridMiniSwin2D5_CSRF(channels=[32, 64, 128, 256, 512]).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  PediMS best Dice: {checkpoint.get('best_dice', 'unknown')}")
    print()
    
    # Prepare datasets - use first 5 patients for train, next 2 for val
    train_indices = list(range(NUM_TRAIN_PATIENTS))
    val_indices = list(range(NUM_TRAIN_PATIENTS, NUM_TRAIN_PATIENTS + NUM_VAL_PATIENTS))
    
    train_dataset = MS60FewShotDataset(DATA_DIR, train_indices, mode='train')
    val_dataset = MS60FewShotDataset(DATA_DIR, val_indices, mode='val')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                               shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, 
                                             shuffle=False, num_workers=0)
    
    # Optimizer - only fine-tune with very small LR
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print()
    print("="*80)
    print("STARTING FEW-SHOT FINE-TUNING")
    print("="*80)
    
    best_val_dice = 0.0
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for volumes, masks in pbar:
            volumes = volumes.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(volumes)
            outputs = outputs.squeeze(1)
            
            loss = dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{np.mean(train_losses):.4f}"})
        
        # Validation
        model.eval()
        val_metrics = []
        
        with torch.no_grad():
            for volumes, masks in val_loader:
                volumes = volumes.to(device)
                masks = masks.to(device)
                
                outputs = model(volumes)
                preds = outputs.squeeze()
                
                metrics = compute_metrics(preds, masks.squeeze())
                val_metrics.append(metrics)
        
        avg_dice = np.mean([m['dice'] for m in val_metrics])
        avg_precision = np.mean([m['precision'] for m in val_metrics])
        avg_recall = np.mean([m['recall'] for m in val_metrics])
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {np.mean(train_losses):.4f}")
        print(f"  Val Dice:   {avg_dice:.4f} ({avg_dice*100:.2f}%)")
        print(f"  Val Prec:   {avg_precision:.4f}")
        print(f"  Val Recall: {avg_recall:.4f}")
        
        # Save best model
        if avg_dice > best_val_dice:
            best_val_dice = avg_dice
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_val_dice,
                'train_loss': np.mean(train_losses),
            }, OUTPUT_PATH)
            print(f"  ✓ Saved best model (Dice: {best_val_dice:.4f})")
    
    print()
    print("="*80)
    print("FEW-SHOT FINE-TUNING COMPLETE")
    print("="*80)
    print(f"Best validation Dice: {best_val_dice:.4f} ({best_val_dice*100:.2f}%)")
    print(f"Model saved to: {OUTPUT_PATH}")
    print()
    print("💡 Now run cross-validation with the fine-tuned model:")
    print(f"   Use model: {OUTPUT_PATH}")
    print("="*80)


if __name__ == "__main__":
    train()
