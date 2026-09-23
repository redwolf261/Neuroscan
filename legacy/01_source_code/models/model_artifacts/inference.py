"""
Production Inference Module for MS Lesion Segmentation
Provides clean API for loading model and making predictions
"""

import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Union, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Import model architecture from final_model.py
from final_model import (
    HybridMiniSwin2D5_ResNet_CSRF,
    SPATIAL_SIZE
)

class MSLesionPredictor:
    """
    Production-ready predictor for MS lesion segmentation
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'auto',
        spatial_size: Tuple[int, int, int] = SPATIAL_SIZE
    ):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to best_model.pth checkpoint
            device: 'cuda', 'cpu', or 'auto' (auto-detect)
            spatial_size: Input volume size (D, H, W)
        """
        self.spatial_size = spatial_size
        
        # Auto-detect device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"✅ Model loaded successfully from {model_path}")
        print(f"📐 Input size: {spatial_size}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained model from checkpoint"""
        # Initialize model architecture
        model = HybridMiniSwin2D5_ResNet_CSRF(
            in_channels=3,
            num_classes=1,
            k=5,
            channels=[32, 64, 128, 256, 512],
            embed_dim=96,
            num_heads=4,
            mlp_ratio=4.0,
            drop_path=0.1
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Print model info
        if 'epoch' in checkpoint:
            print(f"📊 Trained for {checkpoint['epoch']} epochs")
        if 'val_metrics' in checkpoint:
            metrics = checkpoint['val_metrics']
            print(f"📈 Validation Dice: {metrics.get('dice', 'N/A'):.4f}")
            print(f"📈 Precision: {metrics.get('precision', 'N/A'):.4f}")
            print(f"📈 Recall: {metrics.get('recall', 'N/A'):.4f}")
        
        return model
    
    def preprocess(
        self,
        flair: Union[str, np.ndarray],
        t1: Union[str, np.ndarray],
        t2: Union[str, np.ndarray]
    ) -> torch.Tensor:
        """
        Preprocess input MRI volumes
        
        Args:
            flair: FLAIR volume (path or numpy array)
            t1: T1 volume (path or numpy array)
            t2: T2 volume (path or numpy array)
            
        Returns:
            Preprocessed tensor of shape (1, 3, D, H, W)
        """
        # Load volumes if paths are provided
        volumes = []
        for vol in [flair, t1, t2]:
            if isinstance(vol, str):
                nii = nib.load(vol)
                vol_data = nii.get_fdata()
            else:
                vol_data = vol
            volumes.append(vol_data)
        
        flair_data, t1_data, t2_data = volumes
        
        # Ensure same shape
        assert flair_data.shape == t1_data.shape == t2_data.shape, \
            f"All volumes must have same shape. Got FLAIR: {flair_data.shape}, T1: {t1_data.shape}, T2: {t2_data.shape}"
        
        # Normalize each modality independently (z-score)
        def normalize(volume):
            volume = volume.astype(np.float32)
            mean = volume.mean()
            std = volume.std()
            if std > 0:
                volume = (volume - mean) / std
            return volume
        
        flair_norm = normalize(flair_data)
        t1_norm = normalize(t1_data)
        t2_norm = normalize(t2_data)
        
        # Stack modalities: (D, H, W) -> (3, D, H, W)
        volume_3ch = np.stack([flair_norm, t1_norm, t2_norm], axis=0)
        
        # Resize to target spatial size if needed
        current_shape = volume_3ch.shape[1:]  # (D, H, W)
        if current_shape != self.spatial_size:
            print(f"⚙️  Resizing from {current_shape} to {self.spatial_size}")
            volume_3ch = self._resize_volume(volume_3ch, self.spatial_size)
        
        # Convert to tensor and add batch dimension: (1, 3, D, H, W)
        volume_tensor = torch.from_numpy(volume_3ch).unsqueeze(0).float()
        
        return volume_tensor
    
    def _resize_volume(self, volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """Resize 4D volume (C, D, H, W) to target size"""
        import torch.nn.functional as F
        
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).float()  # (1, C, D, H, W)
        resized = F.interpolate(
            volume_tensor,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        return resized.squeeze(0).numpy()
    
    @torch.no_grad()
    def predict(
        self,
        flair: Union[str, np.ndarray],
        t1: Union[str, np.ndarray],
        t2: Union[str, np.ndarray],
        threshold: float = 0.5,
        return_probability: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Predict MS lesion segmentation
        
        Args:
            flair: FLAIR volume (path or numpy array)
            t1: T1 volume (path or numpy array)
            t2: T2 volume (path or numpy array)
            threshold: Probability threshold for binary segmentation (default: 0.5)
            return_probability: If True, also return probability map
            
        Returns:
            Dictionary with:
                - 'segmentation': Binary mask (0 or 1)
                - 'probability': Probability map (if return_probability=True)
                - 'lesion_volume_ml': Total lesion volume in mL
                - 'num_lesions': Estimated number of lesions
        """
        print("🔄 Preprocessing input volumes...")
        volume_tensor = self.preprocess(flair, t1, t2)
        volume_tensor = volume_tensor.to(self.device)
        
        print("🧠 Running inference...")
        # Forward pass
        output = self.model(volume_tensor)  # (1, 1, D, H, W)
        
        # Apply sigmoid to get probabilities
        probability = torch.sigmoid(output).cpu().numpy()[0, 0]  # (D, H, W)
        
        # Apply threshold for binary mask
        segmentation = (probability >= threshold).astype(np.uint8)
        
        # Calculate lesion statistics
        lesion_voxels = segmentation.sum()
        
        # Assuming 1mm isotropic voxels (adjust based on your data)
        # If voxel size is different, pass it as parameter
        voxel_volume_ml = 1.0 / 1000.0  # 1mm³ = 0.001 mL
        lesion_volume_ml = float(lesion_voxels * voxel_volume_ml)
        
        # Estimate number of lesions (simple connected component analysis)
        from scipy.ndimage import label
        labeled_array, num_lesions = label(segmentation)
        
        print(f"✅ Prediction complete!")
        print(f"📊 Lesion volume: {lesion_volume_ml:.2f} mL")
        print(f"📊 Number of lesions: {num_lesions}")
        
        result = {
            'segmentation': segmentation,
            'lesion_volume_ml': lesion_volume_ml,
            'num_lesions': int(num_lesions),
            'lesion_voxels': int(lesion_voxels)
        }
        
        if return_probability:
            result['probability'] = probability
        
        return result
    
    def save_prediction(
        self,
        segmentation: np.ndarray,
        output_path: str,
        reference_nifti: str = None
    ):
        """
        Save segmentation as NIfTI file
        
        Args:
            segmentation: Binary segmentation mask
            output_path: Output file path (.nii.gz)
            reference_nifti: Optional reference NIfTI to copy header/affine
        """
        if reference_nifti:
            ref_nii = nib.load(reference_nifti)
            affine = ref_nii.affine
            header = ref_nii.header
        else:
            affine = np.eye(4)
            header = None
        
        nii = nib.Nifti1Image(segmentation.astype(np.uint8), affine, header)
        nib.save(nii, output_path)
        print(f"💾 Saved segmentation to: {output_path}")


def quick_predict(
    model_path: str,
    flair_path: str,
    t1_path: str,
    t2_path: str,
    output_path: str = None,
    threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Quick one-line prediction function
    
    Args:
        model_path: Path to best_model.pth
        flair_path: Path to FLAIR NIfTI
        t1_path: Path to T1 NIfTI
        t2_path: Path to T2 NIfTI
        output_path: Optional path to save segmentation
        threshold: Segmentation threshold (default: 0.5)
        
    Returns:
        Prediction results dictionary
    """
    predictor = MSLesionPredictor(model_path)
    results = predictor.predict(flair_path, t1_path, t2_path, threshold=threshold)
    
    if output_path:
        predictor.save_prediction(
            results['segmentation'],
            output_path,
            reference_nifti=flair_path
        )
    
    return results


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 5:
        print("Usage: python inference.py <model_path> <flair> <t1> <t2> [output.nii.gz]")
        print("\nExample:")
        print("  python inference.py best_model.pth flair.nii.gz t1.nii.gz t2.nii.gz output.nii.gz")
        sys.exit(1)
    
    model_path = sys.argv[1]
    flair_path = sys.argv[2]
    t1_path = sys.argv[3]
    t2_path = sys.argv[4]
    output_path = sys.argv[5] if len(sys.argv) > 5 else "prediction.nii.gz"
    
    print("="*80)
    print("MS Lesion Segmentation Inference")
    print("="*80)
    
    results = quick_predict(
        model_path=model_path,
        flair_path=flair_path,
        t1_path=t1_path,
        t2_path=t2_path,
        output_path=output_path
    )
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Lesion Volume: {results['lesion_volume_ml']:.2f} mL")
    print(f"Number of Lesions: {results['num_lesions']}")
    print(f"Segmentation saved to: {output_path}")
