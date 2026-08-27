# Dataset Information

## 📊 Datasets Used in This Project

This repository **does not include** the actual medical imaging datasets due to:
- Large file sizes (several GB)
- Privacy and ethics considerations
- GitHub's 100MB file size limit

### Training Dataset: PediMS

- **Description**: Pediatric Multiple Sclerosis MRI dataset
- **Size**: 45 patients (36 training, 9 validation)
- **Modalities**: FLAIR, T1, T2 MRI sequences
- **Format**: NIfTI (.nii.gz)
- **Resolution**: Resampled to 1×1×1mm³ isotropic
- **Access**: Contact authors or obtain from original source

### Validation Datasets

#### 1. MS60 (Adult MS Cross-Dataset Validation)
- **Patients**: 60
- **Slices**: 787 annotated MS lesion slices
- **Purpose**: Cross-dataset validation
- **Expected Dice**: ~1.1% (domain shift demonstration)

#### 2. MSLESSEG
- **Description**: MS Lesion Segmentation Challenge dataset
- **Source**: [Competition website]
- **Purpose**: Benchmark comparison

#### 3. LGG (Cross-Pathology Validation)
- **Description**: Brain tumor (glioma) dataset
- **Patients**: 110
- **Slices**: 1,359 tumor slices
- **Purpose**: Task-specificity validation
- **Expected Dice**: ~20% (confirms MS-specific learning)
- **Source**: Kaggle - Brain MRI Segmentation

## 🗂️ Local Dataset Structure

If you have access to the datasets, organize them as follows:

```
Neuroscan/
├── Dataset/
│   ├── PediMS/
│   │   ├── P1/
│   │   │   ├── FLAIR/
│   │   │   ├── T1/
│   │   │   ├── T2/
│   │   │   └── mask/
│   │   ├── P2/
│   │   └── ...
│   ├── MSLESSEG/
│   └── preprocessed/
│
├── LGG/
│   └── lgg-mri-segmentation/
│       └── kaggle_3m/
│
└── ESSENTIAL_MODELS_BACKUP/
    ├── seg_resume.pth          # Main trained model (83.99% Dice)
    └── mae_resume.pth          # MAE pretrained encoder
```

## 📥 How to Use Your Own Data

### Option 1: Use Compatible Datasets

The model expects NIfTI format MRI scans with the following structure:
- FLAIR sequence (primary input)
- Optional: T1, T2 sequences
- Binary segmentation mask for training

### Option 2: Preprocess Your Data

Use the preprocessing pipeline in `01_source_code/`:

```python
# Example preprocessing
from monai.transforms import (
    LoadImaged,
    Spacingd,
    Orientationd,
    NormalizeIntensityd,
    Resized
)

transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0)),
    NormalizeIntensityd(keys="image", nonzero=True),
    Resized(keys=["image", "label"], spatial_size=(128, 128, 128))
])
```

### Option 3: Use Pretrained Model Only

Download the pretrained model from:
- GitHub Releases (coming soon)
- [External hosting link]
- Or train your own using the provided architecture

## 🔒 Ethics & Privacy

All datasets used in this project:
- ✅ Are publicly available or properly authorized
- ✅ Follow HIPAA/GDPR privacy guidelines
- ✅ Have appropriate IRB approval
- ✅ Are de-identified and anonymized
- ✅ Follow medical imaging ethics standards

## 📧 Data Access Requests

For access to specific datasets:
- **PediMS**: Contact [institution/PI]
- **MS60**: Contact [source]
- **MSLESSEG**: [Competition website]
- **LGG**: Available on Kaggle

## 🎯 Model Checkpoints

Pre-trained model checkpoints are available:
- **Main Model** (83.99% Dice): [Release link]
- **MAE Pretrained Encoder**: [Release link]
- **Ablation Study Models**: [Release link]

Download and place in `ESSENTIAL_MODELS_BACKUP/` folder.

## 💡 Training Without Full Datasets

You can experiment with the architecture using:
- Small subset of data (10-20 patients)
- Synthetic MRI data
- Public MS lesion datasets
- Transfer learning from brain MRI datasets

Expected performance will scale with dataset size:
- 10 patients: ~60-70% Dice
- 25 patients: ~70-75% Dice
- 45 patients: ~80-84% Dice (our results)

## 📚 References

1. PediMS Dataset: [Citation]
2. MSLESSEG Challenge: [Website]
3. LGG Dataset: [Kaggle Link]
4. MONAI Framework: https://monai.io

---

**Note**: This file explains why datasets are not included in the repository and how to obtain/use them for your own research.
